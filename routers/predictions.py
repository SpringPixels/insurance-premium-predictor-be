import io
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from typing import List

from schema.predictions import UserInput, PredictionResponse, CompareResponse
from schema.auth import StandardResponse
from ml_model.predict import predict_and_explain, CompareRequest
from config.database import get_db
from ml_model.db_models import PredictionLog, User
from dependencies import get_current_user

def run_prediction(user_input):
    res = predict_and_explain(user_input)
    PREMIUM_MAP = {
        "Low": 5000,
        "Medium": 10000,
        "High": 18000,
        "Very High": 30000
    }
    cat = res["prediction_results"]["predicted_category"]
    res["predicted_premium"] = PREMIUM_MAP[cat]
    return res

router = APIRouter(tags=["predictions"])

@router.post('/predict/explain', response_model=PredictionResponse)
async def predict_premium_with_explanation(
    data: UserInput, 
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    user_input = {
        'income_lpa': float(data.income_lpa),
        'occupation': str(data.occupation),
        'bmi': float(data.bmi),
        'age_group': str(data.age_group),
        'lifestyle_risk': str(data.lifestyle_risk),
        'city_tier': int(data.city_tier)
    }

    try:
        result = predict_and_explain(user_input)

        PREMIUM_MAP = {
            "Low": 5000,
            "Medium": 10000,
            "High": 18000,
            "Very High": 30000
        }
        chosen_category = result["prediction_results"]["predicted_category"]
        log = PredictionLog(
            **user_input,
            user_id=current_user.id,
            predicted_category=chosen_category,
            predicted_premium=PREMIUM_MAP[chosen_category]
        )

        current_user.age = data.age
        current_user.weight = data.weight
        current_user.height = data.height
        current_user.is_smoker = data.smoker
        current_user.occupation = data.occupation
        current_user.income_lpa = data.income_lpa
        current_user.city = data.city
        db.add(current_user)
        db.add(log)
        await db.commit()

        # Added mapping to explicitly follow response model shape, handling extra metadata gracefully.
        return PredictionResponse(
            predicted_category=chosen_category,
            confidence=result["prediction_results"]["confidence_score"],
            class_probabilities=result["prediction_results"]["all_class_probabilities"],
            model_metadata=result.get("model_metadata"),
            prediction_results=result.get("prediction_results"),
            explainable_ai=result.get("explainable_ai"),
            predicted_premium=PREMIUM_MAP[chosen_category]
        )

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction model returned an unexpected result structure (missing key: {e}). Please try again."
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input value: {e}")
    except Exception as e:
        # Log the full traceback server-side but return a safe message to the client
        import traceback
        print(f"[ERROR] /predict/explain failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="The prediction engine encountered an unexpected error. Please try again later."
        )

@router.get("/predictions/me")
async def get_my_predictions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # This might need a proper response model for a list of PredictionLogs, but returning the objects works for now
    result = await db.execute(
        select(PredictionLog).filter(PredictionLog.user_id == current_user.id)
    )
    logs = result.scalars().all()
    return logs

@router.get("/predictions/{prediction_id}")
async def get_prediction_by_id(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(PredictionLog).filter(PredictionLog.id == prediction_id)
    )
    log = result.scalars().first()

    if not log:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if log.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view this prediction")

    return log

@router.delete("/predictions/{prediction_id}", response_model=StandardResponse)
async def delete_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(PredictionLog).filter(PredictionLog.id == prediction_id)
    )
    log = result.scalars().first()

    if not log:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if log.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this prediction")

    await db.delete(log)
    await db.commit()
    return {"message": "Prediction deleted successfully"}

@router.post("/predict/compare", response_model=CompareResponse)
async def compare_predictions(
    payload: CompareRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        # Pydantic v2 `.model_dump()` is used here
        result_a = run_prediction(payload.scenario_a.model_dump())
        result_b = run_prediction(payload.scenario_b.model_dump())

        difference = result_a["predicted_premium"] - result_b["predicted_premium"]

        return {
            "scenario_a": result_a,
            "scenario_b": result_b,
            "difference": difference
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] /predict/compare failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Comparison failed due to an internal error. Please try again later.")

@router.get("/predictions/{prediction_id}/pdf")
async def get_prediction_pdf(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(select(PredictionLog).filter(PredictionLog.id == prediction_id))
    log = result.scalars().first()

    if not log:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if log.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view this prediction")

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, "Insurance Premium Prediction Summary")

    p.setFont("Helvetica", 12)
    y = 760
    fields = [
        f"Prediction ID: {log.id}",
        f"Date: {log.created_at.strftime('%Y-%m-%d %H:%M')}",
        f"Age Group: {log.age_group}",
        f"BMI: {log.bmi}",
        f"Occupation: {log.occupation}",
        f"Lifestyle Risk: {log.lifestyle_risk}",
        f"City Tier: {log.city_tier}",
        f"Income (LPA): {log.income_lpa}",
        f"Predicted Risk Category: {log.predicted_category}",
    ]
    for line in fields:
        p.drawString(50, y, line)
        y -= 25

    p.showPage()
    p.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=prediction_{prediction_id}.pdf"}
    )
