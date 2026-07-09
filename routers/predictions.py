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

async def get_pricing_settings(db: AsyncSession):
    from ml_model.db_models import PricingSettings
    result = await db.execute(select(PricingSettings).limit(1))
    settings = result.scalar()
    if not settings:
        settings = PricingSettings()
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return {
        "base_flat_fee": settings.base_flat_fee,
        "premium_map": {
            "Low": settings.low_penalty,
            "Medium": settings.medium_penalty,
            "High": settings.high_penalty,
            "Very High": settings.very_high_penalty
        }
    }

async def run_prediction(user_input, db: AsyncSession):
    res = predict_and_explain(user_input)
    settings = await get_pricing_settings(db)
    base_premium = int(settings['base_flat_fee'] + (user_input.get('age', 30) * 100) + (user_input.get('bmi', 22.0) * 50) + (user_input.get('income_lpa', 10.0) * 20))
    cat = res["prediction_results"]["predicted_category"]
    res["predicted_premium"] = base_premium + settings['premium_map'].get(cat, 0)
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

        settings = await get_pricing_settings(db)
        base_premium = int(settings['base_flat_fee'] + (data.age * 100) + (data.bmi * 50) + (data.income_lpa * 20))
        chosen_category = result["prediction_results"]["predicted_category"]
        final_premium = base_premium + settings['premium_map'].get(chosen_category, 0)
        
        log = PredictionLog(
            **user_input,
            user_id=current_user.id,
            predicted_category=chosen_category,
            predicted_premium=final_premium
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
            predicted_premium=final_premium
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Pydantic v2 `.model_dump()` is used here
        result_a = await run_prediction(payload.scenario_a.model_dump(), db)
        result_b = await run_prediction(payload.scenario_b.model_dump(), db)

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
