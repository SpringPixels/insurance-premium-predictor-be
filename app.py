from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd



from schema.user_input import UserInput
from ml_model.predict import predict_and_explain, MODEL_VERSION, CompareRequest
from schema.role import RoleUpdate
from schema.prediction_response import PredictionResponse
from config.database import Base, engine, get_db
from ml_model.db_models import PredictionLog
from contextlib import asynccontextmanager

from fastapi.security import OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm
from ml_model.db_models import User
from schema.auth import UserSignup, UserLogin, Token
from config.security import hash_password, verify_password, create_access_token, decode_access_token
from jose import JWTError

from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

from sqlalchemy import func


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This securely runs your table creation asynchronously on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

# Pass the lifespan function to your FastAPI instance
app = FastAPI(lifespan=lifespan)


# Define the origins that are allowed to make requests to your API
origins = [
    "http://localhost:4200",  # Default Angular dev server port
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# human readable
@app.get('/')
def home():
    return{'message': 'Insurance Premium Prediction API' }



# machine readable
@app.get('/health')
def health_check():
    return {
        'status' : 'OK',
        'version' : MODEL_VERSION,
        'model_loaded' : model is not None
    }

# Auth
@app.post("/signup")
async def signup(data: UserSignup, db: AsyncSession = Depends(get_db)):
    result= await db.execute(select(User).filter(User.email == data.email))
    existing = result.scalars().first()

    if existing:
        return JSONResponse(status_code=400, content={"error": "Email already registered"})


    # Convert email to lowercase to prevent bypass tricks (e.g., Admin@test.com)
    email_lower = data.email.lower()
    
    # If the email contains the word 'admin', make them an admin. Otherwise, make them a normal user.
    if "admin" in email_lower:
        assigned_role = "admin"
    else:
        assigned_role = "user"
    user = User(email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    await db.commit()
    return {"message": "User created successfully"}

@app.post("/login", response_model=Token)
async def login(form_data:OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalars().first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

    token = create_access_token({"sub": user.email, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_access_token(token)
        email = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def require_admin(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@app.get("/me")
async def read_current_user(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role
    }


# Core Prediction
@app.post('/predict/explain', response_model= PredictionResponse)
async def predict_premium_with_explanation(data: UserInput, db: AsyncSession =Depends(get_db),current_user: User = Depends(get_current_user)):

    user_input = {
        'income_lpa' : float(data.income_lpa),
        'occupation' : str(data.occupation),
        'bmi' : float(data.bmi),
        'age_group' :str(data.age_group),
        'lifestyle_risk' : str(data.lifestyle_risk),
        'city_tier' : int(data.city_tier)
    }

    

    try:
        result = predict_and_explain(user_input)   # returns the full dict already — prediction + explanation

        PREMIUM_MAP = {
            "Low": 5000,
            "Medium": 10000,
            "High": 18000,
            "Very High": 30000
        }
        chosen_category =result["prediction_results"]["predicted_category"]
        log = PredictionLog(
            **user_input,
            user_id=current_user.id,
            predicted_category=chosen_category,
            predicted_premium=PREMIUM_MAP[chosen_category]
        )

        db.add(log)
        await db.commit()

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# History/"My Predictions"
@app.get("/predictions/me")
async def get_my_predictions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(PredictionLog).filter(PredictionLog.user_id == current_user.id)
    )
    logs = result.scalars().all()
    return logs

#Get /predictions/{id}
@app.get("/predictions/{prediction_id}")
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


# DELETE /predictions/{id}
@app.delete("/predictions/{prediction_id}")
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

# POST/predict/compare
@app.post("/predict/compare")
async def compare_predictions(
    payload: CompareRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        result_a = run_prediction(payload.scenario_a)  # replace with your actual predict function
        result_b = run_prediction(payload.scenario_b)

        difference = result_a["predicted_premium"] - result_b["predicted_premium"]

        return {
            "scenario_a": result_a,
            "scenario_b": result_b,
            "difference": difference
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# PDF/Export
@app.get("/predictions/{prediction_id}/pdf")
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

    # Build the PDF in memory
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


# Admin
@app.get("/admin/predictions")
async def get_all_predictions(db: AsyncSession = Depends(get_db), admin: User = Depends(require_admin)):
    result = await db.execute(select(PredictionLog))
    logs = result.scalars().all()
    return logs

# Aggregate stats
@app.get("/admin/predictions/stats")
async def get_prediction_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    total_result = await db.execute(select(func.count(PredictionLog.id)))
    total_predictions = total_result.scalar()

    avg_result = await db.execute(select(func.avg(PredictionLog.predicted_premium)))
    avg_premium = avg_result.scalar()

    by_city_result = await db.execute(
        select(PredictionLog.city_tier, func.avg(PredictionLog.predicted_premium))
        .group_by(PredictionLog.city_tier)
    )
    by_city = [{"city_tier": row[0], "avg_premium": row[1]} for row in by_city_result.all()]

    return {
        "total_predictions": total_predictions,
        "average_premium": avg_premium,
        "average_by_city_tier": by_city
    }


# list all user 
@app.get("/admin/users")
async def get_all_users(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return [{"id": u.id, "email": u.email, "role": u.role} for u in users]

# promote/demote a user
@app.patch("/admin/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    payload: RoleUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")

    user.role = payload.role
    await db.commit()
    await db.refresh(user)

    return {"message": f"User {user.email} role updated to {user.role}"}

# Removes a user
@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(user)
    await db.commit()
    return {"message": f"User {user.email} deleted"}

