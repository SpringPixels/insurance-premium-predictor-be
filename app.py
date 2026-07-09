
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd


from schema.admin_dashboard import AdminDashboardResponse, RevenuePoint
from schema.user_input import UserInput, PaymentCreate, StepBatch, ContactUsCreate
from ml_model.predict import predict_and_explain, MODEL_VERSION, CompareRequest
from ml_model.segmentation import train_segments, predict_segment
from schema.segment import SegmentResponse, TrainSegmentsResponse
from schema.role import RoleUpdate
from schema.prediction_response import PredictionResponse
from config.database import Base, engine, get_db
from ml_model.db_models import PredictionLog, ContactUs, ActivityLog, Payment, PaymentStatus
from schema.activity import ActivityCheckIn, ActivityLogResponse, StreakResponse
from datetime import datetime as dt_module
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
import os

from sqlalchemy import func, extract

import hmac, hashlib

import razorpay

RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "your_key_id")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "fallback_secret_if_missing")

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


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
    result = await db.execute(select(User).filter(User.email == data.email))
    existing = result.scalars().first()

    if existing:
        return JSONResponse(status_code=400, content={"error": "Email already registered"})

    user = User(
        full_name=data.full_name,
        email=data.email,
        phone_no=data.phone_no,
        hashed_password=hash_password(data.password)
        # role intentionally omitted — always defaults to "user" per the model
    )
    db.add(user)
    await db.commit()
    return {"message": "User created successfully"}

@app.post("/login", response_model=Token)
async def login(form_data:OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalars().first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

    token = create_access_token({"sub": user.email, "role": user.role, "full_name":user.full_name})
    return {"access_token": token,
            "token_type": "bearer", 
            "user":{
               "id":user.id,
               "full_name":user.full_name,
               "email":user.email,
               "phone_no":user.phone_no,
               "role":user.role
            }}

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
        "id":current_user.id,
        "full_name":current_user.full_name,
        "email":current_user.email,
        "phone_no":current_user.phone_no,
        "role":current_user.role
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

# payment api design
@app.post("/payments/create")
async def create_payment(data: PaymentCreate, db: AsyncSession = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    order = razorpay_client.order.create({
        "amount": data.amount, "currency": "INR", "payment_capture": 1
    })
    payment = Payment(user_id=current_user.id, amount=data.amount,
                       provider_order_id=order["id"], status=PaymentStatus.PENDING)
    db.add(payment)
    await db.commit()
    return {"order_id": order["id"], "amount": data.amount, "key": RAZORPAY_KEY_ID}

@app.get("/payments/{payment_id}")
async def get_payment_status(payment_id: int, db: AsyncSession = Depends(get_db),
                              current_user: User = Depends(get_current_user)):
    # ownership check like your other GETs, return current .status
    result = await db.execute(select(Payment).filter(Payment.id == payment_id))
    payment = result.scalars().first()

    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")

    if payment.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view this payment")

    return {
        "id": payment.id,
        "amount": payment.amount,
        "currency": payment.currency,
        "status": payment.status,
        "created_at": payment.created_at,
    }

@app.post("/payments/webhook")
async def payment_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    # 1. verify signature (critical — don't trust unsigned webhook bodies)
    # 2. look up Payment by provider_order_id
    # 3. update .status based on event type, set provider_payment_id
    # 4. return 200 quickly
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    expected_signature = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    payload = await request.json()
    event = payload.get("event")
    entity = payload["payload"]["payment"]["entity"]
    order_id = entity.get("order_id")

    result = await db.execute(select(Payment).filter(Payment.provider_order_id == order_id))
    payment = result.scalars().first()

    if not payment:
        # Don't 404/500 here — provider will retry forever. Log and 200.
        return JSONResponse(status_code=200, content={"message": "order not found, ignored"})

    if event == "payment.captured":
        payment.status = PaymentStatus.SUCCEEDED
        payment.provider_payment_id = entity["id"]
    elif event == "payment.failed":
        payment.status = PaymentStatus.FAILED
    elif event == "refund.processed":
        payment.status = PaymentStatus.REFUNDED

    await db.commit()
    return {"message": "ok"}

# POST /steps/log — client sends incremental step counts periodically
@app.post("/steps/log")
async def log_steps(
    payload: StepBatch,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    target_date = payload.date or date.today()

    result = await db.execute(
        select(StepLog).filter(
            StepLog.user_id == current_user.id,
            StepLog.date == target_date
        )
    )
    existing = result.scalars().first()

    if existing:
        existing.steps += payload.steps
        existing.recorded_at = datetime.utcnow()
    else:
        existing = StepLog(
            user_id=current_user.id,
            steps=payload.steps,
            date=target_date
        )
        db.add(existing)

    await db.commit()
    await db.refresh(existing)
    return {"date": existing.date, "total_steps": existing.steps}


# GET /steps/today
@app.get("/steps/today")
async def get_today_steps(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(StepLog).filter(
            StepLog.user_id == current_user.id,
            StepLog.date == date.today()
        )
    )
    log = result.scalars().first()
    return {"date": date.today(), "steps": log.steps if log else 0}


# GET /steps/me — history, like /predictions/me
@app.get("/steps/me")
async def get_my_steps(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(StepLog)
        .filter(StepLog.user_id == current_user.id)
        .order_by(StepLog.date.desc())
    )
    return result.scalars().all()

#contact us
# Public — anyone can submit, no login required
@app.post("/contact-us")
async def submit_contact_form(data: ContactUsCreate, db: AsyncSession = Depends(get_db)):
    contact = ContactUs(
        name=data.name,
        email=data.email,
        phone_no=data.phone_no,
        subject=data.subject,
        message=data.message
    )
    db.add(contact)
    await db.commit()
    return {"message": "Your message has been submitted successfully"}


# Admin-only — view all submissions
@app.get("/admin/contact-us")
async def get_all_contact_submissions(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(ContactUs).order_by(ContactUs.created_at.desc()))
    submissions = result.scalars().all()
    return submissions


# for customer segmentation
@app.post("/segment/train", response_model=TrainSegmentsResponse)
async def train_segmentation_model(db: AsyncSession = Depends(get_db), admin: User = Depends(require_admin)):
    result = await db.execute(select(PredictionLog))
    logs = result.scalars().all()

    if len(logs) < 10:
        raise HTTPException(status_code=400, detail="Not enough data to train segments (need 10+ records)")

    df = pd.DataFrame([{
        "income_lpa": log.income_lpa,
        "bmi": log.bmi,
        "age_group": log.age_group,
        "lifestyle_risk": log.lifestyle_risk,
        "city_tier": log.city_tier,
    } for log in logs])

    result = train_segments(df)
    return result


@app.post("/segment/predict", response_model=SegmentResponse)
async def get_user_segment(data: dict, current_user: User = Depends(get_current_user)):
    try:
        return predict_segment(data)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Segmentation model not trained yet")



# For activity
@app.post("/activity/checkin")
async def check_in_activity(data: ActivityCheckIn, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    today = date.today()

    result = await db.execute(
        select(ActivityLog).filter(ActivityLog.user_id == current_user.id, ActivityLog.date == today)
    )
    existing = result.scalars().first()
    if existing:
        return {"message": "Already checked in today"}

    log = ActivityLog(user_id=current_user.id, date=today, completed=True, activity_type=data.activity_type)
    db.add(log)
    await db.commit()
    return {"message": "Checked in for today"}


@app.get("/activity/streak", response_model=StreakResponse)
async def get_streak(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(
        select(ActivityLog).filter(ActivityLog.user_id == current_user.id).order_by(ActivityLog.date.desc())
    )
    logs = result.scalars().all()
    log_dates = {log.date for log in logs}

    # current streak: count back from today
    current_streak = 0
    day = date.today()
    while day in log_dates:
        current_streak += 1
        day -= timedelta(days=1)

    # longest streak overall
    longest_streak = 0
    streak = 0
    sorted_dates = sorted(log_dates)
    prev = None
    for d in sorted_dates:
        if prev and (d - prev).days == 1:
            streak += 1
        else:
            streak = 1
        longest_streak = max(longest_streak, streak)
        prev = d

    last_7 = [
        ActivityLogResponse(date=log.date, completed=log.completed, activity_type=log.activity_type)
        for log in logs[:7]
    ]

    if current_streak >= 7:
        message = f"{current_streak} days strong — great consistency!"
    elif current_streak >= 3:
        message = f"{current_streak} days in a row, keep going!"
    elif current_streak == 0:
        message = "No check-in yet today — get moving!"
    else:
        message = f"Day {current_streak} — nice start!"

    return StreakResponse(
        current_streak=current_streak,
        longest_streak=longest_streak,
        last_7_days=last_7,
        message=message,
    )

# for admin_dashboard
@app.get("/admin/dashboard", response_model=AdminDashboardResponse)
async def get_admin_dashboard(db: AsyncSession = Depends(get_db), admin: User = Depends(require_admin)):
    # --- Total revenue (only successful payments, stored in paise) ---
    result = await db.execute(
        select(func.sum(Payment.amount)).filter(Payment.status == PaymentStatus.SUCCEEDED)
    )
    total_paise = result.scalar() or 0
    total_revenue = total_paise / 100

    # --- Total users ---
    result = await db.execute(select(func.count(User.id)))
    total_users = result.scalar() or 0

    # --- New users this month ---
    now = dt_module.utcnow()
    result = await db.execute(
        select(func.count(User.id)).filter(
            extract('year', User.created_at) == now.year,
            extract('month', User.created_at) == now.month,
        )
    )
    new_users_this_month = result.scalar() or 0

    # --- Revenue trend: last 6 months ---
    result = await db.execute(
        select(Payment.created_at, Payment.amount).filter(Payment.status == PaymentStatus.SUCCEEDED)
    )
    payments = result.all()

    monthly_totals: dict[str, float] = {}
    for created_at, amount in payments:
        if created_at is None:
            continue
        key = created_at.strftime("%Y-%m")
        monthly_totals[key] = monthly_totals.get(key, 0) + (amount / 100)

    revenue_trend = [
        RevenuePoint(month=month, amount=round(amt, 2))
        for month, amt in sorted(monthly_totals.items())
    ][-6:]

    # --- Segment breakdown: use latest PredictionLog per user, run through trained segmentation model ---
    result = await db.execute(
        select(PredictionLog).order_by(PredictionLog.user_id, PredictionLog.created_at.desc())
    )
    logs = result.scalars().all()

    seen_users = set()
    segment_counts = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0}

    for log in logs:
        if log.user_id in seen_users or log.user_id is None:
            continue
        seen_users.add(log.user_id)
        try:
            seg = predict_segment({
                "income_lpa": log.income_lpa,
                "bmi": log.bmi,
                "age_group": log.age_group,
                "lifestyle_risk": log.lifestyle_risk,
                "city_tier": log.city_tier,
            })
            segment_counts[seg["segment_label"]] += 1
        except FileNotFoundError:
            pass

    return AdminDashboardResponse(
        total_revenue=round(total_revenue, 2),
        total_users=total_users,
        new_users_this_month=new_users_this_month,
        revenue_trend=revenue_trend,
        segment_breakdown=segment_counts,
    )