from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func, extract
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime as dt_module

from schema.admin_dashboard import AdminDashboardResponse, RevenuePoint, PaidMember, RiskAnalysisResponse, ClaimCreate, ClaimResponse
from schema.role import RoleUpdate, StatusUpdate
from config.database import get_db
from ml_model.db_models import User, PredictionLog, Payment, PaymentStatus, ContactUs, ActivityLog, Claim
from dependencies import require_admin
from ml_model.segmentation import predict_segment
from ml_model.risk_analysis import analyze_risk

router = APIRouter(tags=["admin"])

@router.get("/admin/predictions")
async def get_all_predictions(db: AsyncSession = Depends(get_db), admin: User = Depends(require_admin)):
    result = await db.execute(select(PredictionLog))
    logs = result.scalars().all()
    return logs

@router.get("/admin/predictions/stats")
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

@router.get("/admin/users")
async def get_all_users(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return [{"id": u.id, "email": u.email, "role": u.role, "is_active": u.is_active} for u in users]

@router.patch("/admin/users/{user_id}/role")
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

@router.patch("/admin/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    payload: StatusUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change your own status")

    user.is_active = payload.is_active
    await db.commit()
    await db.refresh(user)

    status_str = "activated" if user.is_active else "deactivated"
    return {"message": f"User {user.email} has been {status_str}"}

@router.delete("/admin/users/{user_id}")
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

@router.get("/admin/dashboard", response_model=AdminDashboardResponse)
async def get_admin_dashboard(db: AsyncSession = Depends(get_db), admin: User = Depends(require_admin)):
    result = await db.execute(
        select(func.sum(Payment.amount)).filter(Payment.status == PaymentStatus.SUCCEEDED)
    )
    total_amount = result.scalar() or 0
    total_revenue = float(total_amount)

    result = await db.execute(select(func.count(User.id)))
    total_users = result.scalar() or 0

    now = dt_module.utcnow()
    result = await db.execute(
        select(func.count(User.id)).filter(
            extract('year', User.created_at) == now.year,
            extract('month', User.created_at) == now.month,
        )
    )
    new_users_this_month = result.scalar() or 0

    result = await db.execute(
        select(Payment).filter(Payment.status == PaymentStatus.SUCCEEDED)
    )
    payments_objs = result.scalars().all()

    monthly_totals: dict[str, float] = {}
    paid_members_dict = {}
    
    for p in payments_objs:
        if p.created_at is None:
            continue
        key = p.created_at.strftime("%Y-%m")
        amt = float(p.amount)
        monthly_totals[key] = monthly_totals.get(key, 0) + amt
        
        if p.user_id not in paid_members_dict:
            paid_members_dict[p.user_id] = 0.0
        paid_members_dict[p.user_id] += amt

    revenue_trend = [
        RevenuePoint(month=month, amount=round(amt, 2))
        for month, amt in sorted(monthly_totals.items())
    ][-6:]
    
    paid_members = []
    if paid_members_dict:
        paid_users_result = await db.execute(select(User).filter(User.id.in_(list(paid_members_dict.keys()))))
        paid_users = paid_users_result.scalars().all()
        paid_members = [
            PaidMember(user_id=u.id, email=u.email, total_paid=round(paid_members_dict[u.id], 2))
            for u in paid_users
        ]

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
            print("[WARN] Segmentation model file not found — segment_breakdown will be empty.")
        except Exception as seg_err:
            print(f"[WARN] Segment prediction failed for log {log.id}: {seg_err}")

    return AdminDashboardResponse(
        total_revenue=round(total_revenue, 2),
        total_users=total_users,
        new_users_this_month=new_users_this_month,
        revenue_trend=revenue_trend,
        segment_breakdown=segment_counts,
        paid_members=paid_members,
    )

@router.get("/admin/users/{user_id}/analysis", response_model=RiskAnalysisResponse)
async def get_user_risk_analysis(user_id: int, db: AsyncSession = Depends(get_db), admin: User = Depends(require_admin)):
    # 1. Get user and their latest prediction log
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    pred_result = await db.execute(
        select(PredictionLog).filter(PredictionLog.user_id == user_id).order_by(PredictionLog.created_at.desc())
    )
    latest_pred = pred_result.scalars().first()
    
    base_premium = latest_pred.predicted_premium if latest_pred else 0.0
    
    # 2. Get total exercises completed in the last 30 days
    now = dt_module.utcnow()
    thirty_days_ago = now.date() - __import__('datetime').timedelta(days=30)
    act_result = await db.execute(
        select(func.count(ActivityLog.id)).filter(
            ActivityLog.user_id == user_id, 
            ActivityLog.completed == True,
            ActivityLog.date >= thirty_days_ago
        )
    )
    total_exercises = act_result.scalar() or 0
    
    # 3. Get all claims
    claim_result = await db.execute(select(Claim).filter(Claim.user_id == user_id).order_by(Claim.created_at.desc()))
    claims = claim_result.scalars().all()
    total_claims_amount = sum(c.amount for c in claims)
    
    # 4. Run Risk Analysis Model
    user_data = {
        'age': user.age or 30,
        'bmi': latest_pred.bmi if latest_pred else 22.0,
        'income_lpa': user.income_lpa or 10.0,
        'city_tier': latest_pred.city_tier if latest_pred else 2,
        'lifestyle_risk': latest_pred.lifestyle_risk if latest_pred else "Moderate",
        'total_exercises': total_exercises,
        'total_claims_amount': total_claims_amount,
        'base_premium': base_premium
    }
    
    analysis = analyze_risk(user_data)
    renewal_premium = base_premium * analysis['renewal_multiplier']
    
    return RiskAnalysisResponse(
        risky_behaviour_rate=analysis['risky_behaviour_rate'],
        renewal_multiplier=analysis['renewal_multiplier'],
        base_premium=base_premium,
        predicted_renewal_premium=round(renewal_premium, 2),
        total_exercises=total_exercises,
        total_claims_amount=total_claims_amount,
        claims=[ClaimResponse(id=c.id, amount=c.amount, description=c.description, date=c.date) for c in claims]
    )

@router.post("/admin/users/{user_id}/claims", response_model=ClaimResponse)
async def add_user_claim(user_id: int, payload: ClaimCreate, db: AsyncSession = Depends(get_db), admin: User = Depends(require_admin)):
    result = await db.execute(select(User).filter(User.id == user_id))
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="User not found")
        
    claim = Claim(
        user_id=user_id,
        amount=payload.amount,
        description=payload.description
    )
    db.add(claim)
    await db.commit()
    await db.refresh(claim)
    return ClaimResponse(id=claim.id, amount=claim.amount, description=claim.description, date=claim.date)

@router.get("/admin/contact-us")
async def get_all_contact_submissions(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(ContactUs).order_by(ContactUs.created_at.desc()))
    submissions = result.scalars().all()
    return submissions

class PremiumUpdate(BaseModel):
    base_premium: float

@router.patch("/admin/users/{user_id}/premium")
async def update_base_premium(
    user_id: int, 
    payload: PremiumUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(
        select(PredictionLog)
        .filter(PredictionLog.user_id == user_id)
        .order_by(PredictionLog.created_at.desc())
    )
    latest_pred = result.scalars().first()
    
    if not latest_pred:
        raise HTTPException(status_code=400, detail="User does not have an AI prediction yet. They must run the calculator first.")
        
    latest_pred.predicted_premium = int(payload.base_premium)
    await db.commit()
    return {"message": "Base premium updated successfully", "new_premium": latest_pred.predicted_premium}

from ml_model.db_models import PricingSettings

class PricingSettingsUpdate(BaseModel):
    base_flat_fee: int
    low_penalty: int
    medium_penalty: int
    high_penalty: int
    very_high_penalty: int

@router.get("/admin/settings/pricing")
async def get_pricing_settings(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(PricingSettings).limit(1))
    settings = result.scalar()
    if not settings:
        settings = PricingSettings()
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings

@router.patch("/admin/settings/pricing")
async def update_pricing_settings(
    payload: PricingSettingsUpdate, 
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(PricingSettings).limit(1))
    settings = result.scalar()
    if not settings:
        settings = PricingSettings()
        db.add(settings)
    
    settings.base_flat_fee = payload.base_flat_fee
    settings.low_penalty = payload.low_penalty
    settings.medium_penalty = payload.medium_penalty
    settings.high_penalty = payload.high_penalty
    settings.very_high_penalty = payload.very_high_penalty
    await db.commit()
    return {"message": "Settings updated"}
