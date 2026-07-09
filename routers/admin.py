from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, extract
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime as dt_module

from schema.admin_dashboard import AdminDashboardResponse, RevenuePoint
from schema.role import RoleUpdate, StatusUpdate
from config.database import get_db
from ml_model.db_models import User, PredictionLog, Payment, PaymentStatus, ContactUs
from dependencies import require_admin
from ml_model.segmentation import predict_segment

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
    total_paise = result.scalar() or 0
    total_revenue = total_paise / 100

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
    )

@router.get("/admin/contact-us")
async def get_all_contact_submissions(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin)
):
    result = await db.execute(select(ContactUs).order_by(ContactUs.created_at.desc()))
    submissions = result.scalars().all()
    return submissions
