from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date, datetime, timedelta
from typing import List

from schema.activity import StepBatch, StepLogResponse, ActivityCheckIn, StreakResponse, ActivityLogResponse, StandardResponse
from config.database import get_db
from ml_model.db_models import User, StepLog, ActivityLog
from dependencies import get_current_user

router = APIRouter(tags=["activity"])

@router.post("/steps/log", response_model=StepLogResponse)
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

@router.get("/steps/today", response_model=StepLogResponse)
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
    return {"date": date.today(), "total_steps": log.steps if log else 0}

@router.get("/steps/me", response_model=List[StepLogResponse])
async def get_my_steps(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(StepLog)
        .filter(StepLog.user_id == current_user.id)
        .order_by(StepLog.date.desc())
    )
    # the response model maps log.steps correctly due to matching fields if we return objects, wait,
    # StepLog model has 'steps' but StepLogResponse expects 'total_steps'. We need to map it manually or change the schema.
    logs = result.scalars().all()
    return [{"date": log.date, "total_steps": log.steps} for log in logs]

@router.post("/activity/checkin", response_model=StandardResponse)
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

@router.get("/activity/streak", response_model=StreakResponse)
async def get_streak(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(
        select(ActivityLog).filter(ActivityLog.user_id == current_user.id).order_by(ActivityLog.date.desc())
    )
    logs = result.scalars().all()
    log_dates = {log.date for log in logs}

    current_streak = 0
    day = date.today()
    while day in log_dates:
        current_streak += 1
        day -= timedelta(days=1)

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
