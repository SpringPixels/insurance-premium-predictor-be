import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from schema.segment import TrainSegmentsResponse, SegmentResponse
from config.database import get_db
from ml_model.db_models import User, PredictionLog
from dependencies import require_admin, get_current_user
from ml_model.segmentation import train_segments, predict_segment

router = APIRouter(tags=["segmentation"])

@router.post("/segment/train", response_model=TrainSegmentsResponse)
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

@router.post("/segment/predict", response_model=SegmentResponse)
async def get_user_segment(data: dict, current_user: User = Depends(get_current_user)):
    try:
        return predict_segment(data)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Segmentation model not trained yet")
