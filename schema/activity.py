from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional

class StepBatch(BaseModel):
    steps: int
    date: Optional[date] = None

class StepLogResponse(BaseModel):
    date: date
    total_steps: int

class ActivityCheckIn(BaseModel):
    activity_type: str | None = None
    is_recommended: bool = False

class ActivityLogResponse(BaseModel):
    date: date
    completed: bool
    activity_type: str | None

class ActivityHistoryItem(BaseModel):
    id: int
    date: date
    activity_type: str | None
    is_recommended: bool
    created_at: datetime

class StreakResponse(BaseModel):
    current_streak: int
    longest_streak: int
    last_7_days: list[ActivityLogResponse]
    message: str

class StandardResponse(BaseModel):
    message: str