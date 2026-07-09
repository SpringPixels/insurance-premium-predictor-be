from pydantic import BaseModel
from datetime import date
from typing import Optional

class StepBatch(BaseModel):
    steps: int
    date: Optional[date] = None

class StepLogResponse(BaseModel):
    date: date
    total_steps: int

class ActivityCheckIn(BaseModel):
    activity_type: str | None = None

class ActivityLogResponse(BaseModel):
    date: date
    completed: bool
    activity_type: str | None

class StreakResponse(BaseModel):
    current_streak: int
    longest_streak: int
    last_7_days: list[ActivityLogResponse]
    message: str

class StandardResponse(BaseModel):
    message: str