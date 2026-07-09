from pydantic import BaseModel
from typing import List, Dict

class RevenuePoint(BaseModel):
    month: str  # e.g. "2026-06"
    amount: float  # in rupees

class AdminDashboardResponse(BaseModel):
    total_revenue: float
    total_users: int
    new_users_this_month: int
    revenue_trend: List[RevenuePoint]
    segment_breakdown: Dict[str, int]