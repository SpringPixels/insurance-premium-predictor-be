from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import date

class RevenuePoint(BaseModel):
    month: str  # e.g. "2026-06"
    amount: float  # in rupees

class PaidMember(BaseModel):
    user_id: int
    email: str
    total_paid: float

class AdminDashboardResponse(BaseModel):
    total_revenue: float
    total_claims: float
    total_users: int
    new_users_this_month: int
    revenue_trend: List[RevenuePoint]
    segment_breakdown: Dict[str, int]
    paid_members: List[PaidMember]

class ClaimResponse(BaseModel):
    id: int
    amount: float
    description: str
    date: date

class RiskAnalysisResponse(BaseModel):
    risky_behaviour_rate: float
    renewal_multiplier: float
    base_premium: float
    predicted_renewal_premium: float
    total_exercises: int
    total_claims_amount: float
    claims: List[ClaimResponse]

class ClaimCreate(BaseModel):
    amount: float
    description: str