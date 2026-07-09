from pydantic import BaseModel
from datetime import datetime

class PaymentCreate(BaseModel):
    amount: int  # Razorpay amounts are usually passed in paise (integers)

class PaymentCreateResponse(BaseModel):
    order_id: str
    amount: int
    key: str

class PaymentStatusResponse(BaseModel):
    id: int
    amount: int
    currency: str
    status: str
    created_at: datetime | None

class StandardResponse(BaseModel):
    message: str
