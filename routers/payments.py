import os
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from schema.payments import PaymentCreate, PaymentCreateResponse, PaymentStatusResponse, StandardResponse
from config.database import get_db
from ml_model.db_models import User, Payment, PaymentStatus
from dependencies import get_current_user

router = APIRouter(tags=["payments"])

@router.post("/payments/create", response_model=PaymentCreateResponse)
async def create_payment(data: PaymentCreate, db: AsyncSession = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    # Mocking successful payment creation
    mock_order_id = f"order_{uuid.uuid4().hex[:14]}"
    mock_payment_id = f"pay_{uuid.uuid4().hex[:14]}"
    
    payment = Payment(
        user_id=current_user.id, 
        amount=data.amount,
        provider_order_id=mock_order_id, 
        provider_payment_id=mock_payment_id,
        status=PaymentStatus.SUCCEEDED
    )
    db.add(payment)
    await db.commit()
    
    return {"order_id": mock_order_id, "amount": data.amount, "key": "mock_key"}

@router.get("/payments/{payment_id}", response_model=PaymentStatusResponse)
async def get_payment_status(payment_id: int, db: AsyncSession = Depends(get_db),
                              current_user: User = Depends(get_current_user)):
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
