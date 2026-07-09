from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from schema.auth import UserSignup, Token, UserOut, StandardResponse
from config.database import get_db
from ml_model.db_models import User
from config.security import hash_password, verify_password, create_access_token
from dependencies import get_current_user

router = APIRouter(tags=["auth"])

@router.post("/signup", response_model=StandardResponse)
async def signup(data: UserSignup, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == data.email))
    existing = result.scalars().first()

    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        full_name=data.full_name,
        email=data.email,
        phone_no=data.phone_no,
        hashed_password=hash_password(data.password)
    )
    db.add(user)
    await db.commit()
    return {"message": "User created successfully"}

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalars().first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.email, "role": user.role, "full_name": user.full_name})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "phone_no": user.phone_no,
            "role": user.role
        }
    }

@router.get("/me", response_model=UserOut)
async def read_current_user(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "phone_no": current_user.phone_no,
        "role": current_user.role,
        "is_paid": current_user.is_paid or False,
        "age": current_user.age,
        "weight": current_user.weight,
        "height": current_user.height,
        "is_smoker": current_user.is_smoker,
        "occupation": current_user.occupation,
        "income_lpa": current_user.income_lpa,
        "city": current_user.city,
    }
