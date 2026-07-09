from pydantic import BaseModel, EmailStr, Field

class UserSignup(BaseModel):
    full_name: str
    email: EmailStr
    phone_no: str
    password: str = Field(min_length=8, description="Must be at least 8 characters")

class UserOut(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    phone_no: str
    role: str
    
    is_paid: bool = False
    age: int | None = None
    weight: float | None = None
    height: float | None = None
    is_smoker: bool | None = None
    occupation: str | None = None
    income_lpa: float | None = None
    city: str | None = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class StandardResponse(BaseModel):
    message: str
