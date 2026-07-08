from pydantic import BaseModel, EmailStr, Field

class UserSignup(BaseModel):
    full_name:str
    email: EmailStr
    phone_no: str
    password: str = Field(min_length=8, description="Must be at least 8 characters")

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"