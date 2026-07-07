from pydantic import BaseModel, EmailStr, Field

class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, description="Must be at least 8 characters")

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"