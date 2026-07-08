from pydantic import BaseModel, EmailStr, Field

class UserSignup(BaseModel):
    full_name:str
    email: EmailStr
    phone_no: str
    password: str = Field(min_length=8, description="Must be at least 8 characters")

class UserLogin(BaseModel):
    id:int
    full_name: str
    email: EmailStr
    phone_no: str
    password: str

class UserOut(BaseModel):
    full_name: str
    email: EmailStr
    phone_no: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut