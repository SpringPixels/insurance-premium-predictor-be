from pydantic import BaseModel, EmailStr

class ContactUsCreate(BaseModel):
    name: str
    email: EmailStr
    phone_no: str
    subject: str
    message: str

class StandardResponse(BaseModel):
    message: str
