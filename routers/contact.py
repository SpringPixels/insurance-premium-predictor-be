from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from schema.contact import ContactUsCreate, StandardResponse
from config.database import get_db
from ml_model.db_models import ContactUs

router = APIRouter(tags=["contact"])

@router.post("/contact-us", response_model=StandardResponse)
async def submit_contact_form(data: ContactUsCreate, db: AsyncSession = Depends(get_db)):
    contact = ContactUs(
        name=data.name,
        email=data.email,
        phone_no=data.phone_no,
        subject=data.subject,
        message=data.message
    )
    db.add(contact)
    await db.commit()
    return {"message": "Your message has been submitted successfully"}
