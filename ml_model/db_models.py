from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from config.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    income_lpa = Column(Float)
    occupation = Column(String)
    bmi = Column(Float)
    age_group = Column(String)
    lifestyle_risk = Column(String)
    city_tier = Column(Integer)
    predicted_category = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("Users", back_populates="predictions")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # "user" or "admin"
    is_active = Column(Boolean, default=True)

    predictions = relationship("PredictionLog", back_populates="user")