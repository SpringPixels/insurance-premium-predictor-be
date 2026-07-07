from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from config.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    income_lpa = Column(Float)
    occupation = Column(String)
    bmi = Column(Float)
    age_group = Column(String)
    lifestyle_risk = Column(String)
    city_tier = Column(Integer)
    predicted_category = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)