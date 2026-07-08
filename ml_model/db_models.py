from enum import Enum
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Enum as SQLEnum, Date
from sqlalchemy.orm import relationship
from datetime import datetime, date as py_date
from config.database import Base


class PredictionInput(Base):
    __tablename__ = "prediction_inputs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    age = Column(Integer, nullable=False)
    weight = Column(Float, nullable=False)      # in kg
    height = Column(Float, nullable=False)      # in cm
    is_smoker = Column(Boolean, default=False)
    occupation = Column(String, nullable=False)
    income_lpa = Column(Float, nullable=False)
    city = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="prediction_inputs")
    prediction_log = relationship("PredictionLog", back_populates="input", uselist=False)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    input_id = Column(Integer, ForeignKey("prediction_inputs.id"), nullable=True)

    income_lpa = Column(Float)
    occupation = Column(String)
    bmi = Column(Float)
    age_group = Column(String)
    lifestyle_risk = Column(String)
    city_tier = Column(Integer)
    predicted_category = Column(String)
    predicted_premium = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")
    input = relationship("PredictionInput", back_populates="prediction_log")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone_no =Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # "user" or "admin"
    is_active = Column(Boolean, default=True)

    predictions = relationship("PredictionLog", back_populates="user")
    prediction_inputs = relationship("PredictionInput", back_populates="user")



class PaymentStatus(str, Enum):
    CREATED = "created"       # order created, no payment attempt yet
    PENDING = "pending"       # payment intent created, awaiting user action
    PROCESSING = "processing" # provider is processing (e.g. bank auth)
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True)
    amount = Column(Integer)  # store in paise/cents, never float
    currency = Column(String, default="INR")
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.CREATED)
    provider_order_id = Column(String, nullable=True)   # e.g. Razorpay order id
    provider_payment_id = Column(String, nullable=True) # set once payment succeeds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

class StepLog(Base):
    __tablename__ = "step_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    steps = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)  # when batch was synced
    
    date = Column(Date, default=py_date.today) # day the steps belong to



class ContactUs(Base):
    __tablename__ = "contact_us"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    phone_no = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    message = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)