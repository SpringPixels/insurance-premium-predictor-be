from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import pandas as pd


from schema.user_input import UserInput
from ml_model.predict import predict_and_explain, MODEL_VERSION
from schema.prediction_response import PredictionResponse
from config.database import Base, engine, get_db
from ml_model.db_models import PredictionLog
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This securely runs your table creation asynchronously on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

# Pass the lifespan function to your FastAPI instance
app = FastAPI(lifespan=lifespan)





# human readable
@app.get('/')
def home():
    return{'message': 'Insurance Premium Prediction API' }



# machine readable
@app.get('/health')
def health_check():
    return {
        'status' : 'OK',
        'version' : MODEL_VERSION,
        'model_loaded' : model is not None
    }



@app.post('/predict/explain', response_model= PredictionResponse)
def predict_premium_with_explanation(data: UserInput, db: Session =Depends(get_db)):

    user_input = {
       
        'income_lpa' : float(data.income_lpa),
        'occupation' : str(data.occupation),
        'bmi' : float(data.bmi),
        'age_group' :str(data.age_group),
        'lifestyle_risk' : str(data.lifestyle_risk),
        'city_tier' : int(data.city_tier)
    }

    

    try:
        result = predict_and_explain(user_input)   # returns the full dict already — prediction + explanation

        log = PredictionLog(
            **user_input,
            predicted_category=result["prediction_results"]["predicted_category"]
        )
        db.add(log)
        db.commit()

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})