from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from ml_model.predict import predict_output, model, MODEL_VERSION
from schema.prediction_response import PredictionResponse

app = FastAPI()


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



@app.post('/predict', response_model= PredictionResponse)
def predict_premium(data: UserInput):

    user_input = {
       
        'income_lpa' : float(data.income_lpa),
        'occupation' : str(data.occupation),
        'bmi' : float(data.bmi),
        'age_group' :str(data.age_group),
        'lifestyle_risk' : str(data.lifestyle_risk),
        'city_tier' : int(data.city_tier)
    }


    try:
    
       prediction = predict_output([user_input])

       return JSONResponse(status_code=200, content={'response': prediction})


    except Exception as e:

        return JSONResponse(status_code=500, content=str(e))