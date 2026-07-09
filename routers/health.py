from fastapi import APIRouter
from ml_model.predict import MODEL_VERSION

# Assuming `model` check logic is simplified or needs an import from predict.
# In app.py, `model is not None` was checked. We will import model from predict.
from ml_model.predict import model

router = APIRouter(tags=["health"])

@router.get('/')
def home():
    return {'message': 'Insurance Premium Prediction API'}

@router.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaded': model is not None
    }
