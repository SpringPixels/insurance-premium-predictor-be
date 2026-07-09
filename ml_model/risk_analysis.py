import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Risk_Model.pkl')

_model = None

def get_risk_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Risk model not found at {MODEL_PATH}")
    return _model

def analyze_risk(user_data: dict) -> dict:
    """
    user_data must contain:
    - age (int)
    - bmi (float)
    - income_lpa (float)
    - city_tier (int)
    - lifestyle_risk (str/int) -> map to 1-4
    - total_exercises (int)
    - total_claims_amount (float)
    """
    model = get_risk_model()
    
    # Map lifestyle risk if it's a string
    lifestyle_mapping = {
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Extreme": 4
    }
    lr = user_data.get('lifestyle_risk', 2)
    if isinstance(lr, str):
        lr = lifestyle_mapping.get(lr, 2)
        
    df = pd.DataFrame([{
        'age': user_data.get('age', 30),
        'bmi': user_data.get('bmi', 22.0),
        'income_lpa': user_data.get('income_lpa', 10.0),
        'city_tier': user_data.get('city_tier', 2),
        'lifestyle_risk': lr,
        'total_exercises': user_data.get('total_exercises', 0),
        'total_claims_amount': user_data.get('total_claims_amount', 0.0),
    }])
    
    risk_rate = float(model.predict(df)[0])
    
    # Renewal multiplier logic:
    # 1.0 (base) + (risk_rate - 0.2) -> standard multiplier.
    # We want a high risk rate to heavily multiply the premium.
    multiplier = 1.0 + (risk_rate - 0.2)
    multiplier = max(1.0, multiplier)
    
    return {
        "risky_behaviour_rate": round(risk_rate * 100, 1),
        "renewal_multiplier": round(multiplier, 2)
    }
