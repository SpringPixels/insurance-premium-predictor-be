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
    # Baseline threshold shifted to 0.4 (40%) so average risk doesn't immediately increase prices
    multiplier = 1.0 + (risk_rate - 0.4)
    
    # Reward for zero claims
    if user_data.get('total_claims_amount', 0.0) == 0:
        multiplier -= 0.15 # 15% discount for claim-free year
        
    # Further reward for being active
    exercises = user_data.get('total_exercises', 0)
    if exercises > 0:
        multiplier -= min(0.15, exercises * 0.02) # Up to 15% discount for exercises
        
    claims_amount = user_data.get('total_claims_amount', 0.0)
    base_premium = user_data.get('base_premium', 0.0)
    
    if claims_amount > 0:
        # If they claimed more than what they paid, add a huge penalty red flag
        if base_premium > 0 and claims_amount > base_premium:
            multiplier += 0.5 + (claims_amount / base_premium) * 0.1
            
        # Ensure it never goes below 1.0 if they have a claim
        multiplier = max(1.0, multiplier)
        
    # Cap the multiplier between 0.5x (50% discount max) and 3.0x (300% penalty max)
    multiplier = max(0.5, min(multiplier, 3.0))
    
    
    
    return {
        "risky_behaviour_rate": round(risk_rate * 100, 1),
        "renewal_multiplier": round(multiplier, 2)
    }
