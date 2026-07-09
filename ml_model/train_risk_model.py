import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def generate_data(n_samples=2000):
    np.random.seed(42)
    
    age = np.random.randint(18, 65, n_samples)
    bmi = np.random.uniform(18.0, 40.0, n_samples)
    income_lpa = np.random.uniform(3.0, 30.0, n_samples)
    city_tier = np.random.choice([1, 2, 3], n_samples)
    lifestyle_risk = np.random.choice([1, 2, 3, 4], n_samples) # 1: Low, 4: Extreme
    
    total_exercises = np.random.randint(0, 30, n_samples)
    # Most people have 0 claims
    total_claims = np.random.choice(
        [0, np.random.uniform(1000, 50000)], 
        n_samples, 
        p=[0.8, 0.2]
    )
    
    # Calculate synthetic risky behaviour rate (0.0 to 1.0)
    # Base risk from BMI and Lifestyle
    base_risk = (bmi - 18) / 22 * 0.3 + (lifestyle_risk / 4) * 0.4
    
    # Exercises reduce risk by up to 0.3 (if 30 exercises)
    exercise_factor = (total_exercises / 30) * 0.3
    
    # Claims increase risk by up to 0.4
    claims_factor = np.clip(total_claims / 50000 * 0.4, 0, 0.4)
    
    risk_rate = np.clip(base_risk - exercise_factor + claims_factor, 0.05, 0.95)
    
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'income_lpa': income_lpa,
        'city_tier': city_tier,
        'lifestyle_risk': lifestyle_risk,
        'total_exercises': total_exercises,
        'total_claims_amount': total_claims,
        'risky_behaviour_rate': risk_rate
    })
    
    return df

def train_and_save():
    print("Generating synthetic risk data...")
    df = generate_data(5000)
    
    X = df.drop('risky_behaviour_rate', axis=1)
    y = df['risky_behaviour_rate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model R2 Score: {score:.4f}")
    
    save_path = os.path.join(os.path.dirname(__file__), 'Risk_Model.pkl')
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_and_save()
