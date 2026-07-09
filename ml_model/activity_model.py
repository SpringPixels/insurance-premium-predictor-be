import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "ml_model/artifacts/activity_recommender.pkl"
SCALER_PATH = "ml_model/artifacts/activity_scaler.pkl"

AGE_GROUP_MAP = {"young": 0, "adult": 1, "middle_aged": 2, "senior": 3}
LIFESTYLE_RISK_MAP = {"low": 0, "medium": 1, "high": 2}

def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        age_group = np.random.choice(["young", "adult", "middle_aged", "senior"])
        lifestyle_risk = np.random.choice(["low", "medium", "high"])
        bmi = np.random.uniform(18.0, 40.0)
        income_lpa = np.random.uniform(3.0, 50.0)
        city_tier = np.random.choice([1, 2, 3])
        
        # Rule-based synthetic logic to create ground truth for the model to learn
        if bmi > 30 and lifestyle_risk == "high":
            activity = "Low-impact supervised exercise (walking, swimming)"
            goal = "Reduce health risk under guided monitoring"
        elif age_group in ["senior"]:
            activity = "Light Yoga and Walking"
            goal = "Maintain mobility and joint health"
        elif bmi > 25 and lifestyle_risk != "low":
            activity = "Guided cardio + strength training, 30-40 min, 4-5x/week"
            goal = "Improve BMI and reduce lifestyle risk factors"
        elif age_group == "young" and bmi < 25 and lifestyle_risk == "low":
            activity = "High Intensity Interval Training (HIIT)"
            goal = "Build endurance and peak fitness"
        else:
            activity = "Brisk walking or light jogging, 30 min, 4x/week"
            goal = "Maintain fitness and cardiovascular health"
            
        data.append({
            "age_group": age_group,
            "lifestyle_risk": lifestyle_risk,
            "bmi": bmi,
            "income_lpa": income_lpa,
            "city_tier": city_tier,
            "activity": activity,
            "goal": goal
        })
    
    return pd.DataFrame(data)

def preprocess_features(df):
    df_proc = df.copy()
    df_proc["age_group_enc"] = df_proc["age_group"].map(AGE_GROUP_MAP).fillna(1)
    df_proc["lifestyle_risk_enc"] = df_proc["lifestyle_risk"].map(LIFESTYLE_RISK_MAP).fillna(1)
    return df_proc[["income_lpa", "bmi", "age_group_enc", "lifestyle_risk_enc", "city_tier"]]

def train_activity_model():
    print("Generating synthetic data...")
    df = generate_synthetic_data(10000)
    
    X = preprocess_features(df)
    y = df["activity"]
    
    # Store goals map for later lookup during inference
    import os
    os.makedirs("ml_model/artifacts", exist_ok=True)
    goals_map = df.drop_duplicates("activity").set_index("activity")["goal"].to_dict()
    joblib.dump(goals_map, "ml_model/artifacts/activity_goals.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_activity_model()
