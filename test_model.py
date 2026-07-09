import pandas as pd
import os
from ml_model.segmentation import train_segments, predict_segment

os.makedirs("ml_model/artifacts", exist_ok=True)

# Create dummy data for KMeans
df = pd.DataFrame([
    {"age_group": "young", "lifestyle_risk": "low", "bmi": 22.0, "income_lpa": 10.0, "city_tier": 1},
    {"age_group": "senior", "lifestyle_risk": "high", "bmi": 32.0, "income_lpa": 5.0, "city_tier": 2},
    {"age_group": "adult", "lifestyle_risk": "medium", "bmi": 26.0, "income_lpa": 15.0, "city_tier": 1},
])

print("Training KMeans segments...")
train_segments(df, n_clusters=3)

print("Testing prediction...")
user_data = {
    "age_group": "young",
    "lifestyle_risk": "low",
    "bmi": 22.0,
    "income_lpa": 10.0,
    "city_tier": 1
}

result = predict_segment(user_data)
print("Result for young/low_risk/22_bmi:", result)
