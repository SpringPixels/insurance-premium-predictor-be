import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "ml_model/artifacts/segment_kmeans.pkl"
SCALER_PATH = "ml_model/artifacts/segment_scaler.pkl"
LABEL_MAP_PATH = "ml_model/artifacts/segment_labels.pkl"

AGE_GROUP_MAP = {"young": 0, "adult": 1, "middle_aged": 2, "senior": 3}
LIFESTYLE_RISK_MAP = {"low": 0, "medium": 1, "high": 2}

FEATURE_COLUMNS = ["income_lpa", "bmi", "age_group_enc", "lifestyle_risk_enc", "city_tier"]

WORKOUT_PLANS = {
    "Low Risk": {
        "activity": "Brisk walking or light jogging, 30 min, 4x/week",
        "goal": "Maintain fitness and cardiovascular health",
    },
    "Moderate Risk": {
        "activity": "Guided cardio + strength training, 30-40 min, 4-5x/week",
        "goal": "Improve BMI and reduce lifestyle risk factors",
    },
    "High Risk": {
        "activity": "Low-impact supervised exercise (walking, swimming) + dietary consult, 20-30 min, 5x/week",
        "goal": "Reduce health risk under guided monitoring; consult a physician before starting",
    },
}


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group_enc"] = df["age_group"].map(AGE_GROUP_MAP).fillna(1)
    df["lifestyle_risk_enc"] = df["lifestyle_risk"].map(LIFESTYLE_RISK_MAP).fillna(1)
    return df


def _risk_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["bmi"] * 0.4
        + df["lifestyle_risk_enc"] * 20
        + df["age_group_enc"] * 10
    )


def train_segments(df: pd.DataFrame, n_clusters: int = 3):
    df = _encode(df)
    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    df["cluster"] = cluster_ids
    df["risk_score"] = _risk_score(df)

    ranked = df.groupby("cluster")["risk_score"].mean().sort_values()
    ordered_labels = ["Low Risk", "Moderate Risk", "High Risk"][:n_clusters]
    label_map = {cluster_id: label for cluster_id, label in zip(ranked.index, ordered_labels)}

    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_map, LABEL_MAP_PATH)

    return {
        "n_users": len(df),
        "cluster_sizes": df["cluster"].value_counts().to_dict(),
        "label_map": label_map,
    }


def predict_segment(user_input: dict) -> dict:
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)

    df = pd.DataFrame([user_input])
    df = _encode(df)
    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    cluster_id = int(kmeans.predict(X_scaled)[0])
    label = label_map.get(cluster_id, "Moderate Risk")
    plan = WORKOUT_PLANS[label]

    return {
        "cluster_id": cluster_id,
        "segment_label": label,
        "recommended_activity": plan["activity"],
        "goal": plan["goal"],
    }