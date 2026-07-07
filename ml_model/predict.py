import shap
import joblib
import pandas as pd
from pydantic import BaseModel
from schema.user_input import UserInput



# import the ml model
with open('ml_model/Insurance_Premium_Predictor_mdl.pkl', 'rb') as f:
    model = joblib.load(f)

# ML-FLOW
MODEL_VERSION = '1.0.0'

# Get class labels from model (important for matching probabilities to class names)
class_labels = model.classes_.tolist()


preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']
explainer = shap.TreeExplainer(classifier)


def predict_and_explain(user_input: dict) -> dict:
    """
    Accepts raw user metrics, predicts the premium tier,
    calculates classification confidence, and extracts targeted SHAP explanations.
    """
    # Convert input dict to DataFrame
    df = pd.DataFrame([user_input])
    feature_names = df.columns.tolist()

    # --- Step A: Machine Learning Inference ---
    # Predict the class
    predicted_class = model.predict(df)[0]
    # Get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    # Map class names to their exact output probabilities
    class_probs = dict(zip(class_labels, [round(float(p), 4) for p in probabilities]))

    # --- Step B: Explainable AI (SHAP) — needs TRANSFORMED data, not raw ---
    df_transformed = preprocessor.transform(df)
    if hasattr(df_transformed, "toarray"):
        df_transformed = df_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    shap_matrix = explainer.shap_values(df_transformed)   # <-- use transformed data here
    
    
    predicted_class_idx = class_labels.index(predicted_class)

    # Handle both old (list) and new (3D array) SHAP output formats
    if isinstance(shap_matrix, list):
       # Old format: list of arrays, one per class
       target_shap_values = shap_matrix[predicted_class_idx][0]
    else:
       # New format: single array shaped (n_samples, n_features, n_classes)
       target_shap_values = shap_matrix[0, :, predicted_class_idx]

    # Map features to their raw SHAP impact values
    raw_explanations = dict(zip(feature_names, [round(float(v), 4) for v in target_shap_values]))

    # --- Step C: Structure Human-Readable Interpretations ---
    # Sort features by absolute impact so the user sees the most influential factors first
    sorted_features = sorted(raw_explanations.items(), key=lambda item: abs(item[1]), reverse=True)
    
    insights = []
    for feature, value in sorted_features:
        direction = "increased" if value > 0 else "decreased"
        insights.append(f"Feature '{feature}' {direction} the likelihood of being in this tier.")

    # Return unified, production-grade JSON dictionary
    return {
        "model_metadata": {
            "version": MODEL_VERSION,
            "status": "success"
        },
        "prediction_results": {
            "predicted_category": predicted_class,
            "confidence_score": round(float(confidence), 4),
            "all_class_probabilities": class_probs
        },
        "explainable_ai": {
            "raw_shap_values": raw_explanations,
            "top_driving_factors": dict(sorted_features),
            "human_readable_insights": insights
        }
    }

# compare two scenarios
class CompareRequest(BaseModel):
    scenario_a: UserInput
    scenario_b: UserInput




