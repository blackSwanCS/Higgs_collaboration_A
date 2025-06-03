#%pip install shap
from sklearn.metrics import accuracy_score

def feature_corrilations(data):
    pass


def systematics_dependence(data):
    pass


# shap for boosted decision tree
import shap
import numpy as np

def minimal_dependent_features(data):
    # Scale the data using the internal scaler of the BoostedDecisionTree
    X_scaled = bdt._BoostedDecisionTree__scaler.transform(data)

    # Compute SHAP values using the internal XGBoost model
    explainer = shap.Explainer(bdt._BoostedDecisionTree__model)
    shap_values = explainer(X_scaled)

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Get feature names
    if hasattr(data, 'columns'):
        feature_names = data.columns.tolist()
    else:
        feature_names = [f"f{i}" for i in range(X_scaled.shape[1])]

    # Sort feature names by importance
    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    sorted_names = [name for name, _ in importance]

    return sorted_names

