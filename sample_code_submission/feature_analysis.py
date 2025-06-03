#%pip install shap
import shap
import numpy as np
from sklearn.metrics import accuracy_score

def feature_corrilations(data):
    pass


def systematics_dependence(data):
    pass


# shap for boosted decision tree
def minimal_dependent_features(self, data, feature_names=None, top_k=None):
    # Scale the input like in training
    X_scaled = self.__scaler.transform(data)

    # Use SHAP TreeExplainer on the trained model
    explainer = shap.Explainer(self.__model)
    shap_values = explainer(X_scaled)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Handle feature names
    if feature_names is None:
        if hasattr(data, 'columns'):
            feature_names = data.columns.tolist()
        else:
            feature_names = [f"f{i}" for i in range(X_scaled.shape[1])]

    # Sort and return
    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    return importance[:top_k] if top_k else importance


# Permutation feature importance for neutral network
def permutation_feature_importance(model, X_val, y_val, metric_fn=accuracy_score, n_repeats=5):

    baseline = metric_fn(y_val, model.predict(X_val))
    importances = []

    for col in range(X_val.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_temp = X_val.copy()
            np.random.shuffle(X_temp[:, col])  # Permute column
            score = metric_fn(y_val, model.predict(X_temp))
            scores.append(baseline - score)  # Performance drop
        importances.append((col, np.mean(scores)))

    # Sort by importance
    return sorted(importances, key=lambda x: x[1], reverse=True)
