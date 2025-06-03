#%pip install shap
from sklearn.metrics import accuracy_score

def feature_corrilations(data):
    pass


def systematics_dependence(data):
    pass


# shap for boosted decision tree
def minimal_dependent_features(self, data):
    import shap
    import numpy as np

    # Scale the input like in training
    X_scaled = self.__scaler.transform(data)

    # Use SHAP TreeExplainer on the trained model
    explainer = shap.Explainer(self.__model)
    shap_values = explainer(X_scaled)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Determine feature names
    if hasattr(data, 'columns'):
        feature_names = data.columns.tolist()
    else:
        feature_names = [f"f{i}" for i in range(X_scaled.shape[1])]

    # Sort features by importance
    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    sorted_names = [name for name, _ in importance]

    return sorted_names

