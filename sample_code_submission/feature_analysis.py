#%pip install shap
from sklearn.metrics import accuracy_score

def feature_corrilations(data):
    pass


def systematics_dependence(data):
    pass


import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

module_path = os.path.join(os.getcwd(), "sample_code_submission", "BDT")
if module_path not in sys.path:
    sys.path.append(module_path)

import sample_code_submission.BDT.boosted_decision_tree as BoostedDecisionTree

def minimal_dependent_features(data):
    """
    Uses permutation importance on BoostedDecisionTree to get top 10 important features.
 
    Parameters:
        data (pd.DataFrame): Dataset with 'Label' column.
 
    Returns:
        List[str]: Top 10 most important feature names.
    """
    X = data.drop(columns=["labels", "Weight", "DetailedLabel"], errors="ignore")
    y = data["labels"]
 
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    model = BoostedDecisionTree.get_best_model()

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean
    }).sort_values(by="Importance", ascending=False)

    return importance_df["Feature"].head(10).tolist()

