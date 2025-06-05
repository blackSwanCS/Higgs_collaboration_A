import xgb_boosted_decision_tree
import numpy as np


def get_best_model():
    """
    Returns the best pre-trained Boosted Decision Tree model we found so far
    """

    model = xgb_boosted_decision_tree.XGBBoostedDecisionTree()
    model.load_model()
    return model
    """
    params = {
        "n_estimators": np.int64(236),
        "max_depth": np.int64(9),
        "max_leaves": np.int64(0),
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    params = {
        "n_estimators": np.int64(450),
        "max_depth": np.int64(7),
        "max_leaves": np.int64(0),
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "eta": np.float64(0.15525),
        "subsample": np.float64(0.9375),
    }
    return BDT.xgb_boosted_decision_tree.XGBBoostedDecisionTree(params)
    """
