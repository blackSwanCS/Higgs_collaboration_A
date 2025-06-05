import sample_code_submission.BDT.XGB_boosted_decision_tree
import numpy as np


def get_best_model():
    """
    Returns the best pre-trained Boosted Decision Tree model we found so far
    """
    model = BDT.XGB_boosted_decision_tree.XGBBoostedDecisionTree()
    model.load_model()
    return model


"""
    params = {
        "n_estimators": np.int64(191),
        "max_depth": np.int64(5),
        "max_leaves": np.int64(0),
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    return BDT.xgb_boosted_decision_tree.XGBBoostedDecisionTree(params)"""
