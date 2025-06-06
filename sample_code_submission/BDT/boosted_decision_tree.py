import numpy as np

import os
import sys


def find_and_add_module_path(filename):
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(3):
        candidate = os.path.join(cur, filename)
        if os.path.isfile(candidate):
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return
        cur = os.path.dirname(cur)


find_and_add_module_path("XGB_boosted_decision_tree.py")

import XGB_boosted_decision_tree


def get_best_model():
    """
    Returns the best pre-trained Boosted Decision Tree model we found so far
    """
    model = XGB_boosted_decision_tree.XGBBoostedDecisionTree()
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


def get_model_with_best_param():
    params = {
        "n_estimators": np.int64(191),
        "max_depth": np.int64(5),
        "max_leaves": np.int64(0),
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    return XGB_boosted_decision_tree.XGBBoostedDecisionTree(params)
