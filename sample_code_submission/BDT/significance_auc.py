import numpy as np
from time import time
from get_data import get_data
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

from XGB_boosted_decision_tree import XGBBoostedDecisionTree


def evaluate(
    params, train_data, train_labels, train_weights, val_data, val_labels, val_weights
):
    model = XGBBoostedDecisionTree(params)
    start_time = time()
    model.fit(train_data, train_labels, train_weights)
    end_time = time()
    print(f"Fitting time: {end_time - start_time:.2f} seconds")
    model.predict(val_data, labels=val_labels, weights=val_weights)
    significance = model.significance(val_labels)
    auc = model.auc()
    return significance, auc


def test(n_estimator, max_depth, eta, subsample):
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = (
        get_data()
    )
    params = {
        "n_estimators": np.int64(n_estimator),
        "max_depth": max_depth,
        "max_leaves": 0,
        "eta": eta,
        "subsample": subsample,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
    }

    significance, auc = evaluate(
        params,
        train_data,
        train_labels,
        train_weights,
        val_data,
        val_labels,
        val_weights,
    )
    print(f"Parameters : {params}")
    print(f"Significance: {significance:.4f},AUC : {auc}")


test(450, 7, 0.15525, 0.9375)
