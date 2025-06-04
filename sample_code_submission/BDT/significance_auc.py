import numpy as np
from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree
from itertools import product
from tqdm import tqdm
from time import time


def get_data():
    data = download_dataset("blackSwan_data")
    data.load_train_set()
    training_set = data.get_train_set()
    feature_keys = [
        k
        for k in training_set.keys()
        if k not in ("labels", "weights", "detailed_labels")
    ]
    X = np.column_stack([training_set[k] for k in feature_keys])
    y = training_set["labels"]
    w = training_set["weights"]

    n = len(y)
    split = int(n * 0.8)
    train_data, val_data = X[:split], X[split:]
    train_labels, val_labels = y[:split], y[split:]
    train_weights, val_weights = w[:split], w[split:]

    return train_data, train_labels, train_weights, val_data, val_labels, val_weights


def evaluate(
    params, train_data, train_labels, train_weights, val_data, val_labels, val_weights
):
    model = BoostedDecisionTree(params)
    start_time = time()
    model.fit(train_data, train_labels, train_weights)
    end_time = time()
    print(f"Fitting time: {end_time - start_time:.2f} seconds")
    model.predict(val_data, labels=val_labels, weights=val_weights)
    significance = model.significance(val_labels)
    auc=model.auc()
    return significance,auc


def test(n_estimator,max_depth,eta,subsample):
        train_data, train_labels, train_weights, val_data, val_labels, val_weights = (
        get_data()
    )
        params = {
            "n_estimators": np.int(n_estimator),
            "max_depth": max_depth,
            "max_leaves": 0,
            "eta": eta,
            "subsample": subsample,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method" : "hist",
            "device" : "cuda"
        }

        significance,auc = evaluate(
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


test(325,10,0.1525,0.875)