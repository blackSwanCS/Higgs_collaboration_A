import csv
import numpy as np
from HiggsML.datasets import download_dataset
from sample_code_submission.BDT.XGB_boosted_decision_tree import XGBBoostedDecisionTree
from itertools import product
from tqdm import tqdm
from time import time
from get_data import get_data


def evaluate_significance(
    params, train_data, train_labels, train_weights, val_data, val_labels, val_weights
):
    model = XGBBoostedDecisionTree(params)
    start_time = time()
    model.fit(train_data, train_labels, train_weights)
    end_time = time()
    print(f"Fitting time: {end_time - start_time:.2f} seconds")
    model.predict(val_data, labels=val_labels, weights=val_weights)
    significance = model.significance()
    auc = model.auc()
    return significance, auc


if __name__ == "__main__":
    # Load data once
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = (
        get_data()
    )

    # Define parameter grid (edit these lists as needed)
    n_estimators_list = np.linspace(500, 550, 1, dtype=int)
    max_depth_list = np.arange(2, 10, 1)
    eta_list = np.linspace(0.01, 0.2, 1)
    subsample_list = np.linspace(0.75, 1.0, 1)

    param_grid = list(
        product(n_estimators_list, max_depth_list, eta_list, subsample_list)
    )

    best_significance = -np.inf
    auc_best_significance = 0.5
    best_params = None
    excel = np.zeros(
        (
            len(n_estimators_list)
            * len(max_depth_list)
            * len(eta_list)
            * len(subsample_list),
            6,
        )
    )
    i = 0
    for n_estimators, max_depth, eta, subsample in tqdm(param_grid):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_leaves": 0,
            "eta": eta,
            "subsample": subsample,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        #        print(f"Testing params: {params}")
        significance, auc = evaluate_significance(
            params,
            train_data,
            train_labels,
            train_weights,
            val_data,
            val_labels,
            val_weights,
        )
        t = time()
        excel[i] = [n_estimators, max_depth, eta, subsample, t, significance]
        i += 1
        #        print(f"Significance: {significance:.4f}")
        if significance > best_significance:
            best_significance, auc_best_significance = significance, auc
            best_params = params

    print("\nBest parameters:")
    print(best_params)
    print(f"Best significance: {best_significance:.4f}")
    csv_file_path = "C:/Users/julie/Documents/CS/Cours/Black Swans/EI/donnees_500_550"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(excel)
