import numpy as np
from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree
from itertools import product
from tqdm import tqdm

def get_data():
    data = download_dataset("blackSwan_data")
    data.load_train_set()
    training_set = data.get_train_set()
    feature_keys = [k for k in training_set.keys() if k not in ("labels", "weights", "detailed_labels")]
    X = np.column_stack([training_set[k] for k in feature_keys])
    y = training_set["labels"]
    w = training_set["weights"]

    n = len(y)
    split = int(n * 0.8)
    train_data, val_data = X[:split], X[split:]
    train_labels, val_labels = y[:split], y[split:]
    train_weights, val_weights = w[:split], w[split:]

    return train_data, train_labels, train_weights, val_data, val_labels, val_weights

def evaluate_significance(params, train_data, train_labels, train_weights, val_data, val_labels, val_weights):
    model = BoostedDecisionTree(params)
    model.fit(train_data, train_labels, train_weights)
    model.predict(val_data, labels=val_labels, weights=val_weights) 
    significance = model.significance()
    model.save()
    return significance

if __name__ == "__main__":
    # Load data once
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()

    # Define parameter grid (edit these lists as needed)
    n_estimators_list = np.linspace(100, 510, 10, dtype=int)
    max_depth_list = np.arange(2, 10, 1)

    param_grid = list(product(
        n_estimators_list,
        max_depth_list,
    ))

    best_significance = -np.inf
    best_params = None

    for n_estimators, max_depth in tqdm(param_grid):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_leaves": 0,
            "objective": "binary:logistic",
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
#        print(f"Testing params: {params}")
        params = {'n_estimators': np.int64(215), 'max_depth': np.int64(5), 'max_leaves': np.int64(0), 'objective': 'binary:logistic', 'use_label_encoder': False, 'eval_metric': 'logloss'}

        significance = evaluate_significance(
            params,
            train_data, train_labels, train_weights,
            val_data, val_labels, val_weights
        )
        
        
#        print(f"Significance: {significance:.4f}")
        if significance > best_significance:
            best_significance = significance
            best_params = params
        break

    print("\nBest parameters:")
    print(best_params)
    print(f"Best significance: {best_significance:.4f}")