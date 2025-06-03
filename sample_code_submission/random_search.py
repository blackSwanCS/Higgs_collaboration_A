import numpy as np
from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree

# Define the parameter search space
from random import randint, uniform, choice

def random_params():
    return {
        "n_estimators": randint(100, 1000),
        "learning_rate": round(uniform(0.01, 0.3), 3),
        "max_depth": randint(3, 10),
        "subsample": round(uniform(0.6, 1.0), 2),
        "colsample_bytree": round(uniform(0.6, 1.0), 2),
        "gamma": round(uniform(0, 5), 2),
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

def get_data():
    data = download_dataset("blackSwan_data")
    data.load_train_set()
    training_set = data.get_train_set()
    feature_keys = [k for k in training_set.keys() if k not in ("labels", "weights", "detailed_labels")]
    X = np.column_stack([training_set[k] for k in feature_keys])
    y = training_set["labels"]
    w = training_set["weights"]

    # Split into train/validation (e.g., 80/20 split)
    n = len(y)
    split = int(n * 0.8)
    train_data, val_data = X[:split], X[split:]
    train_labels, val_labels = y[:split], y[split:]
    train_weights, val_weights = w[:split], w[split:]

    return train_data, train_labels, train_weights, val_data, val_labels, val_weights

def evaluate_significance(params):
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()
    model = BoostedDecisionTree(params)
    model.fit(train_data, train_labels, train_weights)
    # Use the model's significance method, passing validation labels and weights
    significance = model.significance(val_data, val_labels, weights=val_weights)
    return significance

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Use random parameters")
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0)
    args = parser.parse_args()

    if args.random:
        params = random_params()
        print("Random parameters:", params)
    else:
        params = {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "gamma": args.gamma,
            "objective": "binary:logistic",
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        print("User parameters:", params)

    significance = evaluate_significance(params)
    print(f"Significance: {significance:.4f}")
    
# either run with "py /sample_code_submission/random_search.py --random" to use random parameters
# or with "py /sample_code_submission/random_search.py --n_estimators 500 --learning_rate 0.05 --max_depth 4 --subsample 0.8 --colsample_bytree 0.8 --gamma 1"