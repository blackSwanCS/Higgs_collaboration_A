from HiggsML.datasets import download_dataset
import numpy as np

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
