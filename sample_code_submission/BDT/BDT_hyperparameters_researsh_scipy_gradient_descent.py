import numpy as np
from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree
from itertools import product
from tqdm import tqdm
from scipy.optimize import minimize

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
    return significance

def evaluate_auc(params, train_data, train_labels, train_weights, val_data, val_labels, val_weights):
    model = BoostedDecisionTree(params)
    model.fit(train_data, train_labels, train_weights)
    model.predict(val_data, labels=val_labels, weights=val_weights) 
    return(model.auc()) 

def loss(input):
            params = {
                "n_estimators": input[0],
                "max_depth": input[1],
                "eta":input[2],
                "subsample":input[3],
                "max_leaves": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss"
            }
            model = BoostedDecisionTree(params)
            model.fit(train_data, train_labels, train_weights)
            model.predict(val_data, labels=val_labels, weights=val_weights) 
            model.auc() 
            return 1200/model.significance() - np.log(model.auc())
   


if __name__ == "__main__":
    # Load data once
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()

    eps = 0.01
    alpha = 1
    
    names = ["n_estimators","max_depth","eta","subsample"]
    pas = [1,1,0.01,0.01]
    input = [450,7,0.15525,0.9375]
    min_input = [4,1,0.0,0.0]
    max_input = [np.inf,np.inf,1.0,1.0]
    approx = [0,0,2,2]
    
    constraints = [
    {'type': 'ineq', 'fun': lambda input: input[0] - 2},  # x1 >= 3
    {'type': 'ineq', 'fun': lambda input: input[1] - 2},  # x2 >= 3
    {'type': 'ineq', 'fun': lambda input: input[2]},      # y1 >= 0
    {'type': 'ineq', 'fun': lambda input: 1 - input[2]},  # y1 <= 1
    {'type': 'ineq', 'fun': lambda input: input[3]},      # y2 >= 0
    {'type': 'ineq', 'fun': lambda input: 1 - input[3]}   # y2 <= 1
]   
    # Appeler la fonction de minimisation
    result = minimize(loss, input, constraints=constraints)

    # Afficher les résultats
    print("Paramètres optimaux :", result.x)
    print("Valeur minimale de la fonction :", result.fun)