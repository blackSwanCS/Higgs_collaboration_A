import numpy as np
from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree
from itertools import product
from tqdm import tqdm
import math

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

if __name__ == "__main__":
    # Load data once
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()

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
            significance = model.significance()
            auc = model.auc() 
            return (1200/significance - np.log(auc),significance,auc)

    # noms des différent paramètres à changer. C'est l'espace sur lequel on optimise
    names = ["n_estimators","max_depth","eta","subsample"]
    
    pas = [1,1,0.01,0.01]
    
    # point de départ
    best_input = [191,5,0.3,0.99]
    
    # Les extrémaux pour les paramètres
    min_input = [4,1,0.0,0.0]
    max_input = [np.inf,np.inf,1.0,1.0]
    
    #approx = [0,0,2,2]
    
    input = [0,0,0.0,0.0]
    epoch = 0
    
    eps=0.01
    los_min = float("inf"),0,0
    loss_here = 0,0,0
    loss_neighbor = float("inf"),0,0
    memoisation={}
    
    while los_min[0]-loss_here[0]>eps and epoch<300 and input!=best_input:
        input = best_input
        if not str(input) in memoisation:
            loss_here = loss(input)
            memoisation[str(input)] = loss_here
        else:
            loss_here = memoisation[str(input)]
            
        loss_min = loss_here
        for i in range(len(input)):
            if input[i]+pas[i]<max_input[i]:
                input[i]=input[i]+pas[i]
                if not str(input) in memoisation:
                    loss_neighbor = loss(input)
                    memoisation[str(input)]=loss_neighbor
                if loss_neighbor[0]<=loss_min[0]:
                    loss_min = loss_neighbor
                    best_input = input.copy()
                    best_input[i] = input[i]+pas[i]
                input[i] = input[i]-pas[i]

            if input[i]-pas[i]>min_input[i]:
                input[i]=input[i]-pas[i]
                if not str(input) in memoisation:
                    loss_neighbor = loss(input)
                    memoisation[str(input)]=loss_neighbor
                if loss_neighbor[0]<=loss_min[0]:
                    loss_min = loss_neighbor
                    best_input = input.copy()
                    best_input[i] = input[i]-pas[i]
                input[i] = input[i]+pas[i]
        
        print("\n_____________________________________________________________________________________________________________________________\n")
        print("epoch",epoch,"| input :",best_input,"| loss", loss_min[0],"| significance", loss_min[1],"| auc", loss_min[2])
        print("_____________________________________________________________________________________________________________________________\n")
        
        epoch +=1