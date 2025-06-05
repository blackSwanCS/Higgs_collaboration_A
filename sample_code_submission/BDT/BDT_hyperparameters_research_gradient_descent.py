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

def loss(significance,auc):
            return 1200/significance - np.log(auc)
   


if __name__ == "__main__":
    # Load data once
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()

    memoisation = []

    gradient = np.inf
    eps = 0.01
    alpha = 1
    
    names = ["n_estimators","max_depth","eta","subsample"]
    pas = [1,1,0.01,0.01]
    input = [450,7,0.15525,0.9375]
    min_input = [4,1,0.0,0.0]
    max_input = [np.inf,np.inf,1.0,1.0]
    approx = [0,0,2,2]
    
    params = {
                "n_estimators": input[0],
                "max_depth": input[1],
                "eta":input[2],
                "subsample":input[3],
                "max_leaves": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss"
            }
    
    epoch = 0
    
    #for n_estimators, max_depth in tqdm(param_grid):
    while np.linalg.norm(np.array(gradient)) > eps or epoch<300:
        
        for i in range(len(input)):
            params[names[i]]=input[i]
        
#        print(f"Testing params: {params}")
        significance = evaluate_significance(
            params,
            train_data, train_labels, train_weights,
            val_data, val_labels, val_weights
        )
#        print(f"Significance: {significance:.4f}")
        auc = evaluate_auc(
            params,
            train_data, train_labels, train_weights,
            val_data, val_labels, val_weights
        )
        
        loss_now = loss(significance,auc)
        gradient = []
        
        for i in range(len(input)):
            params_plus_i = params 
            params_plus_i[names[i]]=min(max(input[i]+pas[i], max_input[i]), min_input[i])
            loss_plus_i = loss(
                evaluate_significance(
                params_plus_i,
                train_data, train_labels, train_weights,
                val_data, val_labels, val_weights
                ),
                evaluate_auc(
                params_plus_i,
                train_data, train_labels, train_weights,
                val_data, val_labels, val_weights
            ))
            gradient.append(loss_plus_i-loss_now)
        
        print("\n__________________________________________________________________________________________________________________________________________\n")
        print("epoch",epoch,"| input :",input,"| significance",significance,"| auc",auc,"|loss", loss_now,"gradiant",gradient)
        print("__________________________________________________________________________________________________________________________________________\n")
        
        ''' Version 1
        for i in range(4):
            input[i] = max(min(input[i] - pas[i]*int(math.copysign(alpha,gradient[i])),max_input[i]-pas[i]),min_input[i]+pas[i])
        '''
        
        # Version 2
        i_max = 0
        for i in range(1,4):
            if abs(gradient[i])>=abs(gradient[i_max]):
                i_max = i
        input[i_max] = max(min(round(input[i_max] - pas[i_max]*int(math.copysign(alpha,gradient[i_max])),max_input[i_max]-pas[i_max]),approx[i]),min_input[i_max]+pas[i_max])    
        
        epoch +=1