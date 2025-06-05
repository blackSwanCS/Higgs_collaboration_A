from boosted_decision_tree import get_best_model
from donnee import get_data
from xgb_boosted_decision_tree import XGBBoostedDecisionTree
import numpy as np
import matplotlib.pyplot as plt
from time import time

train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()

params_best = {
        "n_estimators": np.int64(450),
        "max_depth": np.int64(7),
        "max_leaves": np.int64(0),
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "eta": np.float64(0.15525),
        "subsample": np.float64(0.9375),
    }

def curve(params=None):
    if params==params_best:
        model = get_best_model()
        model.predict(val_data, labels=val_labels, weights=val_weights)
        model.roc_curve()
        model.significance_curve(val_labels)
    else :
        model=XGBBoostedDecisionTree(params)
        model.fit(train_data,train_labels,train_weights)
        model.predict(val_data, labels=val_labels, weights=val_weights)
        model.roc_curve()
        model.significance_curve(val_labels)


def learning_curve(params,train_data, train_labels, train_weights, val_data, val_labels, val_weights):
        train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        ntrains=[]
        train_aucs=[]
        times=[]

        for train_size in train_sizes:
            mod=XGBBoostedDecisionTree(params)
            ntrain=int(train_size*len(train_data))
            print("training with ",ntrain," events")
            ntrains+=[ntrain]
            starting_time = time()

            # train using the first ntrain event of the training dataset
            mod.fit(train_data[:ntrain,],train_labels[:ntrain],train_weights[:ntrain])
            training_time = time( ) - starting_time
            times+=[training_time]

            # score on test dataset (always the same)
            # score on the train dataset
            mod.predict(val_data, labels=val_labels, weights=val_weights)
            auc_train_xgb = mod.auc()
            train_aucs+=[auc_train_xgb]
        fig, ax1 = plt.subplots()

        ax1.plot(ntrains,train_aucs , 'b+', label="Train_AUC")
        ax1.set_xlabel("Ntraining")
        ax1.set_ylabel("AUC", color='b')

        ax2 = ax1.twinx()
        ax2.plot(ntrains, times, 'r+', label="Time")
        ax2.set_ylabel("Time (s)", color='r')
        
        plt.title("Learning curve")
        plt.show()
        

params = {
        "n_estimators": np.int64(450),
        "max_depth": np.int64(7),
        "max_leaves": np.int64(0),
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "eta": np.float64(0.15525),
        "subsample": np.float64(0.9375),
    }

#curve(params)
learning_curve(params_best,train_data, train_labels, train_weights, val_data, val_labels, val_weights)