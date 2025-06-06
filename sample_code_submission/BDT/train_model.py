from get_data import get_data
from boosted_decision_tree import get_model_with_best_param
import time 

if __name__ == "__main__":
    train_data, train_labels, train_weights, val_data, val_labels, val_weights=get_data()
    model=get_model_with_best_param()
    starting_time = time.time()
    model.fit(train_data, train_labels, train_weights)
    training_time = time.time( ) - starting_time
    print("Training time:",training_time)
    model.predict(train_data, labels=train_labels, weights=train_weights)
    print("auc train:",model.auc())
    model.predict(val_data, labels=val_labels, weights=val_weights)
    print ("auc test:",model.auc())
    print("Z :",model.significance())
    model.roc_curve()
    model.significance_curve(val_labels)
