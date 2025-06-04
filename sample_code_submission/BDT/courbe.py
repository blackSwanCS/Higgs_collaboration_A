from boosted_decision_tree import get_best_model
from donnee import get_data

train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()
model = get_best_model()
model.predict(val_data, labels=val_labels, weights=val_weights)
model.roc_curve()
