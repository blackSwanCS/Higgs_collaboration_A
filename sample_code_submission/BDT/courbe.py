from boosted_decision_tree import get_best_model
from get_data import get_data
from xgb_boosted_decision_tree import XGBBoostedDecisionTree
import numpy as np
import matplotlib.pyplot as plt
from time import time

train_data, train_labels, train_weights, val_data, val_labels, val_weights = get_data()
model = get_best_model()
model.predict(val_data, labels=val_labels, weights=val_weights)
model.roc_curve()
