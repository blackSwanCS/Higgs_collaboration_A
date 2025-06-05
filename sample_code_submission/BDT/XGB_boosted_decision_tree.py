# library for calculation
from xgboost import XGBClassifier

import os
import sys


def find_and_add_module_path(filename):
    """Ajoute à sys.path le dossier contenant filename, en remontant jusqu'à 3 niveaux."""
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(3):
        candidate = os.path.join(cur, filename)
        if os.path.isfile(candidate):
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return
        cur = os.path.dirname(cur)


find_and_add_module_path("constants.py")
find_and_add_module_path("abstract_boosted_decision_tree.py")

# constants
from constants import *

# Objects
from abstract_boosted_decision_tree import AbstractBoostedDecisionTree


class XGBBoostedDecisionTree(AbstractBoostedDecisionTree):
    """
    This class implements a boosted decision tree model using XGBoost
    """

    def __init__(self, params=None):
        super().__init__("XGBBoostedDecisionTree")
        if params is None:
            self._model = XGBClassifier(n_jobs=THREADS_NUMBER, device="cuda")
        else:
            self._model = XGBClassifier(**params, n_jobs=THREADS_NUMBER, device="cuda")

    def fit(self, train_data, labels, weights=None):
        if super().fit(train_data, labels, weights):
            return
        self._model.fit(self._scaler.transform(train_data), self._labels, self._weights)

    def predict_proba(self, test_data):
        return self._model.predict_proba(test_data)

    def predict(self, test_data, labels=None, weights=None):
        self._predicted_data = self.predict_full_output(test_data, labels, weights)[
            :, 1
        ]
        return self._predicted_data

    def predict_full_output(self, test_data, labels=None, weights=None):
        super().predict_full_output(test_data, labels, weights)
        return self._model.predict_proba(self._scaler.transform(test_data))

    def load_model(self):
        super().load_model()
        self._model.load_model(BEST_BDT_MODEL_PATH + ".json")

    def save(self):
        super().save()
        self._model.save_model(BEST_BDT_MODEL_PATH + ".json")
