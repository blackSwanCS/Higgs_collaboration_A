# library for calculation
from xgboost import XGBClassifier

# constants
from BDT.constants import *

# Objects
from BDT.abstract_boosted_decision_tree import AbstractBoostedDecisionTree


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

    def load_model(self, *args):
        if super().load_model():
            return
        self._model.load_model(BEST_BDT_MODEL_PATH + ".json")

    def save(self):
        super().save()
        self._model.save_model(BEST_BDT_MODEL_PATH + ".json")
