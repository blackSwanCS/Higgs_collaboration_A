from boosted_decision_tree import AbstractBoostedDecisionTree
from tabpfn import PFNClassifier


class PFN_boosted_decision_tree(AbstractBoostedDecisionTree):
    """
    This class implements a boosted decision tree model using PFN's implementation.
    """

    def __init__(self, params=None):
        super().__init__("PFNBoostedDecisionTree")
        if params is None:
            self._model = PFNClassifier()
        else:
            self._model = PFNClassifier(**params)

    def fit(self, train_data, labels, weights=None):
        if super().fit(train_data, labels, weights):
            return
        self._model.fit(
            self._scaler.transform(train_data),
            self._labels,
            sample_weight=self._weights,
        )

    def predict(self, test_data, labels=None, weights=None):
        return self.predict_full_output(test_data, labels, weights)[:, 1]

    def predict_full_output(self, test_data, labels=None, weights=None):
        super().predict_full_output(test_data, labels, weights)
        return self._model.predict_proba(self._scaler.transform(test_data))
