from abstract_boosted_decision_tree import AbstractBoostedDecisionTree
from tabpfn import TabPFNClassifier
from get_data import get_data
from time import time
import torch


class PFN_boosted_decision_tree(AbstractBoostedDecisionTree):
    """
    PFN model optimisé pour GPU.
    """

    def __init__(self):
        super().__init__("PFNBoostedDecisionTree")
        # Vérifie si CUDA est dispo, sinon fallback CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = TabPFNClassifier(ignore_pretraining_limits=True, device=device)

    def fit(self, train_data, labels, weights=None):
        if super().fit(train_data, labels, weights):
            return
        self._model.fit(
            self._scaler.transform(train_data),
            self._labels,
        )

    def predict(self, test_data, labels=None, weights=None):
        self._predicted_data = self.predict_full_output(test_data, labels, weights)[
            :, 1
        ]
        return self._predicted_data

    def predict_full_output(self, test_data, labels=None, weights=None):
        super().predict_full_output(test_data, labels, weights)
        return self._model.predict_proba(self._scaler.transform(test_data))

    def load_model(self):
        return super().load_model()

    def save(self):
        return super().save()


if __name__ == "__main__":
    train_data, train_labels, train_weights, val_data, val_labels, val_weights = (
        get_data()
    )

    # Utilise tout le dataset pour l'entraînement (sauf si test rapide)
    # train_data = train_data[:4000]
    # train_labels = train_labels[:4000]
    # train_weights = train_weights[:4000]
    # val_data = val_data[:2000]
    # val_labels = val_labels[:2000]
    # val_weights = val_weights[:2000]

    model = PFN_boosted_decision_tree()
    t0 = time()
    model.fit(train_data, train_labels, train_weights)
    t1 = time()
    model.save()
    print(f"Fitting time: {t1 - t0:.2f} seconds")
    model.predict(val_data, val_labels, val_weights)
    significance = model.significance()
    auc = model.auc(val_labels, val_weights)
    print(f"AUC: {auc:.4f}")
    print(f"Significance: {significance:.4f}")
