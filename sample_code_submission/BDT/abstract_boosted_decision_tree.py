# library for calculation
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve

# library for structure
from enum import Enum, auto
import warnings
from abc import ABC, abstractmethod
import json

# constants
from constants import *

# Curves
import matplotlib.pyplot as plt


class BDT_Status(Enum):
    """
    Enum representing the status of the Boosted Decision Tree
    """

    NOT_FITTED = auto()
    FITTED = auto()
    PREDICTED = auto()


class AbstractBoostedDecisionTree(ABC):

    def __init__(self, name):
        self._scaler = StandardScaler()
        self.__name = name
        self.__status = BDT_Status.NOT_FITTED

        self._model = None
        self._labels = None
        self._weights = None
        self._predicted_data = None
        self.__test_labels = None
        self.__test_weights = None

    @abstractmethod
    def fit(self, train_data, labels, weights=None):
        if self.__status != BDT_Status.NOT_FITTED:
            warnings.warn("Model has already been fitted, skipping fiting", UserWarning)
            return True

        self._scaler.fit_transform(train_data)
        self._labels = labels
        self._weights = weights
        self.__status = BDT_Status.FITTED
        return False

    @abstractmethod
    def predict(self, test_data, labels=None, weights=None):
        pass

    @abstractmethod
    def predict_full_output(self, test_data, labels=None, weights=None):
        if self.__status == BDT_Status.NOT_FITTED:
            raise ValueError(
                "Model has not been fitted yet. Please call fit() before predict()."
            )
        self.__handle_input_weight_and_labels(labels, weights)
        self.__status = BDT_Status.PREDICTED

    def vamsasimov(self, test_labels=None, test_weights=None):
        """
        Calculate the tab of significance  for differents treshold of the predicted data using the AMS (A More Sensitive) method.
        """

        if self.__status != BDT_Status.PREDICTED:
            raise ValueError(
                "Model has not been fitted or predict yet. Please call fit() and predict() before significance()."
            )

        self.__handle_input_weight_and_labels(test_labels, test_weights)
        if self.__test_labels is None:
            raise ValueError(
                "True labels for test data are not available. Please provide them when calling predict()."
            )

        def amsasimov(s_in, b_in):
            s = np.copy(s_in)
            b = np.copy(b_in)
            s = np.where((b_in == 0), 0.0, s_in)
            b = np.where((b_in == 0), 1.0, b)
            ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
            ams = np.where((s < 0) | (b < 0), np.nan, ams)
            if np.isscalar(s_in):
                return float(ams)
            else:
                return ams

        def significance_vscore(y_true, y_score, sample_weight=None):
            if sample_weight is None:
                sample_weight = np.full(len(y_true), 1.0)
            else:
                sample_weight = np.asarray(sample_weight)
            bins = np.linspace(0, 1.0, 101)
            s_hist, bin_edges = np.histogram(
                y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
            )
            b_hist, bin_edges = np.histogram(
                y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
            )
            print(s_hist, b_hist)
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]
            significance = amsasimov(s_cumul, b_cumul)
            return significance

        vamsasimov = significance_vscore(
            y_true=self.__test_labels,
            y_score=self._predicted_data,
            sample_weight=self.__test_weights,
        )
        return vamsasimov

    def significance(self, test_labels=None, test_weights=None):
        """
        Calculate the significance of the predicted data using the AMS (A More Sensitive) method.
        """
        return np.max(self.vamsasimov(test_labels, test_weights))

    def auc(self, test_labels=None, test_weights=None):
        """
        Calculate the Area Under the Curve (AUC) for the predicted data.
        """

        if self.__status != BDT_Status.PREDICTED:
            raise ValueError(
                "Model has not been fitted or predicted yet. Please call fit() and predict() before auc()."
            )

        # handle the case when labels and weights aren't provided during prediction (in the model class)
        self.__handle_input_weight_and_labels(test_labels, test_weights)

        return roc_auc_score(
            y_true=self.__test_labels,
            y_score=self._predicted_data,
            sample_weight=self.__test_weights,
        )

    def roc_curve(self, test_labels=None, test_weights=None):
        if self.__status != BDT_Status.PREDICTED:
            raise ValueError(
                "Model has not been fitted or predicted yet. Please call fit() and predict() before auc()."
            )

        # handle the case when labels and weights aren't provided during prediction (in the model class)
        self.__handle_input_weight_and_labels(test_labels, test_weights)

        fpr_xgb, tpr_xgb, _ = roc_curve(
            y_true=self.__test_labels,
            y_score=self._predicted_data,
            sample_weight=self.__test_weights,
        )
        auc_test = self.auc(test_labels, test_weights)
        plt.plot(
            fpr_xgb,
            tpr_xgb,
            color="darkgreen",
            lw=2,
            label="XGBoost (AUC  = {:.3f})".format(auc_test),
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Background Efficiency")
        plt.ylabel("Signal Efficiency")
        plt.title(f"ROC curve for {self.name}")
        plt.legend(loc="lower right")
        plt.show()

    def significance_curve(self, test_labels=None, test_weights=None):
        vams = self.vamsasimov(test_labels, test_weights)
        x = np.linspace(0, 1, num=len(vams))
        significance = np.max(vams)
        plt.plot(x, vams, label="(Z = {:.2f})".format(significance))
        plt.title(f"BDT Significance for {self.name} ")
        plt.xlabel("Threshold")
        plt.ylabel("Significance")
        plt.legend()
        plt.show()

    def __handle_input_weight_and_labels(self, labels=None, weights=None):
        """
        Handle the input labels and weights for training or prediction.
        """
        if labels is not None:
            self.__test_labels = np.asarray(labels)

        if weights is not None:
            self.__test_weights = np.asarray(weights)

    @abstractmethod
    def load_model(self):
        if self.__status != BDT_Status.NOT_FITTED:
            raise ValueError(
                "Model has already been fitted, please create a new instance to load a new model."
            )
        with open(BEST_BDT_MODEL_PATH + "_scaler.json", "r") as f:
            scaler_params = json.load(f)
        self._scaler = StandardScaler()
        self._scaler.mean_ = scaler_params["mean_"]
        self._scaler.scale_ = scaler_params["scale_"]
        self.__status = BDT_Status.FITTED

    @abstractmethod
    def save(self):
        if self.__status == BDT_Status.NOT_FITTED:
            raise ValueError(
                "Model has not been fitted yet. Please call fit() before save()."
            )
        scaler_params = {
            "mean_": self._scaler.mean_.tolist(),
            "scale_": self._scaler.scale_.tolist(),
        }
        with open(BEST_BDT_MODEL_PATH + "_scaler.json", "w") as f:
            json.dump(scaler_params, f)
