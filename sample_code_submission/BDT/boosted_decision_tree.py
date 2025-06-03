from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import multiprocessing
from enum import Enum, auto
import warnings
import json

BEST_BDT_MODEL_PATH = "best_bdt_model"
THREADS_NUMBER = multiprocessing.cpu_count()


def get_best_model():
    model = BoostedDecisionTree()
    model.load_model(BEST_BDT_MODEL_PATH)
    return model
    #params = {'n_estimators': np.int64(236), 'max_depth': np.int64(9), 'max_leaves': np.int64(0), 'objective': 'binary:logistic', 'use_label_encoder': False, 'eval_metric': 'logloss'}
    #return BoostedDecisionTree(params)


class BDT_Status(Enum):
    NOT_FITTED = auto()
    FITTED = auto()
    PREDICTED = auto()


class BoostedDecisionTree:
    """
    This class implements a boosted decision tree model using XGBoost
    """

    def __init__(self, params=None):
        print("Initializing Boosted Decision Tree model...")
        # Initialize the model and scaler
        if params is None:
            self.__model = XGBClassifier(n_jobs=THREADS_NUMBER)
        else:
            self.__model = XGBClassifier(**params, n_jobs=THREADS_NUMBER)
        self.__scaler = StandardScaler()
        self.__status = BDT_Status.NOT_FITTED

    def fit(self, train_data, labels, weights=None):
        print("Fitting Boosted Decision Tree model...")
        if self.__status != BDT_Status.NOT_FITTED:
            warnings.warn("Model has already been fitted, skipping fiting", UserWarning)
            return

        self.__scaler.fit_transform(train_data)
        self.__train_data = self.__scaler.transform(train_data)
        self.__labels = labels
        self.__weights = weights

        self.__model.fit(self.__train_data, self.__labels, self.__weights)

        self.__status = BDT_Status.FITTED

    def predict(self, test_data, labels=None, weights=None):
        if self.__status == BDT_Status.NOT_FITTED:
            raise ValueError(
                "Model has not been fitted yet. Please call fit() before predict()."
            )

        self.__test_data = self.__scaler.transform(test_data)
        self.__predicted_data = self.__model.predict_proba(self.__test_data)[:, 1]
        self.__test_labels = labels
        if weights is not None:
            self.__test_weights = np.asarray(weights)
        else:
            self.__test_weights = None
        self.__status = BDT_Status.PREDICTED
        return self.__predicted_data

    def significance(self, test_labels):
        self.__test_labels = test_labels
        if self.__status != BDT_Status.PREDICTED:
            raise ValueError(
                "Model has not been fitted or predict yet. Please call fit() and predict() before significance()."
            )
        if self.__test_labels is None:
            raise ValueError(
                "True labels for test data are not available. Please provide them when calling predict()."
            )

        def __amsasimov(s_in, b_in):
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

        def __significance_vscore(y_true, y_score, sample_weight=None):
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
            significance = __amsasimov(s_cumul, b_cumul)
            return significance

        vamsasimov_xgb = __significance_vscore(
            y_true=self.__test_labels,
            y_score=self.__predicted_data,
            sample_weight=self.__test_weights,
        )
        return np.max(vamsasimov_xgb)

    def auc(self):
        return roc_auc_score(
            y_true=self.__test_labels,
            y_score=self.__predicted_data,
            sample_weight=self.__test_weights,
        )

    def load_model(self, model_path):
        """
        Load a pre-trained model from the specified path.
        """
        self.__model.load_model(model_path + ".json")

        print(model_path + "_scaler.json")
        with open(model_path + "_scaler.json", "r") as f:
            scaler_params = json.load(f)
        self.__scaler = StandardScaler()
        self.__scaler.mean_ = scaler_params["mean_"]
        self.__scaler.scale_ = scaler_params["scale_"]

        self.__status = BDT_Status.FITTED

    def save(self):
        if self.__status == BDT_Status.NOT_FITTED:
            raise ValueError(
                "Model has not been fitted yet. Please call fit() before save()."
            )
        self.__model.save_model(BEST_BDT_MODEL_PATH + ".json")
        scaler_params = {
            "mean_": self.__scaler.mean_.tolist(),
            "scale_": self.__scaler.scale_.tolist(),
        }
        with open(BEST_BDT_MODEL_PATH + "_scaler.json", "w") as f:
            json.dump(scaler_params, f)
