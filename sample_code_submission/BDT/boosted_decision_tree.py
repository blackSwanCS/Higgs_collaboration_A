from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import multiprocessing
from enum import Enum, auto


class BDT_Status(Enum):
    NOT_FITTED = auto()
    FITTED = auto()
    PREDICTED = auto()


class BoostedDecisionTree:
    """
    This class implements a boosted decision tree model using XGBoost
    """

    def __init__(self, params=None):
        # Initialize the model and scaler
        if params is None:
            params = {
                "n_estimators": 210,
                "max_depth": 5,
                "max_leaves": 0,
                "objective": "binary:logistic",
                "use_label_encoder": False,
                "eval_metric": "logloss"
            }
        self.__model = XGBClassifier(**params, n_jobs=multiprocessing.cpu_count())
        self.__scaler = StandardScaler()        
        self.__status = BDT_Status.NOT_FITTED

    def fit(self, train_data, labels, weights=None):
        if self.__status != BDT_Status.NOT_FITTED:
            raise ValueError("Model has already been fitted. Please create a new instance to fit again.")
        
        self.__scaler.fit_transform(train_data)
        self.__train_data = self.__scaler.transform(train_data)
        self.__labels = labels
        self.__weights = weights

        self.__model.fit(self.__train_data, self.__labels, self.__weights)

        self.__status = BDT_Status.FITTED
    
    def predict(self, test_data, labels=None, weights=None):
        if self.__status == BDT_Status.NOT_FITTED:
            raise ValueError("Model has not been fitted yet. Please call fit() before predict().")
        
        self.__test_data = self.__scaler.transform(test_data)
        self.__predicted_data = self.__model.predict_proba(self.__test_data)[:, 1]
        self.__test_labels = labels
        if weights is not None:
            self.__test_weights = np.asarray(weights)
        else:
            self.__test_weights = None
        self.__status = BDT_Status.PREDICTED
        return self.__predicted_data

    def significance(self):
        if self.__status != BDT_Status.PREDICTED:
            raise ValueError("Model has not been fitted or predict yet. Please call fit() and predict() before significance().")
        if self.__test_labels is None:
            raise ValueError("True labels for test data are not available. Please provide them when calling predict().")
        
        def __amsasimov(s_in,b_in):
            s=np.copy(s_in)
            b=np.copy(b_in)
            s=np.where( (b_in == 0) , 0., s_in)
            b=np.where( (b_in == 0) , 1., b)
            ams = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
            ams=np.where( (s < 0)  | (b < 0), np.nan, ams)
            if np.isscalar(s_in):
                return float(ams)
            else:
                return  ams
        def __significance_vscore(y_true, y_score, sample_weight=None):
            if sample_weight is None:
                sample_weight = np.full(len(y_true), 1.)
            else:
                sample_weight = np.asarray(sample_weight)
            bins = np.linspace(0, 1., 101)
            s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1])
            b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0])
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]
            significance=__amsasimov(s_cumul,b_cumul)
            return significance
        vamsasimov_xgb=__significance_vscore(
            y_true=self.__test_labels,
            y_score=self.__predicted_data,
            sample_weight=self.__test_weights
        )
        return np.max(vamsasimov_xgb)

