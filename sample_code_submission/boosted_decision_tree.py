from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import multiprocessing


class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """


    def __init__(self, params):
        # Initialize the model and scaler
        self.__model = XGBClassifier(**params)
        self.__scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None):

        # Set the scaler to look at the data
        self.__scaler.fit_transform(train_data)

        # Resize the data using the scaler (mean 0, std 1)
        X_train_data = self.__scaler.transform(train_data)

        # Fit the model to the training data
        self.__model.fit(X_train_data, labels, weights)

    def predict(self, test_data):

        # Resize the test data using the scaler
        test_data = self.__scaler.transform(test_data)

        # Predict the probabilities for the positive class
        return self.__model.predict_proba(test_data)[:, 1]

    def significance(self, test_data, labels, weights=None):
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
            bins = np.linspace(0, 1., 101)
            s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1])
            b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0])
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]
            significance=__amsasimov(s_cumul,b_cumul)
            max_value = np.max(significance)
            return significance
        y_pred_xgb = self.predict(test_data)
        vamsasimov_xgb=__significance_vscore(y_true=labels, y_score=y_pred_xgb, sample_weight=weights)
        significance_xgb = np.max(vamsasimov_xgb)
        return significance_xgb

