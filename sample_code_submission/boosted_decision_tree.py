from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

class BoostedDecisionTree:
    """
    This class implements a boosted devision tree model
    """

    def __init__(self, train_data):
        # Initialize the model and scaler
        self.__model = XGBClassifier()
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

    def significance(self, test_data, weights=None):
        def __amsasimov(s_in,b_in): # asimov significance arXiv:1007.1727 eq. 97 (reduces to s/sqrt(b) if s<<b)
            # if b==0 ams is undefined, but return 0 without warning for convenience (hack)
            s=np.copy(s_in)
            b=np.copy(b_in)
            s=np.where( (b_in == 0) , 0., s_in)
            b=np.where( (b_in == 0) , 1., b)

            ams = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
            ams=np.where( (s < 0)  | (b < 0), np.nan, ams) # nan if unphysical values.
            if np.isscalar(s_in):
                return float(ams)
            else:
                return  ams
        def __significance_vscore(y_true, y_score, sample_weight=None):
            if sample_weight is None:
                # Provide a default value of 1.
                sample_weight = np.full(len(y_true), 1.)


            # Define bins for y_score, adapt the number as needed for your data
            bins = np.linspace(0, 1., 101)


            # Fills s and b weighted binned distributions
            s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1])
            b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0])


            # Compute cumulative sums (from the right!)
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]

            # Compute significance
            significance=__amsasimov(s_cumul,b_cumul)

            # Find the bin with the maximum significance
            max_value = np.max(significance)

            return significance
        y_test = self.__scaler.transform(test_data)
        y_pred_xgb = self.predict(test_data)
        vamsasimov_xgb=__significance_vscore(y_true=y_test, y_score=y_pred_xgb, sample_weight=weights)
        significance_xgb = np.max(vamsasimov_xgb)
        return significance_xgb
