from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


class BoostedDecisionTree:
    """
    This class implements a boosted devision tree model
    """

    def __init__(self, train_data):
        # Initialize the model and scaler
        self.model = XGBClassifier()
        self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None):

        # Set the scaler to look at the data
        self.scaler.fit_transform(train_data)

        # Resize the data using the scaler (mean 0, std 1)
        X_train_data = self.scaler.transform(train_data)

        # Fit the model to the training data
        self.model.fit(X_train_data, labels, weights)

    def predict(self, test_data):

        # Resize the test data using the scaler
        test_data = self.scaler.transform(test_data)

        # Predict the probabilities for the positive class
        return self.model.predict_proba(test_data)[:, 1]
