import tensorflow as tf

tf.config.run_functions_eagerly(True)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os


class NeuralNetwork:
    """
    This class implements a neural network classifier.
    """

    def __init__(self, train_data=None):
        self.model = None
        self.scaler = StandardScaler()

        if train_data is not None:
            self._initialize_model(train_data)

    def _initialize_model(self, train_data):
        """Initialize the model architecture."""
        self.model = Sequential()
        n_dim = train_data.shape[1]

        self.model.add(Dense(100, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def fit(self, train_data, y_train, weights_train=None):
        # Fit the scaler on the training data
        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        # Fit the model using the transformed data
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=5, verbose=2
        )

    def predict(self, test_data, labels=None, weights=None):
        test_data = self.scaler.transform(test_data)
        predictions = self.model.predict(test_data).flatten().ravel()

        # Store predictions for significance calculation
        self.__predicted_data = predictions
        # Optionally store test labels and weights if provided
        if labels is not None:
            self.__test_labels = labels
        if weights is not None:
            self.__test_weights = np.asarray(weights)
        return predictions

    def significance(self, test_labels=None, test_weights=None):
        if test_labels is not None:
            self.__test_labels = test_labels
        if test_weights is not None:
            self.__test_weights = np.asarray(test_weights)
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
            s_hist, _ = np.histogram(
                y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
            )
            b_hist, _ = np.histogram(
                y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
            )
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]
            significance = __amsasimov(s_cumul, b_cumul)
            return significance

        vamsasimov_xgb = __significance_vscore(
            y_true=self.__test_labels,
            y_score=self.__predicted_data,
            sample_weight=self.__test_weights,
        )

        plt.plot(np.linspace(0, 1.0, 100), vamsasimov_xgb, label="AMS Significance")
        plt.xlabel("Score")
        plt.ylabel("Significance")
        return np.max(vamsasimov_xgb)
