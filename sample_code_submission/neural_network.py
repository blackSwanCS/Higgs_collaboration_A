from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import os
import joblib
from tensorflow.keras.models import load_model
class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        self.model.add(Dense(100, input_dim=n_dim, activation="relu"))
        #self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=32, verbose=2
        )

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()
    
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
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "keras_model.h5"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.model = load_model(os.path.join(path, "keras_model.h5"))
        obj.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        return obj