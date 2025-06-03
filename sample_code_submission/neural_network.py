from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal, GlorotNormal, Zeros
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np

class NeuralNetwork:
    """
    Implements the baseline neural network as described:
    - 4 hidden layers, 100 neurons each, He-normal init, ReLU, zero bias
    - Output layer: 1 neuron, sigmoid, Glorot-normal init, zero bias
    - Adam optimizer (lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    - Batch size 256, early stopping with patience=50
    """

    def __init__(self, train_data):
        self.model = Sequential()
        n_dim = train_data.shape[1]
        he_init = HeNormal()
        glorot_init = GlorotNormal()
        zero_bias = Zeros()

        # 4 hidden layers, 100 neurons each
        for i in range(4):
            self.model.add(Dense(
                100,
                activation="relu",
                kernel_initializer=he_init,
                bias_initializer=zero_bias,
                input_dim=n_dim if i == 0 else None
            ))

        # Output layer
        self.model.add(Dense(
            1,
            activation="sigmoid",
            kernel_initializer=glorot_init,
            bias_initializer=zero_bias
        ))

        adam = Adam(learning_rate=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(
            loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"]
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None, val_data=None, val_labels=None):
        self.scaler.fit(train_data)
        X_train = self.scaler.transform(train_data)
        callbacks = []
        if val_data is not None and val_labels is not None:
            X_val = self.scaler.transform(val_data)
            early_stop = EarlyStopping(
                monitor='val_loss', patience=50, restore_best_weights=True
            )
            callbacks.append(early_stop)
            self.model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                epochs=5,
                batch_size=256,
                verbose=2,
                validation_data=(X_val, val_labels),
                callbacks=callbacks
            )
        else:
            self.model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                epochs=500,
                batch_size=256,
                verbose=2
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