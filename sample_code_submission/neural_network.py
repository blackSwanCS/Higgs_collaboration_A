import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal, GlorotNormal
from tensorflow.keras.callbacks import EarlyStopping


class NeuralNetwork:
    """
    Réplique exacte du modèle de référence décrit dans la thèse :
      • 4 couches denses de 100 neurones (init He-normal) + ReLU
      • Couche de sortie 1 neurone (init Glorot-normal) + sigmoïde
      • Adam LR = 2e-3, β1 = 0.9, β2 = 0.999, ε = 1e-8
      • Mini-batch 256, early-stopping patience 50 (val_loss)
    """

    def __init__(self, train_data):
        self.model = Sequential()
        n_dim = train_data.shape[1]

        # 4 couches cachées
        for i in range(4):
            self.model.add(
                Dense(
                    100,
                    input_dim=n_dim if i == 0 else None,
                    activation="relu",
                    kernel_initializer=HeNormal(),
                    bias_initializer="zeros",
                )
            )

        # Couche de sortie
        self.model.add(
            Dense(
                1,
                activation="sigmoid",
                kernel_initializer=GlorotNormal(),
                bias_initializer="zeros",
            )
        )

        # Optimiseur Adam
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=2e-3,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            ),
            metrics=["accuracy"],
        )

        # Normalisation
        self.scaler = StandardScaler()

        # Early-stopping
        self.early_stop = EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True, verbose=1
        )

    def fit(self, train_data, y_train, weights_train=None,
            val_split=0.1, max_epochs=1000):
        """Entraîne le réseau sur les données normalisées."""
        self.scaler.fit(train_data)
        X_train = self.scaler.transform(train_data)

        self.model.fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            epochs=max_epochs,
            batch_size=256,
            validation_split=val_split,
            callbacks=[self.early_stop],
            verbose=2,
            shuffle=True,
        )

    def predict(self, test_data):
        """Renvoie les probabilités prédites (flatten 1-D)."""
        X_test = self.scaler.transform(test_data)
        return self.model.predict(X_test, verbose=0).flatten()

    # --- Métrique AMS Asimov -------------------------------------------------
    @staticmethod
    def _amsasimov(s_in, b_in):
        s = np.copy(s_in)
        b = np.where(b_in == 0, 1., b_in)  # évite log(0)
        ams = np.sqrt(2 * ((s + b) * np.log(1.0 + s / b) - s))
        ams = np.where((s < 0) | (b < 0), np.nan, ams)
        return float(ams) if np.isscalar(s_in) else ams

    @staticmethod
    def _significance_curve(y_true, y_score, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y_true, dtype=float)

        bins = np.linspace(0, 1.0, 101)
        s_hist, _ = np.histogram(
            y_score[y_true == 1], bins=bins,
            weights=sample_weight[y_true == 1]
        )
        b_hist, _ = np.histogram(
            y_score[y_true == 0], bins=bins,
            weights=sample_weight[y_true == 0]
        )
        s_cumul = np.cumsum(s_hist[::-1])[::-1]
        b_cumul = np.cumsum(b_hist[::-1])[::-1]
        return NeuralNetwork._amsasimov(s_cumul, b_cumul)

    def significance(self, test_data, labels, weights=None):
        """Retourne la meilleure significativité AMS sur l’ensemble test."""
        y_pred = self.predict(test_data)
        curve = self._significance_curve(labels, y_pred, weights)
        return np.nanmax(curve)
