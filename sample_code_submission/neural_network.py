import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
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

        self.model.add(Dense(10, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def fit(self, train_data, y_train, weights_train=None):
        """Train the model."""
        if self.model is None:
            raise ValueError("Model is not initialized. Ensure `_initialize_model` is called or load a saved model.")

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=32, verbose=2
        )

    def predict(self, test_data):
        """Make predictions."""
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()

    def save_model(self, path):
        """Save the trained model and scaler to the specified path."""
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, "model.h5")
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        scaler_path = os.path.join(path, "scaler.pkl")
        import joblib
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    def load_model(self, path):
        """Load the trained model and scaler from the specified path."""
        model_path = os.path.join(path, "model.h5")
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")

        scaler_path = os.path.join(path, "scaler.pkl")
        import joblib
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")