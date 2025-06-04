# ------------------------------
# Dummy Sample Submission
# ------------------------------

import os
import json
import numpy as np
from statistical_analysis import calculate_saved_info, compute_mu

BDT = False
NN = True


def _clean_saved_info(info):
    """
    Return a cleaned dictionary where objects not JSON serializable 
    (like functions) are converted to strings.
    """
    cleaned = {}
    for key, value in info.items():
        try:
            json.dumps(value)
            cleaned[key] = value
        except TypeError:
            cleaned[key] = str(value)
    return cleaned


class Model:
    """
    Dummy Sample Submission – implements a classification model.

    The class should have the following methods:
    1) __init__: receives a training set retriever, systematics function, and model type.
    2) fit: used to train a classifier.
    3) predict: receives a test set and returns a dictionary with:
         - mu_hat: predicted mu
         - delta_mu_hat: mu uncertainty bound
         - p16: 16th percentile
         - p84: 84th percentile
    """

    def __init__(self, get_train_set=None, systematics=None, model_type="NN"):
        """
        Constructor

        Params:
            get_train_set: function to retrieve the training set.
            systematics: function (or class) to apply systematics to the dataset.
                         Defaults to the identity function if None.
            model_type: type of model to use (e.g., "NN", "BDT", etc.)
        """
        self.systematics = systematics if systematics is not None else lambda x: x

        # Initialize datasets
        indices = np.arange(15000)
        np.random.shuffle(indices)
        train_indices = indices[:5000]
        holdout_indices = indices[5000:10000]
        valid_indices = indices[10000:]

        training_df = get_train_set(selected_indices=train_indices)
        self.training_set = {
            "labels": training_df.pop("labels"),
            "weights": training_df.pop("weights"),
            "detailed_labels": training_df.pop("detailed_labels"),
            "data": training_df,
        }
        del training_df

        valid_df = get_train_set(selected_indices=valid_indices)
        self.valid_set = {
            "labels": valid_df.pop("labels"),
            "weights": valid_df.pop("weights"),
            "detailed_labels": valid_df.pop("detailed_labels"),
            "data": valid_df,
        }
        del valid_df

        holdout_df = get_train_set(selected_indices=holdout_indices)
        self.holdout_set = {
            "labels": holdout_df.pop("labels"),
            "weights": holdout_df.pop("weights"),
            "detailed_labels": holdout_df.pop("detailed_labels"),
            "data": holdout_df,
        }
        del holdout_df

        # Path for saving/loading the model
        self.model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.model_dir, "saved_model")
        self.model_type = model_type

        # Initialize saved_info (will be updated after training)
        self.saved_info = {}

        # Load or initialize the model
        model_file = os.path.join(self.model_path, "model.h5")
        if os.path.exists(self.model_path) and os.path.exists(model_file):
            print("Loading pre-trained model...")
            self.load_model()
            self.is_trained = True
        else:
            print("No pre-trained model found. Initializing a new model.")
            self.initialize_model()
            self.is_trained = False

        self.name = model_type

    def fit(self):
        """
        Train the model and save saved_info.
        """
        if self.is_trained:
            print("Model is already trained. Skipping training.")
            return

        if self.model is None:
            raise ValueError("Model is not initialized. Ensure 'initialize_model' or 'load_model' is called.")

        balanced_set = self.training_set.copy()
        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
        class_weights_train = (
            weights_train[train_labels == 0].sum(),
            weights_train[train_labels == 1].sum(),
        )
        for i in range(len(class_weights_train)):
            weights_train[train_labels == i] *= (max(class_weights_train) / class_weights_train[i])
        balanced_set["weights"] = weights_train

        # Train the model via the NeuralNetwork instance
        self.model.fit(
            balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
        )

        # Calculate saved_info using the model and holdout_set
        self.saved_info = calculate_saved_info(self.model, self.holdout_set)

        # Save model and saved_info to disk
        self.save_model()

        # Optional: apply systematics and recalculate saved_info
        self.holdout_set = self.systematics(self.holdout_set)
        self.training_set = self.systematics(self.training_set)
        self.saved_info = calculate_saved_info(self.model, self.holdout_set)

        # Compute results (debug display)
        train_score = self.model.predict(self.training_set["data"])
        train_results = compute_mu(train_score, self.training_set["weights"], self.saved_info)
        holdout_score = self.model.predict(self.holdout_set["data"])
        holdout_results = compute_mu(holdout_score, self.holdout_set["weights"], self.saved_info)
        valid_score = self.model.predict(self.valid_set["data"])
        valid_results = compute_mu(valid_score, self.valid_set["weights"], self.saved_info)

        print("Train Results:")
        for key in train_results:
            print("\t", key, ":", train_results[key])
        print("Holdout Results:")
        for key in holdout_results:
            print("\t", key, ":", holdout_results[key])
        print("Valid Results:")
        for key in valid_results:
            print("\t", key, ":", valid_results[key])

    def predict(self, test_set):
        """
        Predict on test_set and return a result dictionary.

        Params:
            test_set: dict containing test data and weights.
        """
        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        if "beta" not in self.saved_info:
            self.saved_info["beta"] = 0.0
        if "gamma" not in self.saved_info:
            self.saved_info["gamma"] = 1.0

        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)
        print("Test Results:", result_mu_cal)
        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }
        return result

    def save_model(self):
        """
        Save the model and saved_info to disk.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.save_model(self.model_path)
        info_path = os.path.join(self.model_path, "saved_info.json")
        cleaned_info = _clean_saved_info(self.saved_info)
        with open(info_path, "w") as f:
            json.dump(cleaned_info, f)
        print(f"Model and saved_info saved to {self.model_path}")

    def load_model(self):
        """
        Load a pre-trained model and saved_info from disk.
        """
        from neural_network import NeuralNetwork

        self.model = NeuralNetwork()
        self.model.load_model(self.model_path)
        info_path = os.path.join(self.model_path, "saved_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                self.saved_info = json.load(f)
            print("saved_info loaded.")
        else:
            self.saved_info = {}
        if self.model.model is None:
            raise ValueError("Failed to load the model. Ensure 'model.h5' exists and is valid.")
        print(f"Model loaded from {self.model_path}")

    def initialize_model(self):
        """
        Initialize a new model based on the type.
        """
        if self.model_type == "NN":
            from neural_network import NeuralNetwork
            # Pass training data to initialize the neural network
            self.model = NeuralNetwork(self.training_set["data"])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
