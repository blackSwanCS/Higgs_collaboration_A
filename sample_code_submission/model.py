# ------------------------------
# Dummy Sample Submission
# ------------------------------

BDT = False
NN = True

from statistical_analysis import calculate_saved_info, compute_mu
import os
import numpy as np


class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consist of the following functions:
    1) init :
        takes 3 arguments: train_set, systematics, and model_type.
        can be used for initializing variables, classifier, etc.
    2) fit :
        takes no arguments
        can be used to train a classifier
    3) predict:
        takes 1 argument: test sets
        can be used to get predictions of the test set.
        returns a dictionary

    Note: Add more methods if needed e.g. save model, load pre-trained model, etc.
    """

    def __init__(self, get_train_set=None, systematics=None, model_type="NN"):
        """
        Model class constructor

        Params:
            get_train_set: function to retrieve the training set
            systematics: a function (or class) to apply systematics to the dataset.
                         If None, an identity function is used.
            model_type: type of model to use (e.g., "NN", "BDT", etc.)
        Returns:
            None
        """
        # Ensure systematics is defined even if not provided.
        if systematics is None:
            # If no systematics provided, use identity (no change).
            self.systematics = lambda x: x
        else:
            self.systematics = systematics

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

        # Define directory to save/load the model
        self.model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.model_dir, "saved_model")
        self.model_type = model_type

        # Vérifiez si un modèle sauvegardé existe
        model_file = os.path.join(self.model_path, "model.h5")
        if os.path.exists(self.model_path) and os.path.exists(model_file):
            print("Loading pre-trained model...")
            self.load_model()
            self.is_trained = True  # Indique que le modèle est déjà entraîné
        else:
            print("No pre-trained model found. Initializing a new model.")
            self.initialize_model()
            self.is_trained = False  # Indique que le modèle doit être entraîné

        # Set the name attribute
        self.name = model_type  # Use the model type as the name

    def fit(self):
        """
        Train the model and save it after training.

        Params:
            None

        Returns:
            None
        """
        if self.is_trained:
            print("Model is already trained. Skipping training.")
            return

        # Vérifiez que le modèle est initialisé
        if self.model is None:
            raise ValueError("Model is not initialized. Ensure `initialize_model` or `load_model` is called.")

        balanced_set = self.training_set.copy()

        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
        class_weights_train = (
            weights_train[train_labels == 0].sum(),
            weights_train[train_labels == 1].sum(),
        )

        for i in range(len(class_weights_train)):  # loop on B then S target
            weights_train[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
            )

        balanced_set["weights"] = weights_train

        self.model.fit(
            balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
        )

        # Save the trained model
        self.save_model()

        self.holdout_set = self.systematics(self.holdout_set)

        self.saved_info = calculate_saved_info(self.model, self.holdout_set)

        self.training_set = self.systematics(self.training_set)

        # Compute  Results
        train_score = self.model.predict(self.training_set["data"])
        train_results = compute_mu(
            train_score, self.training_set["weights"], self.saved_info
        )

        holdout_score = self.model.predict(self.holdout_set["data"])
        holdout_results = compute_mu(
            holdout_score, self.holdout_set["weights"], self.saved_info
        )

        self.valid_set = self.systematics(self.valid_set)

        valid_score = self.model.predict(self.valid_set["data"])

        valid_results = compute_mu(
            valid_score, self.valid_set["weights"], self.saved_info
        )

        print("Train Results: ")
        for key in train_results.keys():
            print("\t", key, " : ", train_results[key])

        print("Holdout Results: ")
        for key in holdout_results.keys():
            print("\t", key, " : ", holdout_results[key])

        print("Valid Results: ")
        for key in valid_results.keys():
            print("\t", key, " : ", valid_results[key])

        self.valid_set["data"]["score"] = valid_score
        from utils import roc_curve_wrapper, histogram_dataset

        histogram_dataset(
            self.valid_set["data"],
            self.valid_set["labels"],
            self.valid_set["weights"],
            columns=["score"],
        )

        from HiggsML.visualization import stacked_histogram

        stacked_histogram(
            self.valid_set["data"],
            self.valid_set["labels"],
            self.valid_set["weights"],
            self.valid_set["detailed_labels"],
            "score",
        )

        roc_curve_wrapper(
            score=valid_score,
            labels=self.valid_set["labels"],
            weights=self.valid_set["weights"],
            plot_label="valid_set" + self.name,
        )

    def predict(self, test_set):
        """
        Make predictions on the test set.

        Params:
            test_set: dictionary containing test data and weights

        Returns:
            dict with keys:
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """
        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        from statistical_analysis import compute_mu

        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result

    def save_model(self):
        """
        Save the trained model to the model directory.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """
        Load a pre-trained model from the model directory.
        """
        from neural_network import NeuralNetwork

        self.model = NeuralNetwork()  # Initialisez sans données d'entraînement
        self.model.load_model(self.model_path)

        # Vérifiez que le modèle est bien chargé
        if self.model.model is None:
            raise ValueError("Failed to load the model. Ensure 'model.h5' exists and is valid.")
        print(f"Model loaded from {self.model_path}")

    def initialize_model(self):
        """
        Initialize a new model based on the model type.
        """
        if self.model_type == "NN":
            from neural_network import NeuralNetwork
            # Pass training data so that _initialize_model is called
            self.model = NeuralNetwork(self.training_set["data"])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
