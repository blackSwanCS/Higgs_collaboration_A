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
    Retourne un dictionnaire nettoyé, où les objets non JSON-sérialisables
    (comme les fonctions) sont convertis en chaîne de caractères.
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
    Dummy Sample Submission – implémente un modèle de classification.

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
        # Définir systematics par défaut si non fourni.
        self.systematics = systematics if systematics is not None else lambda x: x

        # Initialiser les datasets
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

        # Chemin de sauvegarde/chargement du modèle
        self.model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.model_dir, "saved_model")
        self.model_type = model_type

        # Initialiser saved_info (sera complété après entraînement)
        self.saved_info = {}

        # Charger ou initialiser le modèle
        model_file = os.path.join(self.model_path, "model.h5")
        if os.path.exists(self.model_path) and os.path.exists(model_file):
            print("Loading pre-trained model...")
            self.load_model()
            self.is_trained = True
        else:
            print("No pre-trained model found. Initializing a new model.")
            self.initialize_model()
            self.is_trained = False

        # Set the name attribute
        self.name = model_type  # Use the model type as the name

    def fit(self):
        """
        Entraîner le modèle et sauvegarder saved_info.

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
            weights_train[train_labels == i] *= (max(class_weights_train) / class_weights_train[i])
        balanced_set["weights"] = weights_train

        # Entraînement du modèle via l'instance de NeuralNetwork
        self.model.fit(
            balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
        )

        # Calculer saved_info en utilisant l'instance du modèle et holdout_set
        self.saved_info = calculate_saved_info(self.model, self.holdout_set)

        # Sauvegarder le modèle et saved_info sur disque
        self.save_model()

        # Optionnel : appliquer systematics et recalculer saved_info
        self.holdout_set = self.systematics(self.holdout_set)
        self.training_set = self.systematics(self.training_set)
        self.saved_info = calculate_saved_info(self.model, self.holdout_set)

        # Compute results (affichage pour debug)
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
        Prédit sur test_set et retourne un dictionnaire de résultats.

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

        # Vérifier que saved_info contient les clés attendues
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
        Sauvegarder le modèle et saved_info sur disque.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # Sauvegarder le modèle via la méthode de NeuralNetwork
        self.model.save_model(self.model_path)
        # Sauvegarder saved_info dans un fichier JSON
        info_path = os.path.join(self.model_path, "saved_info.json")
        cleaned_info = _clean_saved_info(self.saved_info)
        with open(info_path, "w") as f:
            json.dump(cleaned_info, f)
        print(f"Model and saved_info saved to {self.model_path}")

    def load_model(self):
        """
        Charger un modèle pré-entraîné et saved_info depuis le disque.
        """
        from neural_network import NeuralNetwork

        self.model = NeuralNetwork()  # Initialiser sans données d'entraînement
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
        Initialiser un nouveau modèle selon le type.
        """
        if self.model_type == "NN":
            from neural_network import NeuralNetwork
            # Passer les données d'entraînement pour initialiser le neural network
            self.model = NeuralNetwork(self.training_set["data"])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
