# ------------------------------
# Dummy Sample Submission
# ------------------------------

BDT = False
NN = True

from statistical_analysis import calculate_saved_info, compute_mu
import numpy as np


class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init :
        takes 3 arguments: train_set systematics and model_type.
        can be used for initializing variables, classifier etc.
    2) fit :
        takes no arguments
        can be used to train a classifier
    3) predict:
        takes 1 argument: test sets
        can be used to get predictions of the test set.
        returns a dictionary

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods

            When you add another file with the submission model e.g. a trained model to be loaded and used,
            load it in the following way:

            # get to the model directory (your submission directory)
            model_dir = os.path.dirname(os.path.abspath(__file__))

            your trained model file is now in model_dir, you can load it from here
    """

    def __init__(self, get_train_set=None, systematics=None, model_type="sample_model"):
        """
        Model class constructor

        Params:
            train_set:
                a dictionary with data, labels, weights and settings

            systematics:
                a class which you can use to get a dataset with systematics added
                See sample submission for usage of systematics


        Returns:
            None
        """

        indices = np.arange(150000)

        np.random.shuffle(indices)

        train_indices = indices[:50000]
        holdout_indices = indices[50000:100000]
        valid_indices = indices[100000:]

        training_df = get_train_set(selected_indices=train_indices)

        self.training_set = {
            "labels": training_df.pop("labels"),
            "weights": training_df.pop("weights"),
            "detailed_labels": training_df.pop("detailed_labels"),
            "data": training_df,
        }

        del training_df

        self.systematics = systematics

        print("Training Data: ", self.training_set["data"].shape)
        print("Training Labels: ", self.training_set["labels"].shape)
        print("Training Weights: ", self.training_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 0].sum(),
        )

        valid_df = get_train_set(selected_indices=valid_indices)

        self.valid_set = {
            "labels": valid_df.pop("labels"),
            "weights": valid_df.pop("weights"),
            "detailed_labels": valid_df.pop("detailed_labels"),
            "data": valid_df,
        }
        del valid_df

        print()
        print("Valid Data: ", self.valid_set["data"].shape)
        print("Valid Labels: ", self.valid_set["labels"].shape)
        print("Valid Weights: ", self.valid_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 0].sum(),
        )

        holdout_df = get_train_set(selected_indices=holdout_indices)

        self.holdout_set = {
            "labels": holdout_df.pop("labels"),
            "weights": holdout_df.pop("weights"),
            "detailed_labels": holdout_df.pop("detailed_labels"),
            "data": holdout_df,
        }

        del holdout_df

        print()
        print("Holdout Data: ", self.holdout_set["data"].shape)
        print("Holdout Labels: ", self.holdout_set["labels"].shape)
        print("Holdout Weights: ", self.holdout_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 0].sum(),
        )
        print(" \n ")

        print("Training Data: ", self.training_set["data"].shape)
        print(f"DEBUG: model_type = {repr(model_type)}")

        if model_type == "BDT":
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree(train_data=self.training_set["data"])
        elif model_type == "NN":
            from neural_network import NeuralNetwork

            self.model = NeuralNetwork(train_data=self.training_set["data"])
        elif model_type == "sample_model":
            from sample_model import SampleModel

            self.model = SampleModel()
        else:
            print(f"model_type {model_type} not found")
            raise ValueError(f"model_type {model_type} not found")
        self.name = model_type

        print(f" Model is { self.name}")

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model

        Returns:
            None
        """

        balanced_set = self.training_set.copy()

        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
        class_weights_train = (
            weights_train[train_labels == 0].sum(),
            weights_train[train_labels == 1].sum(),
        )

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
            )
            # test dataset : increase test weight to compensate for sampling

        balanced_set["weights"] = weights_train

        self.model.fit(
            balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
        )

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
        Params:
            test_set

        Functionality:
            this function can be used for predictions using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
