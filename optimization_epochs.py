import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

from HiggsML.ingestion import Ingestion
from HiggsML.datasets import download_dataset

data = download_dataset(
    "blackSwan_data"
)  # change to "blackSwan_data" for the actual data

# load train set
data.load_train_set()
data_set = data.get_train_set()


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]
        # Vecteur de dimension n_dim en entrée

        self.model.add(Dense(10, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        # Simule la fonction de perte
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=5, verbose=2
        )

    # Vérification :
    def significance(self, test_data, labels, weights=None):
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
            bins = np.linspace(0, 1.0, 101)
            s_hist, bin_edges = np.histogram(
                y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
            )
            b_hist, bin_edges = np.histogram(
                y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
            )
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]
            significance = __amsasimov(s_cumul, b_cumul)
            max_value = np.max(significance)
            return significance

        y_pred_xgb = self.predict(test_data)
        vamsasimov_xgb = __significance_vscore(
            y_true=labels, y_score=y_pred_xgb, sample_weight=weights
        )
        significance_xgb = np.max(vamsasimov_xgb)
        return significance_xgb


TEST_SETTINGS = {
    "systematics": {  # Systematics to use
        "tes": False,  # tau energy scale
        "jes": False,  # jet energy scale
        "soft_met": False,  # soft term in MET
        "ttbar_scale": False,  # W boson scale factor
        "diboson_scale": False,  # Diboson scale factor
        "bkg_scale": False,  # Background scale factor
    },
    "num_pseudo_experiments": 25,  # Number of pseudo-experiments to run per set
    "num_of_sets": 25,  # Number of sets of pseudo-experiments to run
}

RANDOM_SEED = 42

test_settings = TEST_SETTINGS.copy()

random_state = np.random.RandomState(RANDOM_SEED)
test_settings["ground_truth_mus"] = (
    random_state.uniform(0.1, 3, test_settings["num_of_sets"])
).tolist()

random_settings_file = os.path.join(output_dir, "test_settings.json")
with open(random_settings_file, "w") as f:
    json.dump(test_settings, f)


L_epochs = np.linspace(2, 15, 14)


def optimization(train_data):
    for k in range(L_epochs):
        ingestion = Ingestion(data)
        # initialize submission
        ingestion.init_submission(Model, "NN")
        # fit submission
        ingestion.fit_submission()
        # load test set
        data.load_test_set()
        # predict submission
        ingestion.predict_submission(test_settings)
        ingestion.process_results_dict()
        significance(
            data_set,
        )
