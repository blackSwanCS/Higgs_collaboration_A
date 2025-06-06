from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree as Model
import numpy as np

data = download_dataset("blackSwan_data")

data.load_train_set()
training_set = data.get_train_set()

feature_keys = [
    k for k in training_set.keys() if k not in ("labels", "weights", "detailed_labels")
]
train_data = np.column_stack([training_set[k] for k in feature_keys])

data.load_test_set()
test_set = data.get_test_set()

feature_keys = [
    k for k in test_set.keys() if k not in ("labels", "weights", "detailed_labels")
]
test_data = np.column_stack([test_set[k] for k in feature_keys])

params = {
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 6,
    "objective": "binary:logistic",
}


def compare(params):
    model = Model(training_set, params=params)

    data = [
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_had_pt",
        "PRI_had_eta",
        "PRI_had_phi",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_n_jets",
        "PRI_jet_all_pt",
        "PRI_met",
        "PRI_met_phi",
        "detailed_labels",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_deltar_had_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_had",
        "DER_met_phi_centrality",
        "DER_lep_eta_centrality",
    ]

    model.fit(train_data, training_set["labels"], training_set["weights"])

    # Predict on the test data
    predictions = model.predict(test_data)

    # Calculate significance
    significance = model.__significance__(
        test_set["labels"], predictions, test_set["weights"]
    )

    print(f"Significance: {significance}")


compare(params)
