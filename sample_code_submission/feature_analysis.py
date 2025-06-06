# %pip install shap
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import chisquare
from HiggsML.datasets import download_dataset
from HiggsML.systematics import systematics
import numpy as np
from matplotlib.font_manager import FontProperties


# Obtain all histograms
def feature_dist(data_set):
    from utils import histogram_dataset

    feature_columns = [
        col
        for col in data_set.columns
        if col.startswith("PRI_") or col.startswith("DER_")
    ]

    for i in range(0, len(feature_columns), 4):
        subset = feature_columns[i : i + 4]
        histogram_dataset(
            data_set, data_set["labels"], data_set["weights"], columns=subset
        )


def feature_correlations(data_set):
    sns.set_theme(rc={"figure.figsize": (14, 12)}, style="whitegrid")

    label_col = next(
        (c for c in data_set.columns if c.lower() in ["label", "labels"]), None
    )
    jet_col = next((c for c in data_set.columns if c in ["PRI_n_jets"]), None)

    if label_col is None or jet_col is None:
        raise ValueError("Missing 'labels' or jet count column ('PRI_n_jets').")

    data_set = data_set.copy()
    data_set["jet_category"] = data_set[jet_col].apply(
        lambda x: "0j" if x == 0 else ("1j" if x == 1 else "2j+")
    )

    captions = {1: "Signal", 0: "Background"}

    for jet_cat in ["0j", "1j", "2j+"]:
        for label_val in [1, 0]:
            df_filtered = data_set[
                (data_set["jet_category"] == jet_cat)
                & (data_set[label_col] == label_val)
            ]

            if df_filtered.empty:
                print(f"⏭️ Skipped {captions[label_val]} - {jet_cat} (no data)")
                continue

            print(f"\n=== {captions[label_val]} - {jet_cat} ===")

            df_numeric = df_filtered.select_dtypes(include=["number"]).copy()

            drop_cols = (
                [label_col, "weights"]
                if "weights" in df_numeric.columns
                else [label_col]
            )
            df_numeric.drop(columns=drop_cols, inplace=True, errors="ignore")

            # Drop constant or mostly-missing columns
            df_numeric = df_numeric.loc[:, df_numeric.nunique(dropna=True) > 1]
            df_numeric = df_numeric.dropna(
                axis=1, thresh=int(0.9 * len(df_numeric))
            )  # keep cols with >90% non-NaN

            if df_numeric.shape[1] < 2:
                print(
                    f"⚠️ Too few valid numeric columns in {jet_cat} - {captions[label_val]}"
                )
                continue

            corr = df_numeric.corr()

            # Plot heatmap with correlation values
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )

            plt.title(
                f"{captions[label_val]} - {jet_cat} - Full Feature Correlation",
                fontsize=14,
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()


features_all = [
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


# show = False for not having all the graphs
# modify feature list for getting the impact of bias graph for only certain features
# returns a table with the chi2 errors measuring the impact for each feature
def impact_syst_bias_all(data, show=False, features=features_all):
    # === Load dataset ===
    data.load_train_set()
    original_df = data.get_train_set()
    original_weights = original_df["weights"]
    original_labels = original_df["labels"]
    # === Systematic bias config ===
    biases = {
        "tes": {"mean": 1.0, "sigma": 0.01},
        "jes": {"mean": 1.0, "sigma": 0.01},
        "soft_met": {"mean": 0.0, "sigma": 1.0},  # log-normal
        "ttbar_scale": {"mean": 1.0, "sigma": 0.02},
        "diboson_scale": {"mean": 1.0, "sigma": 0.25},
        "bkg_scale": {"mean": 1.0, "sigma": 0.001},
    }

    # === Chi-squared test helper ===
    def chi2_stat(observed, expected):
        mask = expected > 0
        return np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])

    chi2_results = {}
    for feat in features:
        chi2_results[f"{feat} (signal)"] = {}
        chi2_results[f"{feat} (background)"] = {}
    for bias_name, bias_info in biases.items():
        for shift in ["+1σ", "-1σ"]:
            if bias_name == "soft_met" and shift == "-1σ":
                continue

            kwargs = {
                "tes": 1.0,
                "jes": 1.0,
                "soft_met": 0.0,
                "ttbar_scale": 1.0,
                "diboson_scale": 1.0,
                "bkg_scale": 1.00,
            }
            value = (
                bias_info["mean"] + bias_info["sigma"]
                if shift == "+1σ"
                else bias_info["mean"] - bias_info["sigma"]
            )
            kwargs[bias_name] = value

            biased_result = systematics(
                {"data": original_df.copy(), "weights": original_weights.copy()},
                **kwargs,
                dopostprocess=False,
            )
            biased_df = biased_result["data"]
            biased_weights = biased_result["weights"]
            biased_labels = biased_df["labels"]

            for feat in features:
                if feat not in original_df.columns or feat not in biased_df.columns:
                    continue

                # Define common bins based on all data
                all_vals = pd.concat([original_df[feat], biased_df[feat]]).dropna()
                x_min, x_max = np.percentile(all_vals, [0.5, 99.5])
                bins = np.linspace(x_min, x_max, 80)

                # Histogram counts (not density) for chi-squared test
                orig_sig, _ = np.histogram(
                    original_df[feat][original_labels == 1],
                    bins=bins,
                    weights=original_weights[original_labels == 1],
                )
                bias_sig, _ = np.histogram(
                    biased_df[feat][biased_labels == 1],
                    bins=bins,
                    weights=biased_weights[biased_labels == 1],
                )

                orig_bkg, _ = np.histogram(
                    original_df[feat][original_labels == 0],
                    bins=bins,
                    weights=original_weights[original_labels == 0],
                )
                bias_bkg, _ = np.histogram(
                    biased_df[feat][biased_labels == 0],
                    bins=bins,
                    weights=biased_weights[biased_labels == 0],
                )

                # Chi-squared statistics
                chi2_sig = chi2_stat(bias_sig, orig_sig)
                chi2_bkg = chi2_stat(bias_bkg, orig_bkg)

                chi2_results[f"{feat} (signal)"][f"{bias_name} ({shift})"] = chi2_sig
                chi2_results[f"{feat} (background)"][
                    f"{bias_name} ({shift})"
                ] = chi2_bkg

                # Plot
                if show:
                    plt.figure(figsize=(6, 4))

                    plt.hist(
                        original_df[feat][original_labels == 1],
                        bins=bins,
                        weights=original_weights[original_labels == 1],
                        label="Signal (original)",
                        color="red",
                        alpha=0.25,
                        density=True,
                    )
                    plt.hist(
                        original_df[feat][original_labels == 0],
                        bins=bins,
                        weights=original_weights[original_labels == 0],
                        label="Background (original)",
                        color="blue",
                        alpha=0.25,
                        density=True,
                    )

                    plt.hist(
                        biased_df[feat][biased_labels == 1],
                        bins=bins,
                        weights=biased_weights[biased_labels == 1],
                        label=f"Signal ({bias_name} {shift})",
                        histtype="step",
                        color="darkred",
                        linewidth=1.5,
                        density=True,
                    )
                    plt.hist(
                        biased_df[feat][biased_labels == 0],
                        bins=bins,
                        weights=biased_weights[biased_labels == 0],
                        label=f"Background ({bias_name} {shift})",
                        histtype="step",
                        color="darkblue",
                        linewidth=1.5,
                        density=True,
                    )

                    plt.xlim(x_min, x_max)
                    plt.title(f"{feat} — effect of {bias_name} {shift}")
                    plt.xlabel(feat)
                    plt.ylabel("Density")
                    font = FontProperties()
                    font.set_size("small")
                    plt.legend(fontsize=font.get_size(), prop=font)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
                    print(
                        f"[{feat}] {bias_name} {shift}\n χ² (signal): {chi2_sig:.2f}, χ² (background): {chi2_bkg:.2f}"
                    )
    chi2_df = pd.DataFrame.from_dict(chi2_results, orient="index")
    return chi2_df


# module_path = os.path.join(os.getcwd(), "sample_code_submission", "BDT")
# if module_path not in sys.path:
#   sys.path.append(module_path)

# import sample_code_submission.BDT.boosted_decision_tree as BoostedDecisionTree


def minimal_dependent_features(data):
    """
    Uses permutation importance on BoostedDecisionTree to get top 10 important features.

    Parameters:
        data (pd.DataFrame): Dataset with 'Label' column.

    Returns:
        List[str]: Top 10 most important feature names.
    """
    X = data.drop(columns=["labels", "Weight", "DetailedLabel"], errors="ignore")
    y = data["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = BoostedDecisionTree.get_best_model()

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
    )

    importance_df = pd.DataFrame(
        {"Feature": X_test.columns, "Importance": result.importances_mean}
    ).sort_values(by="Importance", ascending=False)

    return importance_df["Feature"].head(10).tolist()
