# %pip install shap
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def feature_corrilations(data_set):
    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")

    captions = ["Signal feature", "Background feature"]

    # Assume data_set is a DataFrame and target is a Series or array
    target = data_set["labels"]
    for i in range(2):
        # Filter rows by class
        df_filtered = data_set[target == i]

        # Keep only numeric columns
        df_numeric = df_filtered.select_dtypes(include=["number"])

        # Select DER and PRI columns
        der_cols = [col for col in df_numeric.columns if col.startswith("DER")]
        pri_cols = [col for col in df_numeric.columns if col.startswith("PRI")]

        # Subset the data
        der_data = df_numeric[der_cols]
        pri_data = df_numeric[pri_cols]

        partial_corr_matrix = pd.DataFrame(index=pri_cols, columns=der_cols)

        for pri in pri_cols:
            for der in der_cols:
                partial_corr_matrix.loc[pri, der] = df_numeric[pri].corr(df_numeric[der])

        partial_corr_matrix = partial_corr_matrix.astype(float)
        
        # DER vs DER correlation matrix
        corr_der = der_data.corr()
        print(f"{captions[i]} DER vs DER correlation matrix")
        sns.heatmap(corr_der, annot=True, cmap="coolwarm", center=0)
        plt.title(f"DER Feature Correlation ({captions[i].lower()})")
        plt.tight_layout()
        plt.show()

        # PRI vs PRI correlation matrix
        corr_pri = pri_data.corr()
        print(f"{captions[i]} PRI vs PRI correlation matrix")
        sns.heatmap(corr_pri, annot=True, cmap="coolwarm", center=0)
        plt.title(f"PRI Feature Correlation ({captions[i].lower()})")
        plt.tight_layout()
        plt.show()
        pri_jet_cols = [col for col in pri_cols if "jet" in col]
        pri_nonjet_cols = [col for col in pri_cols if not "jet" in col]

        der_jetjet_cols = [col for col in der_cols if "jet_jet" in col]
        der_nonjetjet_cols = [col for col in der_cols if not "jet_jet" in col]

        # All PRI features vs DER features without "jet_jet"
        partial_corr_jet_vs_nonjetjet = pd.DataFrame(index=pri_jet_cols, columns=der_nonjetjet_cols)

        for pri in pri_cols:
            for der in der_nonjetjet_cols:
                partial_corr_jet_vs_nonjetjet.loc[pri, der] = df_numeric[pri].corr(df_numeric[der])

        partial_corr_jet_vs_nonjetjet = partial_corr_jet_vs_nonjetjet.astype(float)

        print(f"{captions[i]} PRI(jet) vs DER(non-jet_jet) correlation matrix")
        sns.heatmap(partial_corr_jet_vs_nonjetjet, annot=True, cmap="coolwarm", center=0)
        plt.title(f"ALL PRI vs DER(non-jet_jet) ({captions[i].lower()})")
        plt.tight_layout()
        plt.show()

        # PRI features NOT containing "jet" vs ALL DER features
        partial_corr_nonjet_vs_all_der = pd.DataFrame(index=pri_nonjet_cols, columns=der_cols)

        for der in der_cols:
            for pri in pri_nonjet_cols:
                partial_corr_nonjet_vs_all_der.loc[pri, der] = df_numeric[pri].corr(df_numeric[der])

        partial_corr_nonjet_vs_all_der = partial_corr_nonjet_vs_all_der.astype(float)
        print(f"{captions[i]} PRI(non-jet) vs ALL DER correlation matrix")
        sns.heatmap(partial_corr_nonjet_vs_all_der, annot=True, cmap="coolwarm", center=0)
        plt.title(f"PRI(non-jet) vs ALL DER ({captions[i].lower()})")
        plt.tight_layout()
        plt.show()

def systematics_dependence(data):
    pass


import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

module_path = os.path.join(os.getcwd(), "sample_code_submission", "BDT")
if module_path not in sys.path:
    sys.path.append(module_path)

import sample_code_submission.BDT.boosted_decision_tree as BoostedDecisionTree


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
