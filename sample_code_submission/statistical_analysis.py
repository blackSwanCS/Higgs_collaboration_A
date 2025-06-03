import numpy as np
from HiggsML.systematics import systematics

"""
Task 1a : Counting Estimator
1.write the saved_info dictionary such that it contains the following keys
    1. beta
    2. gamma
2. Estimate the mu using the formula
    mu = (sum(score * weight) - beta) / gamma
3. return the mu and its uncertainty

Task 1b : Stat-Only Likelihood Estimator
1. Modify the estimation of mu such that it uses the likelihood function
    1. Write a function for the likelihood function which profiles over mu
    2. Use Minuit to minimize the NLL

Task 2 : Systematic Uncertainty
1. substitute the beta and gamma with the tes_fit and jes_fit functions
2. Write a function to likelihood function which profiles over mu, tes and jes
3. Use Minuit to minimize the NLL
4. return the mu and its uncertainty

"""


def compute_mu(score, weight, saved_info, method="Likelihood"):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """

    score = score.flatten() > saved_info["best_threshold"]
    score = score.astype(int)

    mu, del_mu_stat, del_mu_tot, del_mu_sys = (0, 0, 0, 0)

    if method == "Counting":
        mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
        del_mu_stat = (
            np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
        )
        del_mu_sys = abs(0.0 * mu)
        del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    elif method == "Likelihood":
        mu, del_mu_tot = likelihood_fit_mu(
            saved_info["beta"] + saved_info["gamma"],
            saved_info["gamma"],
            saved_info["beta"],
            0.1,
        )
        plot_likelihood(
            saved_info["beta"] + saved_info["gamma"],
            saved_info["gamma"],
            saved_info["beta"],
            mu,
        )

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def calculate_saved_info(model, holdout_set, method = "Mu"):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(holdout_set["data"])

    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter

    best_threshold = 0

    # Chose an arbitrary cutoff
    if method == "Arbitrary" :

        best_threshold = 0.5
        
    # Chose the cutoff that minimises deltaMu
    elif method == "Mu" :

        # We calculate del_mu for many thresholds between 0 and 1
        threshold = np.linspace(0.01,0.99,100)
        del_mu = [0]*100
        mu_list = [0]*100

        #Iter through thresholds
        for i, t in enumerate(threshold) :

            score2 = score.flatten() > t
            score2 = score2.astype(int)

            label = holdout_set["labels"]

            gamma = np.sum(holdout_set["weights"] * score2 * label)

            beta = np.sum(holdout_set["weights"] * score2 * (1 - label))

            mu = (np.sum(score2 * holdout_set["weights"]) - beta) / gamma
            del_mu_stat = (
                np.sqrt(beta + gamma) / gamma
            )
            del_mu_tot = np.sqrt(del_mu_stat**2)

            mu_list[i] = mu
            del_mu[i] = del_mu_tot

        # Find the minimum of delta_mu
        best_idx = np.argmin(del_mu)
        best_threshold = threshold[best_idx]

    # Calculate saved_info with this optimised cutoff
    score = score.flatten() > best_threshold
    score = score.astype(int)

    label = holdout_set["labels"]

    print("score shape after threshold", score.shape)

    gamma = np.sum(holdout_set["weights"] * score * label)

    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
        "beta": beta,
        "gamma": gamma,
        "tes_fit": tes_fitter(model, holdout_set),
        "jes_fit": jes_fitter(model, holdout_set),
        "best_threshold": best_threshold
    }

    print("saved_info", saved_info)

    return saved_info


from iminuit import Minuit
import matplotlib.pyplot as plt


def neg_ll(mu, n_obs, S, B):

    n_pred = mu * S + B
    n_pred = np.clip(n_pred, 1e-10, None)  # to prevent log(0)
    neg_ll = -np.sum(n_obs * np.log(n_pred) - n_pred)

    return neg_ll


def likelihood_fit_mu(n_obs, S, B, mu_init):
    def neg_ll(mu):
        lam = mu * S + B
        lam = np.clip(lam, 1e-10, None)
        return -(n_obs * np.log(lam) - lam)

    m = Minuit(neg_ll, mu=mu_init)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD

    m.migrad()  # computes the minimum
    m.hesse()  # computes the hessian

    return m.values["mu"], m.errors["mu"]


def plot_likelihood(n_obs, S, B, mu_hat):
    def neg_ll(mu):
        lam = mu * S + B
        lam = np.clip(lam, 1e-10, None)
        return -(n_obs * np.log(lam) - lam)

    mu_vals = np.linspace(0, 3, 200)
    nll_vals = [neg_ll(mu) for mu in mu_vals]

    # finds the minimum (to plot ΔNLL)
    nll_min = min(nll_vals)
    delta_nll = [val - nll_min for val in nll_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(mu_vals, delta_nll, label=r"$\Delta$NLL", color="blue")
    plt.axhline(
        0.5,
        color="gray",
        linestyle="--",
        label=r"1$\sigma$ contour ($\Delta$NLL = 0.5)",
    )
    plt.axvline(mu_hat, color="red", linestyle="--", label=rf"$\hat\mu = {mu_hat:.3f}$")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$ Negative Log-Likelihood")
    plt.title("Profile Likelihood Curve for $\mu$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
