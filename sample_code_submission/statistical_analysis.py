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


def compute_mu(score, weight, saved_info, method = "Likelihood"):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """

    score = score.flatten() > 0.5
    score = score.astype(int)

    mu, del_mu_stat, del_mu_tot, del_mu_sys = (0,0,0,0)

    if method == "Counting" :
        mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
        del_mu_stat = (
            np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
        )
        del_mu_sys = abs(0.0 * mu)
        del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    elif method == "Likelihood" :
        mu, del_mu_tot = likelihood_fit_mu(saved_info["beta"] + saved_info["gamma"], saved_info["gamma"], saved_info["beta"], 1) 
        plot_likelihood(saved_info["beta"] + saved_info["gamma"], saved_info["gamma"], saved_info["beta"], mu)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def calculate_saved_info(model, holdout_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(holdout_set["data"])

    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
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
    }

    print("saved_info", saved_info)

    return saved_info



#Calculation of log likelihood
from iminuit import Minuit
import matplotlib.pyplot as plt


def neg_ll(mu,n_obs,S,B):

    n_pred = mu*S + B
    n_pred = np.clip(n_pred, 1e-10, None) #éviter d'avoir log(0)
    neg_ll = - np.sum(n_obs*np.log(n_pred) - n_pred)
    
    return neg_ll


def likelihood_fit_mu(n_obs, S, B, mu_init):
    def neg_ll(mu):
        lam = mu * S + B
        lam = np.clip(lam, 1e-10, None)
        return -(n_obs * np.log(lam) - lam)

    m = Minuit(neg_ll, mu=mu_init)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD  # critical: tells Minuit it's a log-likelihood

    m.migrad()  # find minimum
    m.hesse()   # compute second derivatives (errors)

    return m.values["mu"], m.errors["mu"]


def plot_likelihood(n_obs, S, B, mu_hat):
    def neg_ll(mu):
        lam = mu * S + B
        lam = np.clip(lam, 1e-10, None)
        return -(n_obs * np.log(lam) - lam)

    # Generate a range of μ values around the best-fit
    mu_vals = np.linspace(0, 3, 200)
    nll_vals = [neg_ll(mu) for mu in mu_vals]

    # Find the minimum value (to plot ΔNLL)
    nll_min = min(nll_vals)
    delta_nll = [val - nll_min for val in nll_vals]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(mu_vals, delta_nll, label=r"$\Delta$NLL", color='blue')
    plt.axhline(0.5, color='gray', linestyle='--', label=r"1$\sigma$ contour ($\Delta$NLL = 0.5)")
    plt.axvline(mu_hat, color='red', linestyle='--', label=fr"$\hat\mu = {mu_hat:.3f}$")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$ Negative Log-Likelihood")
    plt.title("Profile Likelihood Curve for $\mu$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()