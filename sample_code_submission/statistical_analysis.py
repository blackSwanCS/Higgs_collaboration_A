import numpy as np
from HiggsML.systematics import systematics
from iminuit import Minuit
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# from HiggsML.systematics import tes_fit, jes_fit

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

#################################################
#               MAIN FUNCTIONS                  #
#################################################


def compute_mu(score, weight, saved_info, method="Binned_Likelihood"):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """

    score_flat = score.flatten() > saved_info["best_threshold"]
    score_flat = score_flat.astype(int)

    mu, del_mu_stat, del_mu_tot, del_mu_sys = (0, 0, 0, 0)

    # Compute mu with counting method
    if method == "Counting":
        mu, del_mu_stat = counting_mu(score_flat, weight, saved_info)

    # Compute mu with Likelihood method
    elif method == "Likelihood":
        mu, del_mu_stat = likelihood_fit_mu(
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

    # Compute mu with likelihood and tes and jes
    elif method == "Likelihood+Systematics":
        mu, del_mu_stat = likelihood_fit_mu_tes_jes(
            saved_info["beta"] + saved_info["gamma"],
            saved_info["tes_fit"],
            saved_info["jes_fit"],
            1.0,
            1.0,
            1.0,
        )

    # Compute mu with binned likelihood
    elif method == "Binned_Likelihood":
        mu, del_mu_stat = likelihood_fit_mu_binned(score, score_flat, weight)
        plot_likelihood(
            saved_info["beta"] + saved_info["gamma"],
            saved_info["gamma"],
            saved_info["beta"],
            mu,
            plot_show=False,
        )
        plot_binned_likelihood(score, score_flat, weight, mu)
        plot_binned_histrograms(score, score_flat, weight)

    # Calculate del_mu_sys and tot
    del_mu_sys = abs(0.0 * mu)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    # Return results
    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def calculate_saved_info(model, holdout_set, method="AMS"):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(holdout_set["data"])

    #    from systematic_analysis import tes_fitter
    #    from systematic_analysis import jes_fitter

    best_threshold = 0

    # Chose an arbitrary cutoff
    if method == "Arbitrary":

        best_threshold = 0.5

    # Chose the cutoff that minimises deltaMu
    elif method == "Mu":

        # We calculate del_mu for many thresholds between 0 and 1
        threshold = np.linspace(0.01, 0.99, 100)
        del_mu = [0] * 100
        mu_list = [0] * 100

        # Iter through thresholds
        for i, t in enumerate(threshold):

            score2 = score.flatten() > t
            score2 = score2.astype(int)

            label = holdout_set["labels"]

            gamma = np.sum(holdout_set["weights"] * score2 * label)

            beta = np.sum(holdout_set["weights"] * score2 * (1 - label))

            mu = (np.sum(score2 * holdout_set["weights"]) - beta) / gamma
            del_mu_stat = np.sqrt(beta + gamma) / gamma
            del_mu_tot = np.sqrt(del_mu_stat**2)

            mu_list[i] = mu
            del_mu[i] = del_mu_tot

        # Find the minimum of delta_mu
        best_idx = np.argmin(del_mu)
        best_threshold = threshold[best_idx]

        # Uncomment to plot delta_mu
        # plt.plot(threshold, del_mu, label='del_mu')
        # plt.plot(threshold, mu_list, label='mu')
        # plt.axvline(best_threshold, color='green', linestyle='--', label=f'Best threshold = {best_threshold:.3f}')
        # plt.xlabel("Threshold")
        # plt.ylabel("Del_MU")
        # plt.grid()
        # plt.title("Del_mu and Mu vs Threshold")
        # plt.legend()
        # plt.show()

    elif method == "AMS":
        threshold = np.linspace(0.01, 0.99, 100)
        ams = [0] * 100

        # Iter through thresholds
        for i, t in enumerate(threshold):

            score2 = score.flatten() > t
            score2 = score2.astype(int)

            label = holdout_set["labels"]

            gamma = np.sum(holdout_set["weights"] * score2 * label)

            beta = np.sum(holdout_set["weights"] * score2 * (1 - label))

            ams[i] = np.sqrt(2 * ((gamma + beta) * np.log(1 + gamma / beta) - gamma))

        # Find the minimum of AMS
        best_idx = np.argmax(ams)
        best_threshold = threshold[best_idx]

        # Uncomment to plot AMS
        plt.plot(threshold, ams, label="ams")
        plt.axvline(
            best_threshold,
            color="green",
            linestyle="--",
            label=f"Best threshold = {best_threshold:.3f}",
        )
        plt.xlabel("Threshold")
        plt.ylabel("AMS")
        plt.grid()
        plt.title("AMS vs Threshold")
        plt.legend()
        plt.show()

    # Calculate saved_info with this optimised cutoff
    score = score.flatten() > best_threshold
    score = score.astype(int)

    label = holdout_set["labels"]

    gamma = np.sum(holdout_set["weights"] * score * label)

    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
        "beta": beta,
        "gamma": gamma,
        #        "tes_fit": tes_fitter(model, holdout_set),
        #        "jes_fit": jes_fitter(model, holdout_set),
        "best_threshold": best_threshold,
    }

    print("saved_info", saved_info)

    return saved_info


#################################################
#             HELPER FUNCTIONS                  #
#################################################


def counting_mu(score, weight, saved_info):
    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    return mu, del_mu_stat


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


def likelihood_fit_mu_binned(
    score,
    label,
    weights,
    mu_init=1.0,
):

    bins = np.linspace(0, 1, 101)
    # Masks
    signal_mask = label == 1
    background_mask = label == 0
    # Binned histograms
    S_hist, _ = np.histogram(
        score[signal_mask], bins=bins, weights=weights[signal_mask]
    )

    B_hist, _ = np.histogram(
        score[background_mask], bins=bins, weights=weights[background_mask]
    )
    N_obs, _ = np.histogram(score, bins=bins, weights=weights)

    # Binned negative log-likelihood function
    def neg_ll(mu):
        pred = mu * S_hist + B_hist
        pred = np.clip(pred, 1e-10, None)  # avoid log(0)
        return -np.sum(N_obs * np.log(pred) - pred)

    # Fit using Minuit
    m = Minuit(neg_ll, mu=mu_init)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD

    m.migrad()
    m.hesse()
    return m.values["mu"], m.errors["mu"]


def likelihood_fit_mu_tes_jes(
    n_obs, tes_fit, jes_fit, mu_init=1.0, tes_init=1.0, jes_init=1.0
):
    """
    Likelihood fit profiling over mu, tes, and jes.
    tes_fit and jes_fit should be callables/functions that return beta and gamma for given tes, jes.
    """

    def neg_ll(mu, tes, jes):
        # Get beta and gamma from the fit functions
        beta = tes_fit(tes)
        gamma = jes_fit(jes)
        lam = mu * gamma + beta
        lam = np.clip(lam, 1e-10, None)
        return -(n_obs * np.log(lam) - lam)

    m = Minuit(neg_ll, mu=mu_init, tes=tes_init, jes=jes_init)
    m.limits["mu"] = (0, None)
    m.limits["tes"] = (0.5, 1.5)  # Adjust as appropriate
    m.limits["jes"] = (0.5, 1.5)  # Adjust as appropriate
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()

    return m.values["mu"], m.errors["mu"]


#################################################
#                    PLOTS                      #
#################################################
def plot_likelihood(n_obs, S, B, mu_hat, plot_show=True):
    def neg_ll(mu):
        lam = mu * S + B
        lam = np.clip(lam, 1e-10, None)
        return -(n_obs * np.log(lam) - lam)

    mu_vals = np.linspace(max(0, mu_hat - 2), mu_hat + 2, 400)
    nll_vals = np.array([neg_ll(mu) for mu in mu_vals])

    # Normalize to ΔNLL
    nll_min = np.min(nll_vals)
    delta_nll = nll_vals - nll_min

    left_mask = mu_vals < mu_hat
    right_mask = mu_vals > mu_hat

    try:
        # Interpolate to find where ΔNLL = 0.5
        left_interp = interp1d(
            delta_nll[left_mask],
            mu_vals[left_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )
        right_interp = interp1d(
            delta_nll[right_mask],
            mu_vals[right_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        mu_lower = float(left_interp(0.5))
        mu_upper = float(right_interp(0.5))
        delta_mu = mu_upper - mu_lower
    except Exception as e:
        mu_lower = mu_hat
        mu_upper = mu_hat
        delta_mu = 0.0
        print("Interpolation error:", e)

    # Plot
    plt.plot(mu_vals, delta_nll, label=r"Unbinned $\Delta$NLL", color="blue")
    plt.axvline(
        mu_hat, color="red", linestyle="--", label=rf"Unbinned $\hat\mu = {mu_hat:.3f}$"
    )
    plt.axvline(
        mu_lower,
        color="green",
        linestyle="--",
        label=rf"Unbinned $\mu_{{-1\sigma}} = {mu_lower:.3f}$",
    )
    plt.axvline(
        mu_upper,
        color="green",
        linestyle="--",
        label=rf"Unbinned $\mu_{{+1\sigma}} = {mu_upper:.3f}$",
    )
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$ Negative Log-Likelihood")
    plt.title(rf"Unbinned Profile Likelihood: $\delta\mu$ = {delta_mu:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_show:
        plt.show()


def plot_binned_likelihood(score, label, weights, mu_hat, plot_show=True):
    bins = np.linspace(0, 1, 101)

    signal_mask = label == 1
    background_mask = label == 0

    S_hist, _ = np.histogram(
        score[signal_mask], bins=bins, weights=weights[signal_mask]
    )
    B_hist, _ = np.histogram(
        score[background_mask], bins=bins, weights=weights[background_mask]
    )
    N_obs, _ = np.histogram(score, bins=bins, weights=weights)

    def neg_ll(mu):
        pred = mu * S_hist + B_hist
        pred = np.clip(pred, 1e-10, None)
        return -np.sum(N_obs * np.log(pred) - pred)

    mu_vals = np.linspace(max(0, mu_hat - 2), mu_hat + 2, 400)
    nll_vals = np.array([neg_ll(mu) for mu in mu_vals])
    delta_nll = nll_vals - np.min(nll_vals)

    left_mask = mu_vals < mu_hat
    right_mask = mu_vals > mu_hat

    try:
        left_interp = interp1d(
            delta_nll[left_mask],
            mu_vals[left_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )
        right_interp = interp1d(
            delta_nll[right_mask],
            mu_vals[right_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        mu_lower = float(left_interp(0.5))
        mu_upper = float(right_interp(0.5))
        delta_mu = mu_upper - mu_lower
    except Exception as e:
        mu_lower = mu_hat
        mu_upper = mu_hat
        delta_mu = 0.0
        print("Interpolation error:", e)

    # Plot with slightly shifted colors
    plt.plot(
        mu_vals, delta_nll, label=r"Binned $\Delta$NLL", color="#4A90E2"
    )  # Lighter blue
    plt.axvline(
        mu_hat,
        color="#D0021B",
        linestyle="--",
        label=rf"Binned $\hat\mu = {mu_hat:.3f}$",
    )  # Soft red
    plt.axvline(
        mu_lower,
        color="#50E3C2",
        linestyle="--",
        label=rf"Binned $\mu_{{-1\sigma}} = {mu_lower:.3f}$",
    )  # Light teal
    plt.axvline(
        mu_upper,
        color="#50E3C2",
        linestyle="--",
        label=rf"Binned $\mu_{{+1\sigma}} = {mu_upper:.3f}$",
    )

    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$ Negative Log-Likelihood")
    plt.title(rf"Profile Binned Likelihood: $\delta\mu$ = {delta_mu:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_show:
        plt.show()


def plot_binned_histrograms(score, label, weights, plot_show=True, mu_init=1.0):
    bins = np.linspace(0, 1, 101)
    # Masks
    signal_mask = label == 1
    background_mask = label == 0
    # Binned histograms
    S_hist, _ = np.histogram(
        score[signal_mask], bins=bins, weights=weights[signal_mask]
    )

    B_hist, _ = np.histogram(
        score[background_mask], bins=bins, weights=weights[background_mask]
    )
    N_obs, _ = np.histogram(score, bins=bins, weights=weights)

    # Binned negative log-likelihood function
    def neg_ll(mu):
        pred = mu * S_hist + B_hist
        pred = np.clip(pred, 1e-10, None)  # avoid log(0)
        return -np.sum(N_obs * np.log(pred) - pred)

    # Fit using Minuit
    m = Minuit(neg_ll, mu=mu_init)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD

    m.migrad()
    m.hesse()

    plt.figure(figsize=(8, 5))
    width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.bar(
        bin_centers, N_obs, width=width, alpha=0.5, label="Observed", edgecolor="black"
    )
    plt.step(bin_centers, B_hist, where="mid", label="Background", color="orange")
    plt.step(
        bin_centers,
        m.values["mu"] * S_hist + B_hist,
        where="mid",
        label=f"Signal + Background (mu={m.values['mu']:.2f})",
        color="green",
    )

    plt.xlabel("Score")
    plt.ylabel("Weighted Events")
    plt.legend()
    plt.grid(True)
    plt.title("Binned Histogram: Observed vs Model Prediction")
    plt.tight_layout()
    if plot_show:
        plt.show()
