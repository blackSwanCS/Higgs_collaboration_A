import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from HiggsML.datasets import download_dataset
from HiggsML.systematics import systematics
import BDT.boosted_decision_tree


def get_data(dataset):
    """
    Transfor a dataset into a dataset than can be used by the function model.predict()
    Also extract the weights and labels
    """
    training_set = dataset
    feature_keys = [
        k
        for k in training_set.keys()
        if k not in ("labels", "weights", "detailed_labels")
    ]
    X = np.column_stack([training_set[k] for k in feature_keys])
    y = training_set["labels"]
    w = training_set["weights"]

    n = len(y)
    split = int(n * 0.8)
    train_data, val_data = X[:split], X[split:]
    train_labels, val_labels = y[:split], y[split:]
    train_weights, val_weights = w[:split], w[split:]

    return train_data, train_labels, train_weights, val_data, val_labels, val_weights


def signal(score, label, weight):
    signal = []
    s_weight = []
    for i in range(len(score)):
        if label[i] == 1:
            signal.append(score[i])
            s_weight.append(weight[i])
    return signal, s_weight


def bck(score, label, weight):
    bck = []
    b_weight = []
    for i in range(len(score)):
        if label[i] == 0:
            bck.append(score[i])
            b_weight.append(weight[i])
    return bck, b_weight


def tes_fitter(model, train_set, bins_number):
    """
    Task 1 : Analysis TES Uncertainty
    1. Loop over different values of tes and make store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of tes and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given TES

    """
    data_set = train_set

    syst_set = systematics(data_set, tes=1)
    alt_data = get_data(syst_set)[0]

    score = model.predict(alt_data)

    labels_data = syst_set["labels"]
    labels_array = labels_data.to_numpy()
    weights_data = get_data(syst_set)[2]
    weights_array = weights_data.to_numpy()

    s_score, s_weight = signal(score, labels_array, weights_array)
    b_score, b_weight = bck(score, labels_array, weights_array)

    s_histogram, s_bin_edges = np.histogram(
        s_score, bins=bins_number, range=(0, 1), weights=s_weight, density=True
    )
    b_histogram, b_bin_edges = np.histogram(
        b_score, bins=bins_number, range=(0, 1), weights=b_weight, density=True
    )

    s_ref_data = s_histogram
    b_ref_data = b_histogram
    tes = [0.9 + 0.02 * i for i in range(11)]
    deltaS = []
    deltaB = []

    for elem in tes:
        syst_set = systematics(data_set, tes=elem)
        alt_data = get_data(syst_set)[0]
        labels_data = syst_set["labels"]
        labels_array = labels_data.to_numpy()
        weights_data = get_data(syst_set)[2]
        weights_array = weights_data.to_numpy()

        score = model.predict(alt_data)

        s_score, s_weight = signal(score, labels_array, weights_array)
        b_score, b_weight = bck(score, labels_array, weights_array)

        s_histogram, s_bin_edges = np.histogram(
            s_score, bins=bins_number, range=(0, 1), weights=s_weight, density=True
        )
        b_histogram, b_bin_edges = np.histogram(
            b_score, bins=bins_number, range=(0, 1), weights=b_weight, density=True
        )

        deltaS.append([])
        deltaB.append([])
        for i in range(bins_number):
            deltaS[-1].append(s_histogram[i] - s_ref_data[i])
            deltaB[-1].append(b_histogram[i] - b_ref_data[i])

    L = [deltaS, deltaB]

    def coefs(L, tes, bins):
        coef = [[0 for i in range(bins)], [0 for i in range(bins)]]
        for i in range(bins):
            coefficients = np.polyfit(tes, np.transpose(L[0])[i], 6)
            coef[0][i] = coefficients
        for i in range(bins):
            coefficients = np.polyfit(tes, np.transpose(L[1])[i], 6)
            coef[1][i] = coefficients
        return coef

    coeff = coefs(L, tes, bins_number)

    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

    def fit_function(array, tes):
        hist1, hist2 = array[0], array[1]
        newhist1, newhist2 = [], []
        for i in range(len(hist1)):
            newhist1.append(
                hist1[i] + sum([coeff[0][i][j] * tes ** (6 - i) for j in range(7)])
            )
        for i in range(len(hist2)):
            newhist2.append(
                hist2[i] + sum([coeff[1][i][j] * tes ** (6 - i) for j in range(7)])
            )
        return [newhist1, newhist2]

    return fit_function


def jes_fitter(model, train_set, bins_number):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given jes

    """
    data_set = train_set

    syst_set = systematics(data_set, jes=1)
    alt_data = get_data(syst_set)[0]

    score = model.predict(alt_data)

    labels_data = syst_set["labels"]
    labels_array = labels_data.to_numpy()
    weights_data = get_data(syst_set)[2]
    weights_array = weights_data.to_numpy()

    s_score, s_weight = signal(score, labels_array, weights_array)
    b_score, b_weight = bck(score, labels_array, weights_array)

    s_histogram, s_bin_edges = np.histogram(
        s_score, bins=bins_number, range=(0, 1), weights=s_weight, density=True
    )
    b_histogram, b_bin_edges = np.histogram(
        b_score, bins=bins_number, range=(0, 1), weights=b_weight, density=True
    )

    s_ref_data = s_histogram
    b_ref_data = b_histogram
    jes = [0.9 + 0.02 * i for i in range(11)]
    deltaS = []
    deltaB = []

    for elem in jes:
        syst_set = systematics(data_set, jes=elem)
        alt_data = get_data(syst_set)[0]
        labels_data = syst_set["labels"]
        labels_array = labels_data.to_numpy()
        weights_data = get_data(syst_set)[2]
        weights_array = weights_data.to_numpy()

        score = model.predict(alt_data)

        s_score, s_weight = signal(score, labels_array, weights_array)
        b_score, b_weight = bck(score, labels_array, weights_array)

        s_histogram, s_bin_edges = np.histogram(
            s_score, bins=bins_number, range=(0, 1), weights=s_weight, density=True
        )
        b_histogram, b_bin_edges = np.histogram(
            b_score, bins=bins_number, range=(0, 1), weights=b_weight, density=True
        )

        deltaS.append([])
        deltaB.append([])
        for i in range(bins_number):
            deltaS[-1].append(s_histogram[i] - s_ref_data[i])
            deltaB[-1].append(b_histogram[i] - b_ref_data[i])

    L = [deltaS, deltaB]

    def coefs(L, jes, bins):
        coef = [[0 for i in range(bins)], [0 for i in range(bins)]]
        for i in range(bins):
            coefficients = np.polyfit(jes, np.transpose(L[0])[i], 6)
            coef[0][i] = coefficients
        for i in range(bins):
            coefficients = np.polyfit(jes, np.transpose(L[1])[i], 6)
            coef[1][i] = coefficients
        return coef

    coeff = coefs(L, jes, bins_number)

    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

    def fit_function(array, jes):
        hist1, hist2 = array[0], array[1]
        newhist1, newhist2 = [], []
        for i in range(len(hist1)):
            newhist1.append(
                hist1[i] + sum([coeff[0][i][j] * jes ** (6 - i) for j in range(7)])
            )
        for i in range(len(hist2)):
            newhist2.append(
                hist2[i] + sum([coeff[1][i][j] * jes ** (6 - i) for j in range(7)])
            )
        return [newhist1, newhist2]

    return fit_function
