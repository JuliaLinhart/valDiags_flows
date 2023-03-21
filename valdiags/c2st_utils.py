# Utils for Classifier Two Sample Test (C2ST):
# - compute metrics on classifier-predicted class probabilities

import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from .pp_plots import PP_vals


def compute_metric(proba, metrics):
    """Computes metrics on classifier-predicted class probabilities.

    Args:
        proba (numpy.array): predicted probability for class 0.
        metrics (list of str): list of names of metrics to compute.

    Returns:
        (dict): dictionary of computed metrics.
    """
    scores = {}
    for m in metrics:
        # mean of probas
        if m == "probas_mean":
            scores[m] = np.mean(proba)

        # std of probas
        elif m == "probas_std":
            scores[m] = np.std(proba)

        # wasserstein distance between dirac and probas
        elif m == "w_dist":
            scores[m] = wasserstein_distance([0.5] * len(proba), proba)

        # total variation distance between dirac and probas
        elif m == "TV":
            alphas = np.linspace(0, 1, 100)
            pp_vals_dirac = pd.Series(
                PP_vals([0.5] * len(proba), alphas)
            )  # cdf of dirac
            pp_vals = PP_vals(proba, alphas)  # cdf of probas
            scores[m] = ((pp_vals - pp_vals_dirac) ** 2).sum() / len(
                alphas
            )  # TV: mean squared error between cdfs

        # 'custom divergence': mean of max probas
        elif m == "div":
            mask = proba > 1 / 2
            max_proba = np.concatenate([proba[mask], 1 - proba[~mask]])
            scores[m] = np.mean(max_proba)

        # mean squared error between probas and dirac (cf. [Lee et al. (2018)]
        elif m == "mse":
            scores[m] = ((proba - [0.5] * len(proba)) ** 2).mean()

        # not implemented
        else:
            scores[m] = None
            print(f'metric "{m}" not implemented')

    return scores
