# Utils for Classifier Two Sample Test (C2ST):
# - compute metrics on classifier-predicted class probabilities

import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from .pp_plots import PP_vals

from tqdm import tqdm


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


def t_stats_c2st(
    scores_fn, P, Q, null_samples_list, metrics=["accuracy"], verbose=True, **kwargs
):
    """Computes the C2ST test statistics estimated on P and Q, 
    as well as on several samples of data from P to simulate the null hypothesis (Q=P).

    Args:
        scores_fn (function): function to compute metrics on classifier-predicted class probabilities.
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        null_samples_list (list of numpy.array): list of samples from P (= Q under the null)
            of size (n_samples, dim).
        metrics (list of str, optional): list of names of metrics (aka test statistics) to compute.
            Defaults to ["accuracy"].
        verbose (bool, optional): if True, display progress bar. 
            Defaults to True.
        **kwargs: keyword arguments for scores_fn.
    
    Returns:
        (tuple): tuple containing:
            - t_stat_data (dict): dictionary of test statistics estimated on P and Q.
                keys are the names of the metrics. values are floats.
            - t_stats_null (dict): dictionary of test statistics estimated on P and `null_samples_list`.
                keys are the names of the metrics. values are lists of length `len(null_samples_list)`.
    """
    # initialize dicts
    t_stat_data = {}
    t_stats_null = dict(zip(metrics, [[] for _ in range(len(metrics))]))

    # compute test statistics on P and Q
    scores_data = scores_fn(P=P, Q=Q, metrics=metrics, **kwargs)
    # compute their mean (useful if cross_val=True)
    for m in metrics:
        t_stat_data[m] = np.mean(scores_data[m])

    # loop over trials under the null hypothesis
    for i in tqdm(
        range(len(null_samples_list)),
        desc="Testing under the null",
        disable=(not verbose),
    ):
        # compute test statistics on P and null_samples_list[i] (=P_i)
        scores_null = scores_fn(P=P, Q=null_samples_list[i], metrics=metrics, **kwargs,)
        # compute their mean (useful if cross_val=True)
        for m in metrics:
            t_stats_null[m].append(np.mean(scores_null[m]))

    return t_stat_data, t_stats_null
