# Implementation of vanilla C2ST (Classifier Two Sample Test)
# - [Lopez et al. (2017)](https://arxiv.org/abs/1610.06545)
# - [Lee et al. (2018)](https://arxiv.org/abs/1812.08927)

import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import sklearn

from scipy.stats import wasserstein_distance
from .graphical_valdiags import PP_vals

from tqdm import tqdm

# define default classifier
DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


# ==== train / eval functions for the classifier used in C2ST ====


def train_c2st(P, Q, clf=DEFAULT_CLF):
    """ Trains a classifier to distinguish between data from P and Q.

    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        clf (sklearn model, optional): the initialized classifier to use.
            needs to have a method `.fit(X,y)`. 
            Defaults to DEFAULT_CLF.

    Returns:
        (sklearn model): trained classifier (cloned from clf).
    """

    # define features and labels
    features = np.concatenate([P, Q], axis=0)  # (2*n_samples, dim)
    labels = np.concatenate(
        [np.array([0] * len(P)), np.array([1] * len(Q))]
    ).ravel()  # (2*n_samples,)

    # shuffle features and labels
    features, labels = shuffle(features, labels)

    # train the classifier
    clf = sklearn.base.clone(clf)
    clf.fit(X=features, y=labels)

    return clf


def eval_c2st(P, Q, clf, single_class_eval=False):
    """Evaluates a classifier on data from P and Q.
    
    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        clf (sklearn model): the trained classifier on both classes.
            needs to have a methods `.score(X,y)` and `.predict_proba(X)`.
        single_class_eval (bool, optional): if True, only evaluate on P.
            Defaults to False.
    
    Returns:
        (float, numpy.array): accuracy and proba of class 0.
    """

    n_samples = len(P)
    # define features and labels
    if single_class_eval:
        X_val = P  # only evaluate on P
        y_val = np.array([0] * (n_samples))  # labels are all 0
    else:
        X_val = np.concatenate([P, Q], axis=0)  # evaluate on both P and Q
        y_val = np.array(
            [0] * n_samples + [1] * n_samples
        )  # labels are 0 for P and 1 for Q

    # evaluate the classifier
    accuracy = clf.score(X_val, y_val)  # accuracy
    proba = clf.predict_proba(X_val)[:, 0]  # proba of class 0

    return accuracy, proba


# ==== C2ST test functions ====
# - define metrics
# - estimate the test statistics by computing the metrics on a data sample (in-sample or cross-validation)
# - infer test statistics on observed data and under the null (can then be used to perform the test)


def compute_metric(proba, metrics, single_class_eval=False):
    """Computes metrics on classifier-predicted class probabilities.

    Args:
        proba (numpy.array): predicted probability for class 0.
        metrics (list of str): list of names of metrics to compute.

    Returns:
        (dict): dictionary of computed metrics.
    """

    scores = {}
    for m in metrics:
        # mean of success probas (predicting the right class)
        if m == "probas_mean":
            if single_class_eval:
                scores[m] = np.mean(proba)
            else:
                proba_mean_0 = np.mean(proba[: len(proba) // 2])
                proba_mean_1 = 1 - np.mean(proba[len(proba) // 2 :])
                scores[m] = 1 / 2 * (proba_mean_0 + proba_mean_1)

        # std of probas
        elif m == "probas_std":
            scores[m] = np.std(proba)

        # wasserstein distance between dirac and probas
        elif m == "w_dist":
            scores[m] = wasserstein_distance([0.5] * len(proba), proba)

        # total variation distance between dirac and probas
        elif m == "TV":
            alphas = np.linspace(0, 1, 100)
            pp_vals_dirac = np.array(
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


def c2st_scores(
    P,
    Q,
    metrics=["accuracy"],
    n_folds=10,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    single_class_eval=False,
    cross_val=True,
):
    """Computes scores/metrics for a classifier trained on P and Q. 
    They represent the test statistics of the C2ST test estimated on P and Q.
    
    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        metrics (list of str, optional): list of names of metrics to compute.
            Defaults to ["accuracy"].
        n_folds (int, optional): number of folds for cross-validation.
            Defaults to 10.
        clf_class (sklearn model class, optional): the class of classifier to use.
            needs to have a methods `.fit(X,y)`, score(X,y)` and `.predict_proba(X)`.
            Defaults to MLPClassifier.
        clf_kwargs (dict, optional): keyword arguments for clf_class.
            Defaults to {"alpha": 0, "max_iter": 25000}.
        single_class_eval (bool, optional): if True, only evaluate on P.
            Defaults to False.
        cross_val (bool, optional): if True, perform cross-validation.
            Defaults to True.
    
    Returns:
        (dict): dictionary of computed scores, i.e. estimated test statistics on P and Q.
    """
    # initialize classifier
    classifier = clf_class(**clf_kwargs)

    if not cross_val:
        # train classifier
        clf = train_c2st(P, Q, clf=classifier)
        # eval classifier
        accuracy, proba = eval_c2st(P, Q, clf=clf, single_class_eval=single_class_eval)
        # compute metrics
        scores = {}
        for m in metrics:
            if "accuracy" in m:
                scores[m] = accuracy
            else:
                scores[m] = compute_metric(
                    proba, metrics=[m], single_class_eval=single_class_eval
                )[m]

    else:
        # initialize scores as dict of empty lists
        scores = dict(zip(metrics, [[] for _ in range(len(metrics))]))

        # cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(P):
            # split data into train and val sets for n^th cv-fold
            P_train = P[train_index]
            P_val = P[val_index]
            Q_train = Q[train_index]
            Q_val = Q[val_index]

            # train n^th classifier
            clf_n = train_c2st(P_train, Q_train, clf=classifier)
            # eval n^th classifier
            accuracy, proba = eval_c2st(
                P_val, Q_val, clf=clf_n, single_class_eval=single_class_eval
            )
            # compute metrics
            for m in metrics:
                if "accuracy" in m:
                    scores[m].append(accuracy)
                else:
                    scores[m].append(
                        compute_metric(
                            proba, metrics=[m], single_class_eval=single_class_eval
                        )[m]
                    )

    return scores


def t_stats_c2st(
    P,
    Q,
    null_samples_list,
    scores_fn=c2st_scores,
    metrics=["accuracy"],
    verbose=True,
    **kwargs,
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


# ==== experiments to evaluate the classifier and/or test statistics ====
import pandas as pd
import time


def eval_classifier_for_c2st(
    x_samples,
    ref_samples,
    shifted_samples,
    shifts,
    clf_class,
    clf_kwargs,
    metrics=["probas_mean"],
    n_folds=10,
    single_class_eval=False,
):
    shift_list = []
    scores = {}
    accuracies = []
    for m in metrics:
        scores[m] = []
    times = []
    for s_samples, s in zip(shifted_samples, shifts):

        x_samples_shuffled = np.random.permutation(x_samples)

        joint_P_x = np.concatenate([ref_samples, x_samples], axis=1)
        joint_Q_x = np.concatenate([s_samples, x_samples_shuffled], axis=1)

        start = time.time()

        score = c2st_scores(
            P=joint_P_x,
            Q=joint_Q_x,
            metrics=metrics,
            clf_class=clf_class,
            clf_kwargs=clf_kwargs,
            cross_val=True,
            n_folds=n_folds,
            single_class_eval=single_class_eval,
        )

        for m in metrics:
            scores[m] = np.concatenate([scores[m], score[m]])

        accuracies = np.concatenate([accuracies, score["accuracy"]])

        total_cv_time = time.time() - start

        for _ in range(n_folds):
            shift_list.append(s)
            times.append(total_cv_time)
    return shift_list, scores, accuracies, times


# evaluate the test-statistics for C2ST (precision under null)


def eval_null_c2st(
    x_samples,
    null_dist_samples,
    test_stats=["probas_mean"],
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    clf_name="mlp_base",
    n_samples=1000,
    n_folds=10,
    single_class_eval=True,
):

    scores = {}
    for m in test_stats:
        scores[m] = []

    P, Q = null_dist_samples
    x_samples_shuffled = np.random.permutation(x_samples)

    joint_P_x = np.concatenate([P, x_samples], axis=1)
    joint_Q_x = np.concatenate([Q, x_samples_shuffled], axis=1)

    start = time.time()

    scores = c2st_scores(
        P=joint_P_x,
        Q=joint_Q_x,
        metrics=test_stats + ["probas_std"],
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        cross_val=True,
        n_folds=n_folds,
        single_class_eval=single_class_eval,
    )
    total_cv_time = time.time() - start

    times = [total_cv_time] * n_folds
    n_samples_list = [n_samples] * n_folds
    classifier = [clf_name] * n_folds

    df = pd.DataFrame(
        {
            f"nb_samples": n_samples_list,
            "total_cv_time": times,
            "classifier": classifier,
        }
    )
    for m in test_stats + ["probas_std"]:
        df[m] = scores[m]

    return df

