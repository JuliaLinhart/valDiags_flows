import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import sklearn

from scipy.stats import wasserstein_distance
from .pp_plots import PP_vals

import pandas as pd

from tqdm import tqdm

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


def train_c2st(P, Q, clf=DEFAULT_CLF):

    # define features and labels for classification
    features = np.concatenate([P, Q], axis=0)
    labels = np.concatenate([np.array([0] * len(P)), np.array([1] * len(Q))]).ravel()

    features, labels = shuffle(features, labels)

    # train classifier
    clf = sklearn.base.clone(clf)
    clf.fit(X=features, y=labels)
    return clf


def eval_c2st(P, Q, clf, single_class_eval=False):
    # eval n^th classifier
    n_samples = len(P)
    if single_class_eval:
        X_val = P
        y_val = np.array([0] * (n_samples))
    else:
        X_val = np.concatenate([P, Q], axis=0)
        y_val = np.array([0] * n_samples + [1] * n_samples)

    accuracy = clf.score(X_val, y_val)

    proba = clf.predict_proba(X_val)[:, 0]
    return accuracy, proba


def compute_metric(proba, metrics):
    scores = {}
    for m in metrics:
        if m == "probas_mean":
            scores[m] = np.mean(proba)
        elif m == "probas_std":
            scores[m] = np.std(proba)
        elif m == "w_dist":  # wasserstein distance to dirac
            scores[m] = wasserstein_distance([0.5] * len(proba), proba)
        elif m == "TV":  # total variation: distance between cdfs of dirac and probas
            alphas = np.linspace(0, 1, 100)
            pp_vals_dirac = pd.Series(PP_vals([0.5] * len(proba), alphas))
            pp_vals = PP_vals(proba, alphas)
            scores[m] = ((pp_vals - pp_vals_dirac) ** 2).sum() / len(alphas)
        elif m == "div":
            mask = proba > 1 / 2
            max_proba = np.concatenate([proba[mask], 1 - proba[~mask]])
            scores[m] = np.mean(max_proba)
        elif m == "mse":
            scores[m] = ((proba - [0.5] * len(proba)) ** 2).mean()
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
    classifier = clf_class(**clf_kwargs)

    if not cross_val:
        scores = {}
        clf = train_c2st(P, Q, clf=classifier)
        accuracy, proba = eval_c2st(P, Q, clf=clf, single_class_eval=single_class_eval)
        for m in metrics:
            if "accuracy" in m:
                scores[m] = accuracy
            else:
                scores[m] = compute_metric(proba, metrics=[m])[m]

    else:

        scores = dict(zip(metrics, [[] for _ in range(len(metrics))]))

        kf = KFold(n_splits=n_folds, shuffle=True)
        for train_index, val_index in kf.split(P):
            P_train = P[train_index]
            P_val = P[val_index]
            Q_train = Q[train_index]
            Q_val = Q[val_index]

            # train n^th classifier
            clf_n = train_c2st(P_train, Q_train, clf=classifier)

            # eval n^th classifier
            n_samples = len(P_val)
            if single_class_eval:
                X_val = P_val
                y_val = np.array([0] * (n_samples))
            else:
                X_val = np.concatenate([P_val, Q_val], axis=0)
                y_val = np.array([0] * n_samples + [1] * n_samples)

            accuracy = clf_n.score(X_val, y_val)

            proba = clf_n.predict_proba(X_val)[:, 0]
            # if not single_class_eval:
            #     proba_1 = clf_n.predict_proba(Q_val)[:, 1]
            #     proba = np.concatenate([proba, proba_1], axis=0)

            for m in metrics:
                if "accuracy" in m:
                    scores[m].append(accuracy)
                else:
                    scores[m].append(compute_metric(proba, metrics=[m])[m])

    return scores


def t_stats_c2st(P, Q, null_samples_list, metrics=["accuracy"], verbose=True, **kwargs):
    t_stat_data = {}
    t_stats_null = dict(zip(metrics, [[] for _ in range(len(metrics))]))

    scores_data = c2st_scores(P=P, Q=Q, metrics=metrics, **kwargs)
    for m in metrics:
        t_stat_data[m] = np.mean(scores_data[m])
    for i in tqdm(
        range(len(null_samples_list)),
        desc="Testing under the null",
        disable=(not verbose),
    ):
        scores_null = c2st_scores(
            P=P, Q=null_samples_list[i], metrics=metrics, **kwargs,
        )
        for m in metrics:
            t_stats_null[m].append(np.mean(scores_null[m]))

    return t_stat_data, t_stats_null

