import numpy as np

from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import sklearn

from scipy.stats import wasserstein_distance
from .pp_plots import PP_vals

import pandas as pd
import matplotlib.pyplot as plt

import time

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


def c2st_clf(ndim):
    """ same setup as in :
    https://github.com/mackelab/sbi/blob/3e3522f177d4f56f3a617b2f15a5b2e25360a90f/sbi/utils/metrics.py
    """
    return MLPClassifier(
        **{
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        }
    )


def local_flow_c2st(flow_samples_train, x_train, classifier="mlp"):

    N = len(x_train)
    dim = flow_samples_train.shape[-1]
    reference = mvn(mean=np.zeros(dim), cov=np.eye(dim))  # base distribution

    # flow_samples_train = flow._transform(theta_train, context=x_train)[0].detach().numpy()
    ref_samples_train = reference.rvs(N)
    if dim == 1:
        ref_samples_train = ref_samples_train[:, None]

    features_flow_train = np.concatenate([flow_samples_train, x_train], axis=1)
    features_ref_train = np.concatenate([ref_samples_train, x_train], axis=1)
    features_train = np.concatenate([features_ref_train, features_flow_train], axis=0)
    labels_train = np.concatenate([np.array([0] * N), np.array([1] * N)]).ravel()
    features_train, labels_train = shuffle(
        features_train, labels_train, random_state=13
    )

    if classifier == "mlp":
        clf = c2st_clf(features_train.shape[-1])
    else:
        clf = DEFAULT_CLF

    clf.fit(X=features_train, y=labels_train)

    return clf


def eval_local_flow_c2st(clf, x_eval, dim, size=1000, z_values=None):
    if z_values is not None:
        z_values = z_values
    else:
        # sample from normal dist (class 0)
        z_values = mvn(mean=np.zeros(dim), cov=np.eye(dim)).rvs(size)

    if dim == 1 and z_values.ndim == 1:
        z_values = z_values.reshape(-1, 1)

    assert (z_values.shape[0] == size) and (z_values.shape[-1] == dim)

    features_eval = np.concatenate([z_values, x_eval.repeat(size, 1)], axis=1)
    proba = clf.predict_proba(features_eval)[:, 0]

    return proba, z_values


### ================ new functions ================


def train_lc2st(P, Q, x, clf=DEFAULT_CLF):
    # joint samples
    joint_P_x = np.concatenate([P, x], axis=1)
    joint_Q_x = np.concatenate([Q, x], axis=1)

    # define features and labels for classification
    features = np.concatenate([joint_P_x, joint_Q_x], axis=0)
    labels = np.concatenate([np.array([0] * len(x)), np.array([1] * len(x))]).ravel()

    features, labels = shuffle(features, labels, random_state=1)

    # train classifier
    clf = sklearn.base.clone(clf)
    clf.fit(X=features, y=labels)
    return clf


def eval_lc2st(P, x, clf=DEFAULT_CLF):
    # define eval features for classifier
    features_eval = np.concatenate([P, x.repeat(len(P), 1)], axis=1)
    # predict proba for class 0 (P_dist)
    proba = clf.predict_proba(features_eval)[:, 0]
    return proba


def compute_metric(proba, metrics):
    scores = {}
    for m in metrics:
        if m == "accuracy":
            scores[m] = np.mean(proba >= 0.5)
        elif m == "probas_mean":
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
        else:
            scores[m] = None
            print(f'metric "{m}" not implemented')
    return scores


def lc2st_scores(
    P,
    Q,
    x_cal,
    x_eval,
    metrics=["probas_mean"],
    n_folds=10,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
):

    classifier = clf_class(**clf_kwargs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    probas = []
    scores = {}
    for m in metrics:
        scores[m] = []
    for train_index, val_index in kf.split(P):
        P_train = P[train_index]
        P_eval = P[val_index]
        Q_train = Q[train_index]
        x_train = x_cal[train_index]

        # train n^th classifier
        clf_n = train_lc2st(P_train, Q_train, x_train, clf=classifier)

        # eval n^th classifier
        proba = eval_lc2st(P_eval, x_eval, clf=clf_n)
        probas.append(proba)
        score = compute_metric(proba, metrics=metrics)

        for m in metrics:
            scores[m].append(score[m])

    return scores, probas


def lc2st_htest(
    P_cal,
    Q_cal,
    x_cal,
    P_eval,
    x_eval,
    null_dist,
    test_stats=["probas_mean"],
    n_trials_null=100,
    n_ensembles=10,
    clf=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    probas_null=[],
):
    classifier = clf(**clf_kwargs)
    probas = []
    for _ in range(n_ensembles):
        # train clf
        clf_n = train_lc2st(P_cal, Q_cal, x_cal, clf=classifier)
        # eval clf
        probas.append(eval_lc2st(P_eval, x_eval, clf=clf_n))

    proba_ensemble = np.mean(probas, axis=0)
    t_stats_ensemble = compute_metric(proba_ensemble, metrics=test_stats)

    t_stats_null = {}
    for m in test_stats:
        t_stats_null[m] = []
    for t in range(n_trials_null):
        while len(probas_null) < n_trials_null:
            null_samples = null_dist.sample((len(x_cal),))
            # train clf under null
            clf_t = train_lc2st(P_cal, null_samples, x_cal, clf=classifier)
            # eval clf
            probas_null.append(eval_lc2st(P_eval, x_eval, clf=clf_t))

        # compute test stat
        scores = compute_metric(probas_null[t], metrics=test_stats)
        for m in test_stats:
            t_stats_null[m].append(scores[m])

    p_values = {}
    for m in test_stats:
        p_values[m] = (
            sum(1 * (t_stats_ensemble[m] < pd.Series(t_stats_null[m]))) / n_trials_null
        )

    return p_values, t_stats_ensemble, proba_ensemble, probas_null, t_stats_null


## expected c2st score
def expected_lc2st_scores(
    P,
    Q,
    x_cal,
    metrics=["probas_mean"],
    n_folds=10,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
):

    classifier = clf_class(**clf_kwargs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    scores = {"accuracy": []}
    for m in metrics:
        scores[m] = []

    for train_index, val_index in kf.split(P):
        P_train = P[train_index]
        P_eval = P[val_index]
        Q_train = Q[train_index]
        Q_eval = Q[val_index]
        x_train = x_cal[train_index]
        x_eval = x_cal[val_index]

        # train n^th classifier
        clf_n = train_lc2st(P_train, Q_train, x_train, clf=classifier)

        # eval n^th classifier
        # joint samples
        joint_P_x = np.concatenate([P_eval, x_eval], axis=1)
        joint_Q_x = np.concatenate([Q_eval, x_eval], axis=1)

        # define features and labels for classification
        features = np.concatenate([joint_P_x, joint_Q_x], axis=0)
        labels = np.concatenate(
            [np.array([0] * len(x_eval)), np.array([1] * len(x_eval))]
        ).ravel()

        accuracy = clf_n.score(features, labels)
        scores["accuracy"].append(accuracy)

        proba = clf_n.predict_proba(joint_P_x)[:, 0]
        for m in metrics:
            scores[m].append(compute_metric(proba, [m])[m])

    return scores


## ==================== plots ========================
def pp_plot_lc2st(probas, probas_null, labels, colors):
    alphas = np.linspace(0, 1, 100)
    pp_vals_dirac = PP_vals([0.5] * len(probas), alphas)
    plt.plot(alphas, pp_vals_dirac, "--", color="black")

    pp_vals_null = {}
    for t in range(len(probas_null)):
        pp_vals_null[t] = pd.Series(PP_vals(probas_null[t], alphas))

    low_null = pd.DataFrame(pp_vals_null).quantile(0.05 / 2, axis=1)
    up_null = pd.DataFrame(pp_vals_null).quantile(1 - 0.05 / 2, axis=1)
    plt.fill_between(
        alphas,
        low_null,
        up_null,
        color="grey",
        alpha=0.3,
        label="95% confidence region",
    )

    for p, l, c in zip(probas, labels, colors):
        pp_vals = pd.Series(PP_vals(p, alphas))
        plt.plot(alphas, pp_vals, label=l, color=c)

    plt.legend()
    plt.show()


def box_plot_lc2st(
    scores, scores_null, labels, colors, title=r"Box plot for l-c2st at $x_0$"
):
    import matplotlib.cbook as cbook

    data = scores_null
    stats = cbook.boxplot_stats(data)[0]
    stats["q1"] = np.quantile(data, 0.05)
    stats["q3"] = np.quantile(data, 0.95)
    stats["whislo"] = min(data)
    stats["whishi"] = max(data)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    bp = ax.bxp([stats], widths=0.1, vert=False, showfliers=False, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgray")
    ax.set_label(r"95% confidence interval for $\mathcal{H}_0(x_0)$")
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(stats["whislo"] - np.std(scores) / 2, max(scores) + np.std(scores) / 2)

    for s, l, c in zip(scores, labels, colors):
        plt.text(s, 0.9, l, color=c)
        plt.scatter(s, 1, color=c, zorder=10)

    fig.set_size_inches(5, 2)
    plt.title(title)
    plt.show()


## =============== eval clfs : shift experiment ========================
# like in c2st... gl2st...
def eval_classifier_for_lc2st(
    x_samples,
    ref_samples,
    shifted_samples,
    shifts,
    clf_class,
    clf_kwargs,
    metrics=["probas_mean"],
    n_folds=10,
):
    shift_list = []
    scores = {}
    accuracies = []
    for m in metrics:
        scores[m] = []
    times = []
    for s_samples, s in zip(shifted_samples, shifts):
        start = time.time()

        score = expected_lc2st_scores(
            ref_samples,
            s_samples,
            x_samples,
            metrics=metrics,
            n_folds=n_folds,
            clf_class=clf_class,
            clf_kwargs=clf_kwargs,
        )

        for m in metrics:
            scores[m] = np.concatenate([scores[m], score[m]])

        accuracies = np.concatenate([accuracies, score["accuracy"]])

        total_cv_time = time.time() - start

        for _ in range(n_folds):
            shift_list.append(s)
            times.append(total_cv_time)
    return shift_list, scores, accuracies, times


## =============== eval test-stats (precision under null) ========================
def eval_null_lc2st(
    x_samples,
    null_dist,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    clf_name="mlp_base",
    test_stats=["probas_mean"],
    n=1000,
    n_folds=10,
):

    scores = {}
    for m in test_stats:
        scores[m] = []

    P = null_dist.sample((n,))
    Q = null_dist.sample((n,))
    start = time.time()

    scores = expected_lc2st_scores(
        P,
        Q,
        x_samples[n],
        metrics=test_stats + ["probas_std"],
        n_folds=n_folds,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
    )
    total_cv_time = time.time() - start

    times = [total_cv_time] * n_folds
    nb_samples = [n] * n_folds
    classifier = [clf_name] * n_folds

    df = pd.DataFrame(
        {f"nb_samples": nb_samples, "total_cv_time": times, "classifier": classifier}
    )
    for m in test_stats + ["probas_std"]:
        df[m] = scores[m]

    return df


### =============== functions adapted to zuko ==============
def lc2st_scores_flow_zuko(
    flow,
    theta_cal,
    x_cal,
    x_eval,
    metrics=["probas_mean"],
    n_folds=10,
    clf=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
):
    inv_flow_samples = flow(x_cal).transform(theta_cal).detach().numpy()
    base_dist_samples = flow(x_cal).base.sample().numpy()

    return lc2st_scores(
        P=base_dist_samples,
        Q=inv_flow_samples,
        x_cal=x_cal,
        x_eval=x_eval,
        n_folds=n_folds,
        metrics=metrics,
        clf=clf,
        clf_kwargs=clf_kwargs,
    )
