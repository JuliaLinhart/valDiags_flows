import numpy as np
import torch

from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import sklearn

from scipy.stats import wasserstein_distance
from .pp_plots import PP_vals
from .plot_utils import plot_distributions

import pandas as pd
import matplotlib.pyplot as plt

import time

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


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
    Z_eval=None,
):

    classifier = clf_class(**clf_kwargs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    probas = []
    scores = {}
    for m in metrics:
        scores[m] = []
    for train_index, val_index in kf.split(P):
        P_train = P[train_index]
        if Z_eval is None:
            P_eval = P[val_index]
        else:
            P_eval = Z_eval[val_index]
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
    n_ensemble=10,
    clf=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    probas_null=[],
):
    classifier = clf(**clf_kwargs)
    probas = []
    for _ in range(n_ensemble):
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
    ax.set_xlim(stats["whislo"] - np.std(data), max(scores) + np.std(data))

    for s, l, c in zip(scores, labels, colors):
        plt.text(s, 0.9, l, color=c)
        plt.scatter(s, 1, color=c, zorder=10)

    fig.set_size_inches(5, 2)
    plt.title(title)


## =============== reference/ground-truth distributions ==============================


def flow_vs_reference_distribution(
    samples_ref, samples_flow, z_space=True, dim=1, hist=False
):
    if z_space:
        title = (
            r"Base-Distribution vs. Inverse Flow-Transformation (of $\Theta \mid x_0$)"
        )
        labels = [
            r"Ref: $\mathcal{N}(0,1)$",
            r"NPE: $T_{\phi}^{-1}(\Theta;x_0) \mid x_0$",
        ]
    else:
        title = r"True vs. Estimated distributions at $x_0$"
        labels = [r"Ref: $p(\Theta \mid x_0)$", r"NPE: $p(T_{\phi}(Z;x_0))$"]

    if hist:
        colors = ["Blues", "Oranges"]
    else:
        colors = ["blue", "orange"]
    plot_distributions(
        [samples_ref, samples_flow],
        colors=colors,
        labels=labels,
        dim=dim,
        hist=hist,
    )
    plt.title(title)

    if dim == 1:
        plt.xlabel("z")
        plt.xlim(-5, 5)

    elif dim == 2:
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
    plt.legend()


## =============== interpretability ==============================


def z_space_with_proba_intensity(
    probas, probas_null, P_eval, theta_space=None, dim=1, thresholding=False
):
    df = pd.DataFrame({"probas": probas})

    # define low and high thresholds w.r.t to null (95% confidence region)
    low = np.quantile(np.mean(probas_null, axis=0), q=0.05)
    high = np.quantile(np.mean(probas_null, axis=0), q=0.95)
    # high/low proba regions for bad NF
    df["intensity"] = ["uncertain"] * len(df)
    df.loc[df["probas"] > high, "intensity"] = (
        r"high ($p \geq$ " + f"{np.round(high,2)})"
    )
    df.loc[df["probas"] < low, "intensity"] = r"low ($p \leq$ " + f"{np.round(low,2)})"

    if dim == 1:
        from matplotlib import cm

        df["z"] = P_eval[:, 0]
        values = "z"
        xlabel = r"$z$"
        x = df.z
        if theta_space is not None:
            df["theta"] = theta_space
            values = "theta"
            xlabel = r"$\theta$"
            x = df.theta

        if thresholding:
            df.pivot(columns="intensity", values=values).plot.hist(
                bins=50, color=["red", "blue", "grey"], alpha=0.3
            )
        else:
            _, bins, patches = plt.hist(x, 50, density=True, color="green")
            bins[-1] = 10
            df["bins"] = np.select([x <= i for i in bins[1:]], list(range(50)), 1000)

            weights = df.groupby(["bins"]).mean().probas
            id = list(set(range(50)) - set(df.bins))
            patches = np.delete(patches, id)

            cmap = plt.cm.get_cmap("bwr")
            for c, p in zip(weights, patches):
                plt.setp(p, "facecolor", cmap(c))
            plt.colorbar(
                cm.ScalarMappable(cmap=cmap),
                label=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$",
            )
        plt.xlabel(xlabel)

    elif dim == 2:
        df["z_1"] = P_eval[:, 0]
        df["z_2"] = P_eval[:, 1]
        x, y = df.z_1, df.z_2
        xlabel = r"$Z_1$"
        ylabel = r"$Z_2$"
        if theta_space is not None:
            df["theta_1"] = theta_space[:, 0]
            df["theta_2"] = theta_space[:, 1]
            x, y = df.theta_1, df.theta_2
            xlabel = r"$\Theta_1 = T_{\phi,1}(Z; x_0)$"
            ylabel = r"$\Theta_2 = T_{\phi,2}(Z; x_0)$"
        if not thresholding:
            plt.scatter(x, y, c=df.probas, cmap="bwr", alpha=0.3)
            plt.colorbar(label=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$")
        else:
            cdict = {
                "uncertain": "grey",
                r"high ($p \geq$ " + f"{np.round(high,2)})": "red",
                r"low ($p \leq$ " + f"{np.round(low,2)})": "blue",
            }
            groups = df.groupby("intensity")

            _, ax = plt.subplots()
            for name, group in groups:
                x = group.z_1
                y = group.z_2
                if theta_space is not None:
                    x = group.theta_1
                    y = group.theta_2
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="",
                    alpha=0.3,
                    label=name,
                    color=cdict[name],
                )
            plt.legend(title=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    else:
        print("Not implemented.")


def eval_space_with_proba_intensity(
    probas, probas_null, P_eval, dim=1, z_space=True, thresholding=False
):
    df = pd.DataFrame({"probas": probas})

    # define low and high thresholds w.r.t to null (95% confidence region)
    low = np.quantile(np.mean(probas_null, axis=0), q=0.05)
    high = np.quantile(np.mean(probas_null, axis=0), q=0.95)
    # high/low proba regions for bad NF
    df["intensity"] = ["uncertain"] * len(df)
    df.loc[df["probas"] > high, "intensity"] = (
        r"high ($p \geq$ " + f"{np.round(high,2)})"
    )
    df.loc[df["probas"] < low, "intensity"] = r"low ($p \leq$ " + f"{np.round(low,2)})"

    if dim == 1:
        from matplotlib import cm

        df["z"] = P_eval[:, 0]

        if thresholding:
            df.pivot(columns="intensity", values="z").plot.hist(
                bins=50, color=["red", "blue", "grey"], alpha=0.3
            )
        else:
            _, bins, patches = plt.hist(df.z, 50, density=True, color="green")
            bins[-1] = 10
            df["bins"] = np.select([x <= i for i in bins[1:]], list(range(50)), 1000)

            weights = df.groupby(["bins"]).mean().probas
            id = list(set(range(50)) - set(df.bins))
            patches = np.delete(patches, id)

            cmap = plt.cm.get_cmap("bwr")
            for c, p in zip(weights, patches):
                plt.setp(p, "facecolor", cmap(c))
            plt.colorbar(
                cm.ScalarMappable(cmap=cmap),
                label=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$",
            )
        if z_space:
            xlabel = r"$z$"
        else:
            xlabel = r"$\theta$"
        plt.xlabel(xlabel)

    elif dim == 2:
        if z_space:
            xlabel = r"$Z_1$"
            ylabel = r"$Z_2$"
            legend = r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$"
        else:
            xlabel = r"$\Theta_1$"
            ylabel = r"$\Theta_2$"
            legend = r"$\hat{p}(\Theta\sim q_{\phi}(\theta \mid x_0) \mid x_0)$"

        df["z_1"] = P_eval[:, 0]
        df["z_2"] = P_eval[:, 1]

        if not thresholding:
            plt.scatter(df.z_1, df.z_2, c=df.probas, cmap="bwr", alpha=0.3)
            plt.colorbar(label=legend)
        else:
            cdict = {
                "uncertain": "grey",
                r"high ($p \geq$ " + f"{np.round(high,2)})": "red",
                r"low ($p \leq$ " + f"{np.round(low,2)})": "blue",
            }
            groups = df.groupby("intensity")

            _, ax = plt.subplots()
            for name, group in groups:
                x = group.z_1
                y = group.z_2
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="",
                    alpha=0.3,
                    label=name,
                    color=cdict[name],
                )
            plt.legend(title=legend)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    else:
        print("Not implemented.")


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


### =============== functions adapted to sbibm run.py script ==============


def c2st_clf(ndim):
    """same setup as in :
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


def c2st_kwargs(ndim):
    """same setup as in :
    https://github.com/mackelab/sbi/blob/3e3522f177d4f56f3a617b2f15a5b2e25360a90f/sbi/utils/metrics.py
    """
    return {
        "activation": "relu",
        "hidden_layer_sizes": (10 * ndim, 10 * ndim),
        "max_iter": 1000,
        "solver": "adam",
        "early_stopping": True,
        "n_iter_no_change": 50,
    }


def lc2st_sbibm(
    P, Q, x_cal, x_eval, metric="accuracy", n_folds=10, classifier=None, Z_eval=None
):
    ndim = P.shape[-1] + x_cal.shape[-1]
    if classifier is None:
        classifier = MLPClassifier(**c2st_kwargs(ndim))
    scores, _ = lc2st_scores(
        P, Q, x_cal, x_eval, metrics=[metric], n_folds=n_folds, Z_eval=Z_eval
    )
    return torch.tensor([np.mean(scores[metric])])


## expected c2st score
def expected_lc2st_sbibm(
    P,
    Q,
    x_cal,
    metric="accuracy",
    n_folds=10,
    clf_class=None,
    clf_kwargs=None,
):
    if clf_class is None or clf_kwargs is None:
        ndim = P.shape[-1] + x_cal.shape[-1]
        clf_class = MLPClassifier
        clf_kwargs = c2st_kwargs(ndim)

    scores = expected_lc2st_scores(
        P,
        Q,
        x_cal,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        n_folds=n_folds,
    )
    score = np.mean(scores[metric])

    return torch.tensor([score])


def lc2st_htest_sbibm(
    P_cal,
    Q_cal,
    x_cal,
    P_eval,
    x_eval,
    null_dist,
    null_samples_list=None,
    test_stats=["probas_mean"],
    n_trials_null=100,
    n_ensemble=10,
    clf=MLPClassifier,
    clf_kwargs=None,
    probas_null=[],
    trained_clfs=[],
):
    if clf_kwargs is None:
        ndim = P_eval.shape[-1] + x_cal.shape[-1]
        classifier = MLPClassifier(**c2st_kwargs(ndim))
    else:
        classifier = clf(**clf_kwargs)

    probas = []
    clfs = []
    run_time = 0
    for n in range(n_ensemble):
        try:
            # load clf
            clf_n = trained_clfs[n]
            print("loaded clf")
        except IndexError:
            # train clf
            print(f"training clf {n+1}/{n_ensemble}")
            start = time.time()
            clf_n = train_lc2st(P_cal, Q_cal, x_cal, clf=classifier)
            run_time = time.time() - start
        clfs.append(clf_n)
        # eval clf
        probas.append(eval_lc2st(P_eval, x_eval, clf=clf_n))

    proba_ensemble = np.mean(probas, axis=0)
    t_stats_ensemble = compute_metric(proba_ensemble, metrics=test_stats)

    t_stats_null = {}
    for m in test_stats:
        t_stats_null[m] = []
    print("Testing under null hypothesis...")
    for t in range(n_trials_null):
        print(f"...trial {t+1}/{n_trials_null}")
        if len(probas_null) < n_trials_null:
            if null_samples_list is None:
                try:
                    null_samples = null_dist.sample(len(x_cal))
                except TypeError:
                    null_samples = null_dist.sample((len(x_cal),))
            else:
                null_samples = null_samples_list[t]
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

    return (
        p_values,
        t_stats_ensemble,
        proba_ensemble,
        probas_null,
        t_stats_null,
        clfs,
        run_time,
    )
