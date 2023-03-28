# P-values and ROC curves

import numpy as np
from scipy.stats import multivariate_normal as mvn

import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

# GLOBALS
N_SAMPLES = 1000  # number of samples from P and Q ('n0 = n1' in [Lee et al. (2018)]) (same for train and evaluation)
N_TRIALS_NULL = 100  # number of times to compute the test statistic under (H0) (nb of permuations 'B' in [Lee et al. (2018)])
N_RUNS = 1000  # number of test runs to compute the empirical power
ALPHA_LIST = np.linspace(0, 1, 100)  # significance levels alpha in (0,1)

use_permutation = True
in_sample = False
single_class_eval = True


def c2st_p_values_tfpr(
    eval_c2st_fn,
    P_dist,
    Q_dist,
    metrics,
    metrics_cv=None,
    n_samples=N_SAMPLES,
    n_runs=N_RUNS,
    alpha_list=ALPHA_LIST,
    compute_FPR=True,
):
    """Computes the p-values, TPR and FPR over several runs of the Classifier Two Sample Test (C2ST)
    between two distributions P and Q:

                                         (H0): P = Q   (H1): P != Q 
    for different metrics (test statistics).

    The p-value of a test-run is defined as the probability of falsely rejecting the null hypothesis (H0).
    For a given significance level alpha, we reject the null hypothesis (H0) if p-value < alpha.
    - TPR is the average number of times we correctly reject the null hypothesis (H0): power of the test.
    - FPR is the average number of times we incorrectly reject the null hypothesis (H0).
    We compute them for a range of significance levels alpha in (0,1), so that we can plot the ROC curve.
    
    Args:
        eval_c2st_fn (function): function that evaluates the C2ST test
        P_dist (scipy.stats.rv_continuous): distribution of P
        Q_dist (scipy.stats.rv_continuous): distribution of Q
        metrics (list): list of metrics to be used for the test (test statistics)
        metrics_cv (list): list of metrics to be used for the cross-validation. 
            Defauts to None.
        n_samples (int): number of samples from P and Q (same for train and evaluation).
            Defaults to 1000.
        n_runs (int): number of test runs to compute FPR and TPR.
            Defaults to 1000.
        alpha_list (list): list of significance levels alpha in (0,1) to compute FPR and TPR at
            Defaults to np.linspace(0, 1, 100).
        compute_FPR (bool): whether to compute FPR or not. 
            Defaults to True.
    
    Returns:
        p_values_H1 (dict): dict of p-values for each metric under (H1)
        p_values_H0 (dict): dict of p-values for each metric under (H0)
        TPR (dict): dict of TPR for each metric at each alpha in alpha_list
        FPR (dict): dict of FPR for each metric at each alpha in alpha_list
    """
    all_metrics = metrics
    if metrics_cv is not None:
        # combine metrics and metrics_cv
        all_metrics = metrics + metrics_cv

    # initialize dict with empty lists
    p_values_H1 = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    p_values_H0 = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

    # loop over test runs
    for _ in tqdm(range(n_runs), desc="Test runs"):
        # generate samples from P and Q
        P = P_dist.rvs(n_samples)
        Q = Q_dist.rvs(n_samples)
        Q_H0 = P_dist.rvs(n_samples)

        P_eval = P_dist.rvs(n_samples)
        Q_eval = Q_dist.rvs(n_samples)
        Q_H0_eval = P_dist.rvs(n_samples)

        if not use_permutation:
            # generate samples from P to compute the test statistic under (H0)
            null_samples_list = [P_dist.rvs(n_samples) for _ in range(N_TRIALS_NULL)]
        else:
            null_samples_list = None

        # evaluate test under (H1)
        _, p_value = eval_c2st_fn(
            metrics=metrics,
            P=P,
            Q=Q,
            null_samples_list=null_samples_list,
            n_trials_null=N_TRIALS_NULL,
            cross_val=False,
            in_sample=in_sample,
            P_eval=P_eval,
            Q_eval=Q_eval,
        )
        # update the empirical power at alpha for each metric
        for m in metrics:
            p_values_H1[m].append(p_value[m])

        if compute_FPR:
            # evaluate test under (H0)
            _, p_value = eval_c2st_fn(
                metrics=metrics,
                P=P,
                Q=Q_H0,
                null_samples_list=null_samples_list,
                n_trials_null=N_TRIALS_NULL,
                cross_val=False,
                in_sample=in_sample,
                P_eval=P_eval,
                Q_eval=Q_H0_eval,
            )
            # update the FPR at alpha for each metric
            for m in metrics:
                p_values_H0[m].append(p_value[m])

        if metrics_cv is not None:

            # evaluate test under (H1) over several cross-val folds
            _, p_value_cv = eval_c2st_fn(
                metrics=metrics_cv,
                P=P,
                Q=Q,
                null_samples_list=null_samples_list,
                n_trials_null=N_TRIALS_NULL,
                cross_val=True,
                n_folds=2,
            )
            # update the empirical power at alpha for each cv-metric
            for m in metrics_cv:
                p_values_H1[m].append(p_value_cv[m])

            if compute_FPR:
                # evaluate test under (H0) over several cross-val folds
                _, p_value_cv = eval_c2st_fn(
                    metrics=metrics_cv,
                    P=P,
                    Q=Q_H0,
                    null_samples_list=null_samples_list,
                    n_trials_null=N_TRIALS_NULL,
                    cross_val=True,
                    n_folds=2,
                )
                # update the FPR at alpha for each cv-metric
                for m in metrics_cv:
                    p_values_H0[m].append(p_value_cv[m])

    # compute TPR and TPF at every alpha
    TPR = dict(zip(all_metrics, [[0] for _ in range(len(all_metrics))]))
    FPR = dict(zip(all_metrics, [[0] for _ in range(len(all_metrics))]))
    for alpha in alpha_list:
        # append TPR/TPF at alpha for each metric
        for m in all_metrics:
            TPR[m].append(np.mean(np.array(p_values_H1[m]) < alpha))
            FPR[m].append(np.mean(np.array(p_values_H0[m]) < alpha))

    return TPR, FPR, p_values_H1, p_values_H0


if __name__ == "__main__":
    from valdiags.test_utils import eval_htest
    from valdiags.vanillaC2ST import t_stats_c2st

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # ESTIMATED LDA

    # define function to evaluate the test
    eval_c2st_lda = partial(
        eval_htest,
        t_stats_estimator=t_stats_c2st,
        clf_class=LinearDiscriminantAnalysis,
        clf_kwargs={},
        single_class_eval=single_class_eval,
        verbose=False,
    )

    # define distributions P and Q
    dim = 5  # data dimension
    P_dist = mvn(mean=np.zeros(dim), cov=np.eye(dim))
    mu = 0.07  # mean shift between P and Q
    Q_dist = mvn(mean=np.array([mu] * dim), cov=np.eye(dim))

    # metrics / test statistics
    metrics = ["accuracy", "div", "mse"]
    metrics_cv = ["accuracy_cv", "div_cv", "mse_cv"]

    # compute p_value at alpha=0.05, for each metric with `eval_c2st_lda`
    TPR, FPR, p_values_H1, p_values_H0 = c2st_p_values_tfpr(
        eval_c2st_fn=eval_c2st_lda,
        P_dist=P_dist,
        Q_dist=Q_dist,
        metrics=metrics,
        metrics_cv=metrics_cv,
        compute_FPR=True,
        alpha_list=ALPHA_LIST,
    )

    # plot p-values for each metric

    for m in metrics + metrics_cv:
        p_values = np.concatenate(
            [p_values_H1[m], p_values_H0[m]]
        )  # concatenate H1 and H0 p-values
        index = np.concatenate(
            [np.ones(N_RUNS), np.zeros(N_RUNS)]
        )  # 1 for H1, 0 for H0
        sorter = np.argsort(p_values)  # sort p-values
        sorted_index = index[sorter]  # sort index
        idx_0 = np.where(sorted_index == 0)[0]  # find index of H0 p-values
        idx_1 = np.where(sorted_index == 1)[0]  # find index of H1 p-values

        plt.plot(np.sort(p_values), color="blue", label="p-values")

        plt.scatter(
            np.arange(2 * N_RUNS)[idx_1],
            np.sort(p_values)[idx_1],
            c="g",
            label=f"H1 (mu={mu})",
            alpha=0.3,
        )
        plt.scatter(
            np.arange(2 * N_RUNS)[idx_0],
            np.sort(p_values)[idx_0],
            c="r",
            label="H0",
            alpha=0.3,
        )
        plt.legend()
        plt.title(f"C2ST-{m}, single-class (N={N_SAMPLES})")
        plt.savefig(
            f"p_values_{m}_lqda_mu_{mu}_dim_{dim}_nruns_{N_RUNS}_single_class_{single_class_eval}_permutation_{use_permutation}_insample_{in_sample}.pdf"
        )
        plt.show()

    # roc curve
    for m in metrics + metrics_cv:
        plt.plot(FPR[m], TPR[m], label=m)
    plt.legend()
    plt.title(f"ROC for C2ST (H1): mu={mu}, single-class, (N={N_SAMPLES})")
    plt.savefig(
        f"roc_lqda_mu_{mu}_dim_{dim}_nruns_{N_RUNS}_single_class_{single_class_eval}_permutation_{use_permutation}_insample_{in_sample}.pdf"
    )
    plt.show()

