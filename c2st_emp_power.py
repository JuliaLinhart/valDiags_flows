# Empirical power analysis
# see Figure 2 in [Lee et al. (2018)](https://arxiv.org/abs/1812.08927)

import numpy as np
from scipy.stats import multivariate_normal as mvn

import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

# GLOBALS
N_SAMPLES = 1000  # number of samples from P and Q ('n0 = n1' in [Lee et al. (2018)]) (same for train and evaluation)
N_TRIALS_NULL = 100  # number of times to compute the test statistic under (H0) (nb of permuations 'B' in [Lee et al. (2018)])
N_RUNS = 300  # number of test runs to compute the empirical power
ALPHA_LIST = np.linspace(0, 1, 20)  # significance levels alpha in (0,1)

DIM = 5  # dimension of the Gaussians
mu = np.sqrt(0.05)  # mean shift between P and Q

# DIM = 20  # dimension of the Gaussians
# mu = np.sqrt(0.01)  # mean shift between P and Q

use_permutation = True
in_sample = True

# whether to use single class evaluation
single_class_eval = False  # if True, we evaluate the classifier on samples from P only


def empirical_power_c2st_mean_shift(
    eval_c2st_fn,
    metrics,
    metrics_cv=None,
    n_samples=N_SAMPLES,
    n_runs=N_RUNS,
    alpha_list=ALPHA_LIST,
    compute_FPR=False,
):
    """Computes the empirical power of the Classifier Two Sample Test (C2ST) between 
    (multivariate) Gaussians with different mean P ~ N(0,1) and Q ~ N(mu,1):

                            (H0): P=Q vs. (H1): P \neq Q 
    
    over a range of significance levels alpha in (0,1).
    
    The empirical power of a test is defined as the True Positive Rate (TPR), 
    i.e. how many times we (correctly) reject the null hypothesis (H0) over several test 
    runs performed under (H1). Here this corresponds to a test statistic that 
    was estimated on data from P and Q (different from P).

    Args:
        eval_c2st_fn (function): built on `eval_htest` with additional inputs:
            - samples from P, Q 
            - a list of samples from P to compute the test statistic under (H0)
        metrics (list of str): contains the names of the metrics that are used in the scoring function 
            of the test (as in vanillaC2ST.c2st_scores) (= test statistics) 
        metrics_cv (list of str): subset of metrics we want to compute over several cross-val folds
        n_runs (int, optional): number of test runs to compute TPR for a given significance level alpha. 
            Defaults to N_RUNS.
        n_alpha (int, optional): number of significance levels alpha in (0,1). 
            --> alpha in `np.linspace(0,1, n_alpha)`
            Defaults to N_ALPHA.
        compute_FPR (bool, optional): whether to compute the False Positive Rate (FPR)
            Defaults to False.

    Returns:
        emp_power (dict): contains the TPR of each metric, computed for every alpha   
        FPR (dict): contains the FPR of each metric, computed for every alpha
        p_values (dict): contains the p-values of every test-run for each metric, computed for every alpha
        p_values_FPR (dict): contains the p-values of every test-run for each metric, computed for every alpha
    """
    all_metrics = metrics
    if metrics_cv is not None:
        # combine metrics and metrics_cv
        all_metrics = metrics + metrics_cv

    # initialize dict with empty lists
    emp_power = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    FPR = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

    # loop over significance levels alpha
    for i, alpha in enumerate(alpha_list):

        print(f"Significance level alpha = {alpha} ({i+1}/{len(alpha_list)})")

        # if alpha == 0, we don't need to perform the test
        if alpha == 0:
            for m in all_metrics:
                emp_power[m].append(0)
                FPR[m].append(0)

        else:
            # initialize dict with 0s
            emp_power_a = dict(zip(all_metrics, [0] * len(all_metrics)))
            FPR_a = dict(zip(all_metrics, [0] * len(all_metrics)))

            # loop over test runs
            for _ in tqdm(range(n_runs), desc="Empirical power / success rate"):
                # generate samples from P and Q
                P = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(n_samples)
                Q_FPR = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(n_samples)
                Q = mvn(mean=np.array([mu] * DIM), cov=np.eye(DIM)).rvs(n_samples)

                P_eval = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(n_samples)
                Q_eval = mvn(mean=np.array([mu] * DIM), cov=np.eye(DIM)).rvs(n_samples)
                Q_FPR_eval = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(n_samples)

                if not use_permutation:
                    # generate samples from P to compute the test statistic under (H0)
                    null_samples_list = [
                        mvn(mean=np.array([0] * DIM), cov=np.eye(DIM)).rvs(n_samples)
                        for _ in range(N_TRIALS_NULL)
                    ]
                else:
                    null_samples_list = None

                # evaluate test under (H1)
                reject_test, _ = eval_c2st_fn(
                    metrics=metrics,
                    conf_alpha=alpha,
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
                    emp_power_a[m] += reject_test[m] / n_runs

                if compute_FPR:
                    # evaluate test under (H0)
                    reject_test_FPR, _ = eval_c2st_fn(
                        metrics=metrics,
                        conf_alpha=alpha,
                        P=P,
                        Q=Q_FPR,
                        null_samples_list=null_samples_list,
                        n_trials_null=N_TRIALS_NULL,
                        cross_val=False,
                        in_sample=in_sample,
                        P_eval=P_eval,
                        Q_eval=Q_FPR_eval,
                    )
                    # update the FPR at alpha for each metric
                    for m in metrics:
                        FPR_a[m] += reject_test_FPR[m] / n_runs

                if metrics_cv is not None:

                    # evaluate test under (H1) over several cross-val folds
                    reject_test_cv, _ = eval_c2st_fn(
                        metrics=metrics_cv,
                        conf_alpha=alpha,
                        P=P,
                        Q=Q,
                        null_samples_list=null_samples_list,
                        n_trials_null=N_TRIALS_NULL,
                        cross_val=True,
                        n_folds=2,
                    )
                    # update the empirical power at alpha for each cv-metric
                    for m in metrics_cv:
                        emp_power_a[m] += reject_test_cv[m] / n_runs

                    if compute_FPR:
                        # evaluate test under (H0) over several cross-val folds
                        reject_test_FPR_cv, _ = eval_c2st_fn(
                            metrics=metrics_cv,
                            conf_alpha=alpha,
                            P=P,
                            Q=Q_FPR,
                            null_samples_list=null_samples_list,
                            n_trials_null=N_TRIALS_NULL,
                            cross_val=True,
                            n_folds=2,
                        )
                        # update the FPR at alpha for each cv-metric
                        for m in metrics_cv:
                            FPR_a[m] += reject_test_FPR_cv[m] / n_runs

            print("Result: ", emp_power_a)
            print()

            # append the empirical power at alpha for each metric
            for m in all_metrics:
                emp_power[m].append(emp_power_a[m])
                FPR[m].append(FPR_a[m])

    return emp_power, FPR


if __name__ == "__main__":
    from valdiags.test_utils import eval_htest
    from valdiags.vanillaC2ST import t_stats_c2st

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # metrics / test statistics
    metrics = ["accuracy", "div", "mse"]
    metrics_cv = ["accuracy_cv", "div_cv", "mse_cv"]

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

    # compute empirical power for each metric with `eval_c2st_lda``
    emp_power, FPR = empirical_power_c2st_mean_shift(
        eval_c2st_fn=eval_c2st_lda,
        metrics=metrics,
        metrics_cv=metrics_cv,
        compute_FPR=True,
    )

    # plot empirical power for each metric
    for m in metrics + metrics_cv:
        plt.plot(ALPHA_LIST, emp_power[m], label=str(m), marker="o")
    plt.legend()
    # save plot
    plt.savefig(
        f"emp_power_lqda_dim_{DIM}_nruns_{N_RUNS}_single_class_{single_class_eval}_insample_{in_sample}.pdf"
    )
    plt.show()

    # plot FPR for each metric
    for m in metrics + metrics_cv:
        plt.plot(ALPHA_LIST, FPR[m], label=str(m), marker="o")
    plt.legend()
    # save plot
    plt.savefig(
        # f"emp_power_lqda_dim_{DIM}_nruns_{N_RUNS}_single_class_{single_class_eval}_permutation_{use_permutation}.pdf"
        f"FPR_lqda_dim_{DIM}_nruns_{N_RUNS}_single_class_{single_class_eval}_insample_{in_sample}.pdf"
    )
    plt.show()

    # plot ROC/AUC for each metric
    for m in metrics + metrics_cv:
        plt.plot(FPR[m], emp_power[m], label=str(m), marker="o")
    plt.legend()
    # save plot
    plt.savefig(
        # f"emp_power_lqda_dim_{DIM}_nruns_{N_RUNS}_single_class_{single_class_eval}_permutation_{use_permutation}.pdf"
        f"ROC_lqda_dim_{DIM}_nruns_{N_RUNS}_single_class_{single_class_eval}_insample_{in_sample}.pdf"
    )
    plt.show()

