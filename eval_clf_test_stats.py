# Empirical power analysis
# see Figure 2 in [Lee et al. (2018)](https://arxiv.org/abs/1812.08927)

import numpy as np
from scipy.stats import multivariate_normal as mvn

from valdiags.test_utils import eval_htest

import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

# GLOBALS
N_SAMPLES = 100  # number of samples from P and Q ('n0 = n1' in [Lee et al. (2018)])
N_TRIALS_NULL = 100  # number of samples of the test statistic under (H0) ('B' in [Lee et al. (2018)])
N_RUNS = 300  # number of test runs to compute the empirical power
N_ALPHA = 20  # number of significance levels alpha in (0,1)

DIM = 5  # dimension of the Gaussians
mu = np.sqrt(0.05)  # mean shift between P and Q

# DIM = 20 # dimension of the Gaussians
# mu = np.sqrt(0.01) # mean shift between P and Q


def empirical_power_c2st_mean_shift(
    eval_c2st_fn, metrics, metrics_cv, n_runs=N_RUNS, n_alpha=N_ALPHA,
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

    Returns:
        emp_power (dict): contains the TPR of each metric, computed for every alpha   
            - keys: metrics + metrics_cv
            - values: list of TPRs computed for each alpha
    """

    # combine metrics and metrics_cv
    all_metrics = metrics + metrics_cv

    # initialize dict with empty lists
    emp_power = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

    # loop over significance levels alpha
    for i, alpha in enumerate(np.linspace(0, 1, n_alpha)):

        print(f"Significance level alpha = {alpha} ({i+1}/{n_alpha})")

        # if alpha == 0, we don't need to perform the test
        if alpha == 0:
            for m in all_metrics:
                emp_power[m].append(0)

        else:
            # initialize dict with 0s
            emp_power_a = dict(zip(all_metrics, [0] * len(all_metrics)))

            # loop over test runs
            for _ in tqdm(range(n_runs), desc="Empirical error / success rate"):
                # generate samples from P and Q
                ref_samples = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
                shift_samples = mvn(mean=np.array([mu] * DIM), cov=np.eye(DIM)).rvs(
                    N_SAMPLES
                )
                # generate samples from P to compute the test statistic under (H0)
                null_samples_list = [
                    mvn(mean=np.array([0] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
                    for _ in range(N_TRIALS_NULL)
                ]
                # evaluate test under (H1)
                reject_test = eval_c2st_fn(
                    metrics=metrics,
                    conf_alpha=alpha,
                    P=ref_samples,
                    Q=shift_samples,
                    null_samples_list=null_samples_list,
                    cross_val=False,
                )
                # update the empirical power at alpha for each metric
                for m in metrics:
                    emp_power_a[m] += reject_test[m] / n_runs

                # evaluate test under (H1) over several cross-val folds
                reject_test_cv = eval_c2st_fn(
                    metrics=metrics_cv,
                    conf_alpha=alpha,
                    P=ref_samples,
                    Q=shift_samples,
                    null_samples_list=null_samples_list,
                    cross_val=True,
                    n_folds=2,
                )
                # update the empirical power at alpha for each cv-metric
                for m in metrics_cv:
                    emp_power_a[m] += reject_test_cv[m] / n_runs

            print("Result: ", emp_power_a)
            print()

            # append the empirical power at alpha for each metric
            for m in all_metrics:
                emp_power[m].append(emp_power_a[m])

    return emp_power


if __name__ == "__main__":
    from valdiags.vanillaC2ST import t_stats_c2st
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # metrics / test statistics
    metrics = ["accuracy", "div", "mse"]
    metrics_cv = ["accuracy_cv"]

    # whether to use single class evaluation
    single_class_eval = (
        True  # if True, we evaluate the classifier on samples from P only
    )

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
    emp_power = empirical_power_c2st_mean_shift(
        eval_c2st_fn=eval_c2st_lda, metrics=metrics, metrics_cv=metrics_cv
    )

    # plot empirical power for each metric
    for m in metrics + metrics_cv:
        plt.plot(np.linspace(0, 1, N_ALPHA), emp_power[m], label=str(m), marker="o")
    plt.legend()
    # save plot
    plt.savefig(f"emp_power_lqda_dim_{DIM}_single_class{single_class_eval}.pdf")
    plt.show()

    # # OPIMAL BAYES LDA
    #
    # from classifiers.optimal_bayes import AnalyticGaussianLQDA, t_stats_opt_bayes

    # eval_c2st_opt_bayes = partial(
    #     eval_htest,
    #     t_stats_estimator=t_stats_opt_bayes,
    #     clf_data = AnalyticGaussianLQDA(dim=DIM, mu=mu),
    #     clf_null = AnalyticGaussianLQDA(dim=DIM, mu=0),
    #     single_class_eval=True,
    #     verbose=False,
    # )
