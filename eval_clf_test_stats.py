""" Empirical power analysis as in [Lee et al. (2018)](https://arxiv.org/abs/1812.08927) ~cf. Fig.2"""

import numpy as np
from scipy.stats import multivariate_normal as mvn

from valdiags.test_utils import eval_htest

import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

N_SAMPLES = 100
N_TRIALS_NULL = 100
N_RUNS = 300
N_ALPHA = 20

DIM = 5
mu = np.sqrt(0.05)
# DIM = 20
# mu = np.sqrt(0.01)


def empirical_power_c2st_mean_shift(
    eval_c2st_fn, metrics, metrics_cv, n_runs=N_RUNS, n_alpha=N_ALPHA,
):
    """Computes the empirical power of the test as the True Positive Rate (TPR), 
    i.e. how many times we (correctly) reject the null hypothesis (H0) over several 
    test runs performed under the alternative hypothesis (H1) and over a range of 
    possible significance levels alpha in (0,1).

    The considered test is a Classifier Two Sample Test (C2ST) between (multivariate) 
    Gaussians with different mean P ~ N(0,1) and Q ~ N(mu,1):

                        (H0): P=Q vs. (H1): P \neq Q

    The TPR here corresponds to the number of times we reject (H0) for a test statistic 
    (or metric) that was estimated on data from P and Q (under (H1)).

    -----------
    inputs:
    - eval_c2st_fn: function 
        built on `eval_htest` with additional inputs:
            * samples from P, Q 
            * a list of samples from P to compute the test statistic under (H0)
    - metrics: list of str
        contains the names of the metrics that are used in the scoring function 
        of the test (as in vanillaC2ST.c2st_scores) (= test statistics) 
    - metrics_cv: list of str
        subset of metrics we want to compute over several cross-val folds
    - n_runs: int
        number of test runs to compute TPR for a given significance level alpha
    - n_alpha: int
        number of significance levels alpha in (0,1)
        --> alpha in `np.linspace(0,1, n_alpha)`
    
    -----------
    returns:
    - power: dict 
        contains the TPR of each metric, computed for every alpha   
            * keys: metrics+metrics_cv
            * values: list of TPRs computed for each alpha
    
    """
    all_metrics = metrics + metrics_cv
    print(all_metrics)
    emp_power = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

    for i, alpha in enumerate(np.linspace(0, 1, n_alpha)):

        print(f"Significance level alpha = {alpha} ({i+1}/{n_alpha})")

        power_a = dict(zip(all_metrics, [0] * len(all_metrics)))
        if alpha == 0:
            for m in all_metrics:
                emp_power[m].append(0)
        else:
            for _ in tqdm(range(n_runs), desc="Empirical error / success rate"):

                ref_samples = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
                shift_samples = mvn(mean=np.array([mu] * DIM), cov=np.eye(DIM)).rvs(
                    N_SAMPLES
                )
                null_samples_list = [
                    mvn(mean=np.array([0] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
                    for _ in range(N_TRIALS_NULL)
                ]

                success_rate = eval_c2st_fn(
                    metrics=metrics,
                    conf_alpha=alpha,
                    P=ref_samples,
                    Q=shift_samples,
                    null_samples_list=null_samples_list,
                    cross_val=False,
                )
                for m in metrics:
                    power_a[m] += success_rate[m] / n_runs

                success_rate_cv = eval_c2st_fn(
                    metrics=metrics_cv,
                    conf_alpha=alpha,
                    P=ref_samples,
                    Q=shift_samples,
                    null_samples_list=null_samples_list,
                    cross_val=True,
                    n_folds=2,
                )

                for m in metrics_cv:
                    power_a[m] += success_rate_cv[m] / n_runs

            print("Result: ", power_a)
            print()

            for m in all_metrics:
                emp_power[m].append(power_a[m])

    return power


if __name__ == "__main__":
    from valdiags.vanillaC2ST import t_stats_c2st
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    metrics = ["accuracy", "div", "mse"]
    metrics_cv = ["accuracy_cv"]

    # ESTIMATED LDA
    eval_c2st_lda = partial(
        eval_htest,
        t_stats_estimator=t_stats_c2st,
        clf_class=LinearDiscriminantAnalysis,
        clf_kwargs={},
        single_class_eval=True,
        verbose=False,
    )

    power = empirical_power_c2st_mean_shift(
        eval_htest_fn=eval_c2st_lda, metrics=metrics, metrics_cv=metrics_cv
    )

    for m in metrics + metrics_cv:
        plt.plot(np.linspace(0, 1, N_ALPHA), power[m], label=str(m), marker="o")
    plt.legend()
    plt.savefig(f"emp_power_lqda_single_class_dim_{DIM}.pdf")
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
