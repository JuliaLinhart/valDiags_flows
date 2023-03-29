# Reproduce experiments from [Lee et al. 2018](https://arxiv.org/abs/1805.12114) ~ Figure 2;
#
# Empirical power of LDA-based C2ST test between two multivariate Gaussian distributions
# with different means and equal variances:
# P ~ N(0, I_d) and Q ~ N(mu, I_d) with mu = sqrt(0.05)/sqrt(0.01) and d = 5/20.

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from c2st_p_values_roc import c2st_p_values_tfpr

# Globals

# experiment path to save figures
PATH_EXPERIMENT = "saved_experiments/c2st_evaluation/lee_2018/"

# data parameters
N_SAMPLES = 100  # sample sizes for P and Q ('n0 = n1' in [Lee et al. 2018])

# test parameters
N_RUNS = (
    300  # number of test runs (not sure if this is the same as in [Lee et al. 2018])
)
N_TRIALS_NULL = 100  # number of times we compute test statistic under the null (number of permutations 'B' in [Lee et al. 2018])
ALPHA_LIST = np.linspace(0, 1, 20)  # significance levels to evaluate the test at

# use permutation method to estimate the null distribution
PERMUATION = True

# classifier parameters
CLF_NAME = "LDA"  # name of the classifier
CLF_CLASS = LinearDiscriminantAnalysis  # classifier class
CLF_KWARGS = {
    "solver": "eigen",
    "priors": [0.5, 0.5],
}  # classifier kwargs (not sure if this is the same as in [Lee et al. 2018])

# classifier evaluation parameters
IN_SAMPLE = True  # evaluate on training data
SINGLE_CLASS = False  # evaluate on data from both classes
CV_FOLDS = 2  # number of cross-validation folds

# metrics / test statistics that are evaluated in [Lee et al. 2018]
METRICS = ["accuracy", "mse"]
METRICS_CV = ["accuracy_cv"]

if __name__ == "__main__":
    import argparse
    import os

    from functools import partial
    import matplotlib.pyplot as plt

    from valdiags.test_utils import eval_htest
    from valdiags.vanillaC2ST import t_stats_c2st

    from scipy.stats import multivariate_normal as mvn

    # parse arguments
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument(
        "--dim",
        type=int,
        default=5,
        help="Dimension of the data (number of features).",
    )
    parser.add_argument(
        "--mean_shift",
        "-mu",
        type=float,
        default=np.sqrt(0.05),
        help="Mean shift between P and Q.",
    )
    parser.add_argument(
        "--plot_pvalues",
        "-pv",
        action="store_true",
        help="Plot p-values for each metric.",
    )
    args = parser.parse_args()

    # define distributions P and Q
    dim = args.dim  # data dimension
    mu = args.mean_shift  # mean shift between P and Q
    P_dist = mvn(mean=np.zeros(dim), cov=np.eye(dim))
    Q_dist = mvn(mean=np.array([mu] * dim), cov=np.eye(dim))

    # define function to evaluate the test
    eval_c2st = partial(
        eval_htest,
        t_stats_estimator=t_stats_c2st,
        n_trials_null=N_TRIALS_NULL,
        clf_class=CLF_CLASS,
        clf_kwargs=CLF_KWARGS,
        in_sample=IN_SAMPLE,
        single_class_eval=SINGLE_CLASS,
        use_permutation=PERMUATION,
        verbose=False,
    )

    # compute p_value at alpha=0.05, for each metric with `eval_c2st_lda`
    TPR, FPR, p_values_H1, p_values_H0 = c2st_p_values_tfpr(
        eval_c2st_fn=eval_c2st,
        n_runs=N_RUNS,
        n_samples=N_SAMPLES,
        alpha_list=ALPHA_LIST,
        P_dist=P_dist,
        Q_dist=Q_dist,
        metrics=METRICS,
        metrics_cv=METRICS_CV,
        n_folds=CV_FOLDS,
        scores_null=None,
        use_permutation=PERMUATION,
    )

    # plot p-values for each metric
    if args.plot_pvalues:
        for m in METRICS + METRICS_CV:
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
                label=f"H1 (mu={np.round(mu,2)})",
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
            plt.title(f"C2ST-{m}, in-sample eval (N={N_SAMPLES}, dim={dim})")
            # plt.title(f"C2ST-{m}, single_class/out-of-sample (N={N_SAMPLES}, dim={dim})")
            plt.savefig(PATH_EXPERIMENT + f"p_values_mu_{np.round(mu,2)}_dim_{dim}.pdf")
            plt.show()

    # plot TPR for each metric
    for m in METRICS + METRICS_CV:
        plt.plot(ALPHA_LIST, TPR[m], label=m)
    plt.legend()
    plt.title(
        f"TPR for C2ST, (H1): mu={np.round(mu,2)}, in-sample eval (N={N_SAMPLES}, dim={dim})"
        # f"TPR for C2ST, (H1): mu={np.round(mu,2)}, single_class/out-of-sample (N={N_SAMPLES}, dim={dim})"
    )
    plt.savefig(PATH_EXPERIMENT + f"tpr_mu_{np.round(mu,2)}_dim_{dim}.pdf")
    plt.show()

    # plot FPR for each metric
    for m in METRICS + METRICS_CV:
        plt.plot(ALPHA_LIST, FPR[m], label=m)
    plt.legend()
    plt.title(
        f"FPR for C2ST, (H1): mu={np.round(mu,2)}, in-sample eval (N={N_SAMPLES}, dim={dim})"
        # f"FPR for C2ST, (H1): mu={np.round(mu,2)}, single_class/out-of-sample (N={N_SAMPLES}, dim={dim})"
    )
    plt.savefig(PATH_EXPERIMENT + f"fpr_mu_{np.round(mu,2)}_dim_{dim}.pdf")
    plt.show()

    # roc curve
    for m in METRICS + METRICS_CV:
        plt.plot(FPR[m], TPR[m], label=m)
    plt.legend()
    plt.title(
        f"ROC for C2ST, (H1): mu={np.round(mu,2)}, in-sample eval (N={N_SAMPLES}, dim={dim})"
        # f"ROC for C2ST, (H1): mu={np.round(mu,2)}, single_class/out-of-sample (N={N_SAMPLES}, dim={dim})"
    )
    plt.savefig(PATH_EXPERIMENT + f"roc_mu_{np.round(mu,2)}_dim_{dim}.pdf")
    plt.show()

