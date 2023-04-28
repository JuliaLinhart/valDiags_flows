import argparse
import os
from functools import partial
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn
from scipy.stats import t

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from valdiags.vanillaC2ST import c2st_scores, t_stats_c2st

from classifiers.optimal_bayes import (
    opt_bayes_scores,
    AnalyticGaussianLQDA,
    AnalyticStudentClassifier,
)

from c2st_p_values_roc import c2st_p_values_tfpr
from valdiags.test_utils import eval_htest

# GLOBAL PARAMETERS
PATH_EXPERIMENT = "saved_experiments/neurips_2023/exp_1/"

# data parameters
N_SAMPLES_EVAL = 10_000  # N_v (validation set size - used to compute the test statistics for a trained classifier)

# test parameters
ALPHA = 0.05
N_RUNS = 500

N_TRIALS_NULL = 1000
USE_PERMUTATION = False

# metrics / test statistics
METRICS = {
    "accuracy": [r"$\hat{t}_{Acc0}$", r"$\hat{t}_{Acc}$"],
    "mse": [r"$\hat{t}_{Reg0}$", r"$\hat{t}_{Reg}$"],
    "div": [r"$\hat{t}_{Max0}$", r"$\hat{t}_{Max}$"],
}

# plot parameters
test_stat_names = [item for sublist in list(METRICS.values()) for item in sublist]
colors = ["red", "red", "blue", "blue", "orange", "orange"]

# Parse arguments
parser = argparse.ArgumentParser()

# data parameters
parser.add_argument(
    "--dim",
    type=int,
    default=2,
    help="Dimension of the data (number of features).",
)

parser.add_argument(
    "--q_dist",
    "-q",
    type=str,
    default=None,
    choices=["mean", "scale", "df"],
    help="Variable parameter in the distribution of Q.",
)

# whether to use the optimal bayes classifier or not
parser.add_argument(
    "--opt_bayes",
    action="store_true",
    help="Plot Test Statistics for Optimal Bayes Classifier over distribution shifts.",
)

# experiment parameters

parser.add_argument(
    "--t_shift",
    action="store_true",
    help="Compute and Plot Test Statistics for the test over distribution shifts.",
)

parser.add_argument(
    "--err_ns",
    action="store_true",
    help="Compute and Plot Type 1 error/Power for the test over multiple sample sizes.",
)

parser.add_argument(
    "--err_shift",
    action="store_true",
    help="Compute and Plot Type 1 error/Power for the test over distribution shifts.",
)

args = parser.parse_args()

# ==== EXPERIMENT SETUP ====
# Global experiment parameters
dim = args.dim  # dimension of the data

# P - class 0 distribution: standard Gaussian (fixed)
P_dist = mvn(mean=np.zeros(dim), cov=np.eye(dim))  # P is a standard Gaussian

# Parameters for different experiments
if args.q_dist == "mean":
    # variable sample size or distribution shift
    if args.err_ns:
        # N_cal (training set size)
        n_samples_list = [50, 75, 100, 150, 200]
        # mean-shift
        shifts = [np.sqrt(0.05)]
    elif args.err_shift or args.t_shift:
        # N_cal (training set size)
        n_samples_list = [3000]
        # mean-shift
        shifts = np.concatenate(
            [
                [-1, -0.5, -0.3],
                np.arange(-0.1, 0.0, 0.02),
                [0.0],
                np.arange(0.02, 0.12),
                [0.3, 0.5, 1],
            ]
        )
    else:
        raise NotImplementedError

    # H_0 label
    h0_label = r"$\mathcal{H}_0: \mathcal{N}(0, I) = \mathcal{N}(m, I)$"

    # Q - class 1 distrribution: reduced Gaussian with shifted mean (variable)
    Q_dist_list = [mvn(mean=np.array([mu] * dim), cov=np.eye(dim)) for mu in shifts]

    # classifier
    if args.opt_bayes:
        clf_name = "OptBayes"
        clf_list = [AnalyticGaussianLQDA(dim=dim, mu=mu) for mu in shifts]
    else:
        clf_name = "LDA"
        clf_class = LinearDiscriminantAnalysis
        clf_kwargs = {"solver": "eigen", "priors": [0.5, 0.5]}

elif args.q_dist == "scale":
    # variable sample size or distribution shift
    if args.err_ns:
        # N_cal (training set size)
        n_samples_list = [50, 100, 200, 500, 1000, 2000, 3000, 5000]
        # scale-shift
        shifts = [1.3]
    elif args.err_shift or args.t_shift:
        # N_cal (training set size)
        n_samples_list = [2000]
        # scale-shift
        shifts = np.concatenate([[0.01], np.arange(0.1, 1.6, 0.1)])
    else:
        raise NotImplementedError

    # H_0 label
    h0_label = r"$\mathcal{H}_0: \mathcal{N}(0, I) = \mathcal{N}(0,s\times I)$"

    # Q - class 1 distrribution: centered Gaussian with shifted scale (variable)
    Q_dist_list = [mvn(mean=np.zeros(dim), cov=s * np.eye(dim)) for s in shifts]

    # classifier
    if args.opt_bayes:
        clf_name = "OptBayes"
        clf_list = [AnalyticGaussianLQDA(dim=dim, sigma=s) for s in shifts]
    else:
        clf_name = "QDA"
        clf_class = QuadraticDiscriminantAnalysis
        clf_kwargs = {"priors": [0.5, 0.5]}

elif args.q_dist == "df":
    # variable sample size or distribution shift
    if args.err_ns:
        # N_cal (training set size)
        n_samples_list = [50, 100, 200, 500, 1000, 2000, 3000, 5000]
        # df-shift
        shifts = [3]
    elif args.err_shift or args.t_shift:
        # N_cal (training set size)
        n_samples_list = [2000]
        # df-shift
        shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35]
    else:
        raise NotImplementedError

    # H_0 label
    h0_label = r"$\mathcal{H}_0: \mathcal{N}(0, I) = t(df)$"

    # Q - class 1 distrribution: standard Student with shifted degrees of freedom (variable)
    Q_dist_list = [t(df=df, loc=0, scale=1) for df in shifts]

    # classifier
    if args.opt_bayes:
        clf_name = "OptBayes"
        clf_list = [AnalyticStudentClassifier(df=df) for df in shifts]
    else:
        clf_name = "MLP"
        clf_class = MLPClassifier
        clf_kwargs = {"alpha": 0, "max_iter": 25000}
else:
    raise NotImplementedError

# ==== EXP 1: T_STATS UNDER DISTRIBUTION SHIFT ====
if args.t_shift:
    test_stats = dict(zip(test_stat_names, [[] for _ in test_stat_names]))
    for r in tqdm(range(10), desc="Runs"):
        test_stats_r = dict(zip(test_stat_names, [[] for _ in test_stat_names]))
        P_eval = P_dist.rvs(size=N_SAMPLES_EVAL)
        # evaluate test statistics
        for i, (s, Q_dist) in enumerate(zip(shifts, Q_dist_list)):
            for b in [True, False]:
                Q_eval = Q_dist.rvs(size=N_SAMPLES_EVAL)
                if args.opt_bayes:
                    scores = opt_bayes_scores(
                        P=P_eval,
                        Q=Q_eval,
                        clf=clf_list[i],
                        metrics=list(METRICS.keys()),
                        single_class_eval=b,
                    )
                else:
                    P = P_dist.rvs(size=n_samples_list[0])
                    Q = Q_dist.rvs(size=n_samples_list[0])
                    if dim == 1:
                        P = P.reshape(-1, 1)
                        Q = Q.reshape(-1, 1)
                        P_eval = P_eval.reshape(-1, 1)
                        Q_eval = Q_eval.reshape(-1, 1)
                    scores = t_stats_c2st(
                        P=P,
                        Q=Q,
                        cross_val=False,
                        P_eval=P_eval,
                        Q_eval=Q_eval,
                        clf_class=clf_class,
                        clf_kwargs=clf_kwargs,
                        metrics=list(METRICS.keys()),
                        single_class_eval=b,
                        null_hypothesis=False,
                    )

                for metric, t_names in zip(METRICS.keys(), METRICS.values()):
                    if b:
                        name = t_names[0]
                    else:
                        name = t_names[1]

                    if metric == "mse":
                        test_stats_r[name].append(scores[metric] + 0.5)
                    else:
                        test_stats_r[name].append(scores[metric])
        for name in test_stat_names:
            test_stats[name].append(test_stats_r[name])

    test_stats_mean = {k: np.mean(v, axis=0) for k, v in test_stats.items()}
    test_stats_std = {k: np.std(v, axis=0) for k, v in test_stats.items()}

    # plot and save results
    plt.plot(
        shifts,
        [0.5] * len(shifts),
        color="grey",
        linestyle="--",
        label=r"$\mathcal{H}_0$",
    )
    for name, color in zip(test_stat_names, colors):
        linestyle = "-"
        if "0" not in name:
            linestyle = "--"
        plt.plot(
            shifts,
            test_stats_mean[name],
            label=name,
            color=color,
            linestyle=linestyle,
        )
        plt.fill_between(
            x=shifts,
            y1=test_stats_mean[name] - test_stats_std[name],
            y2=test_stats_mean[name] + test_stats_std[name],
            alpha=0.2,
            color=color,
        )
    plt.xlabel(f"{args.q_dist} shift")
    plt.ylabel(r"$\hat{t}$ (test statistic)")
    plt.legend(loc="upper right")
    plt.title(f"{clf_name} classifier for {h0_label}" + f"\n dim={dim}")
    plt.savefig(
        PATH_EXPERIMENT + f"t_stats_{args.q_dist}_shift_{clf_name}_dim_{dim}.pdf"
    )
    plt.show()

# ==== Prepare for EXPS 2 and 3 ====
if args.err_shift or args.err_ns:
    # Initialize test statistics function
    t_stats_c2st_custom = partial(
        t_stats_c2st,
        scores_fn=c2st_scores,
        metrics=list(METRICS.keys()),
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        # args for scores_fn
        cross_val=False,
    )

    # Pre-compute test statistics under the null distribution
    scores_null_list = []
    for n in n_samples_list:
        print()
        print(f"N_cal = {n}")
        if not USE_PERMUTATION:
            # Not using the permutation method to simulate the null distribution
            # Using data from P to compute the scores/test statistics under the true null distribution
            print()
            print(
                "Pre-computing or loading the test statistics under the null distribution."
                + "\n They will be reused at every test-run. The permutation method is not needed."
            )
            print()
            scores_null = dict(zip([True, False], [[], []]))

            filename = f"nt_{N_TRIALS_NULL}_N_{n}_dim_{dim}_{clf_name}.npy"
            if os.path.exists(PATH_EXPERIMENT + "t_stats_null/" + filename):
                # load null scores if they exist
                scores_null = np.load(
                    PATH_EXPERIMENT + "t_stats_null/" + filename,
                    allow_pickle=True,
                ).item()
            else:
                # otherwise, compute them
                # generate data from P
                list_P_null = [P_dist.rvs(n) for _ in range(2 * N_TRIALS_NULL)]
                list_P_eval_null = [
                    P_dist.rvs(N_SAMPLES_EVAL) for _ in range(2 * N_TRIALS_NULL)
                ]
                if dim == 1:
                    for i in range(2 * N_TRIALS_NULL):
                        list_P_null[i] = list_P_null[i].reshape(-1, 1)
                        list_P_eval_null[i] = list_P_eval_null[i].reshape(-1, 1)
                for b in [True, False]:
                    t_stats_null = t_stats_c2st_custom(
                        null_hypothesis=True,
                        n_trials_null=N_TRIALS_NULL,
                        list_P_null=list_P_null,
                        list_P_eval_null=list_P_eval_null,
                        use_permutation=False,
                        # args for scores_fn
                        single_class_eval=b,
                        # unnecessary, but needed inside `t_stats_c2st`
                        P=list_P_null[0],
                        Q=list_P_null[1],
                        P_eval=list_P_eval_null[0],
                        Q_eval=list_P_eval_null[1],
                    )
                    scores_null[b] = t_stats_null
                np.save(
                    PATH_EXPERIMENT + "t_stats_null/" + filename,
                    scores_null,
                )

        else:
            print()
            print(
                f"Not pre-computing the test-statistics under the null."
                + "\n Using the permutation method to estimate them at each test run."
            )
            print()
            scores_null = {True: None, False: None}
        scores_null_list.append(scores_null)

    # Define function to evaluate the test
    eval_c2st = partial(
        eval_htest,
        t_stats_estimator=t_stats_c2st_custom,
        verbose=False,
        metrics=list(METRICS.keys()),
    )

    # ==== EXP 2: EMPIRICAL POWER UNDER DISTRIBUTION SHIFT  ====
    if args.err_shift:
        n = n_samples_list[0]
        scores_null = scores_null_list[0]

        # compute TPR and FPR for each shift
        TPR_list, p_values_H1_list = (
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
        )
        for i, (s, Q_dist) in enumerate(zip(shifts, Q_dist_list)):
            print()
            print(f"{args.q_dist} shift: {np.round(s,2)}")
            print()
            for b in [True, False]:
                TPR, _, p_values_H1, _ = c2st_p_values_tfpr(
                    eval_c2st_fn=partial(
                        eval_c2st, single_class_eval=b, n_trials_null=N_TRIALS_NULL
                    ),
                    n_runs=N_RUNS,
                    n_samples={"train": n, "eval": N_SAMPLES_EVAL},
                    alpha_list=[ALPHA],
                    P_dist=P_dist,
                    Q_dist=Q_dist,
                    metrics=list(METRICS.keys()),
                    metrics_cv=[],
                    scores_null={
                        False: scores_null[b],
                        True: None,
                    },  # no cross val metrics
                    compute_FPR=False,
                )
                for metric, t_names in zip(METRICS.keys(), METRICS.values()):
                    if b:
                        name = t_names[0]
                    else:
                        name = t_names[1]
                    TPR_list[name].append(TPR[metric][0])
                    p_values_H1_list[name].append(p_values_H1[metric][0])

        # plot TPR for each metric as a function of shift
        for name, color in zip(test_stat_names, colors):
            linestyle = "-"
            if "0" not in name:
                linestyle = "--"
            plt.plot(
                shifts,
                TPR_list[name],
                label=name,
                color=color,
                linestyle=linestyle,
                alpha=0.8,
            )
        plt.xlabel(f"{args.q_dist} shift")
        plt.ylabel(r"Power (TPR)")
        plt.legend()
        plt.title(
            f"{clf_name}-C2ST Power for {h0_label}, dim={dim}" + "\n alpha = {ALPHA}"
        )
        plt.savefig(
            PATH_EXPERIMENT
            + f"power_{args.q_dist}_shifts_{clf_name}_alpha_{ALPHA}_N_{n}_dim_{dim}.pdf"
        )
        plt.show()

    # ==== EXP 3: Empirical Power and Type I error as a function of train size (N_cal) ====
    if args.err_ns:
        shift = shifts[0]
        Q_dist = Q_dist_list[0]
        # For each sample size, compute the test results for each metric:
        # p-values, TPR and FPR (at given alphas)
        TPR_list, FPR_list, p_values_H0_list, p_values_H1_list = (
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
            dict(zip(test_stat_names, [[] for _ in test_stat_names])),
        )
        for i, n in enumerate(n_samples_list):
            print()
            print(f"N_cal = {n}")
            print()
            for b in [True, False]:
                TPR, FPR, p_values_H1, p_values_H0 = c2st_p_values_tfpr(
                    eval_c2st_fn=partial(
                        eval_c2st, single_class_eval=b, n_trials_null=N_TRIALS_NULL
                    ),
                    n_runs=N_RUNS,
                    n_samples={"train": n, "eval": N_SAMPLES_EVAL},
                    alpha_list=[ALPHA],
                    P_dist=P_dist,
                    Q_dist=Q_dist,
                    metrics=list(METRICS.keys()),
                    metrics_cv=[],
                    scores_null={
                        False: scores_null_list[i][b],
                        True: None,
                    },  # no cross val metrics
                )

                for metric, t_names in zip(METRICS.keys(), METRICS.values()):
                    if b:
                        name = t_names[0]
                    else:
                        name = t_names[1]
                    TPR_list[name].append(TPR[metric][0])
                    FPR_list[name].append(FPR[metric][0])
                    p_values_H1_list[name].append(p_values_H1[metric][0])
                    p_values_H0_list[name].append(p_values_H0[metric][0])

        # plot TPR and FPR for each metric as a function of N_cal
        # FPR
        for name, color in zip(test_stat_names, colors):
            linestyle = "-"
            if "0" not in name:
                linestyle = "--"
            plt.plot(
                n_samples_list,
                FPR_list[name],
                label=name,
                color=color,
                linestyle=linestyle,
                alpha=0.8,
            )
        plt.ylabel(r"Type I error (FPR)")
        plt.xlabel(r"$N_{cal}$")
        plt.legend()
        plt.title(f"{clf_name}-C2ST Type I error, dim={dim}" + f"\n alpha = {ALPHA}")
        plt.savefig(
            PATH_EXPERIMENT + f"type_I_error_ns_{clf_name}_alpha_{ALPHA}_dim_{dim}.pdf"
        )
        plt.show()

        # TPR
        for name, color in zip(test_stat_names, colors):
            linestyle = "-"
            if "0" not in name:
                linestyle = "--"
            plt.plot(
                n_samples_list,
                TPR_list[name],
                label=name,
                color=color,
                linestyle=linestyle,
                alpha=0.8,
            )

        plt.ylabel(r"Power (TPR)")
        plt.xlabel(r"$N_{cal}$")
        plt.legend()
        plt.title(
            f"{clf_name}-C2ST Power for {h0_label}, s = {shift}, dim={dim}"
            + "\n alpha = {ALPHA}"
        )
        plt.savefig(
            PATH_EXPERIMENT
            + f"power_ns_{clf_name}_alpha_{ALPHA}_{args.q_dist}_shift_s_{shift}_dim_{dim}.pdf"
        )
        plt.show()
