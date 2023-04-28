# ==== Compare different validation methods (2 sample tests - 2ST) on SBIBM tasks (toy-examples) ==== #
## Toy Examples:
# - two moons (D=2)
# - gaussian linear uniform (D=10)
#
# Validation methods to compare:
# REFERENCE for toy-examples when the true posterior is known (not amortized)
#   - C2ST / C2ST_NF (vanilla) - precompute or permutation method (not analytic because depends on classifier)
#   - C2ST-Reg / C2ST-Reg_NF - precompute or permutation method
#   - HPD-2ST (Highest Posterior Density Regions) (analytically known)
# OUR METHOD: when the true posterior is not known (amortized and single-class-eval)
#   - LC2ST / LC2ST_NF (with permutation / pre-computed)
#   - LC2ST-Reg / LC2ST-Reg_NF (with permutation / pre-computed)
#   - Local HPD-2ST [Zhao et al. 2018] (how null?)
# Experiments to evaluate / compare the methods (on average over all observations x_0 from sbibm tasks):
#   - exp 1: test stats as a function of N_train
#   - exp 2: power / type 1 error / runtime as a function of the number of N_cal (at fixed N_train)
#   - exp 3: power as a function of N_train (at fixed N_cal)

import argparse
from pathlib import Path
import os
from functools import partial
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import sbibm
from lc2st_sbibm_experiments import (
    compute_emp_power_l_c2st,
    l_c2st_results_n_train,
)

from valdiags.vanillaC2ST import t_stats_c2st, sbibm_clf_kwargs
from valdiags.localC2ST import t_stats_lc2st, train_lc2st

from valdiags.test_utils import eval_htest

from scipy.stats import multivariate_normal as mvn

# # set seed
# seed = 42

# GLOBAL PARAMETERS
PATH_EXPERIMENT = Path("saved_experiments/neurips_2023/exp_2")

# number of training samples for the estimator (NF)
N_TRAIN_LIST = [
    100,
    # 215,
    # 464,
    1000,
    # 2151,
    # 4641,
    10000,
    # 21544,
    # 46415,
    100000,
]  # np.logspace(2,5,10, dtpye=int)

# numbers of the observations x_0 from sbibm to evaluate the tests at
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# test parameters
ALPHA = 0.05
N_RUNS = 100
N_TRIALS_PRECOMPUTE = 1000

# metrics / test statistics
ALL_METRICS = ["accuracy", "mse", "div"]

# classifier parameters
CROSS_VAL = False
N_ENSEMBLE = 1

# Parse arguments
parser = argparse.ArgumentParser()

# experiment parameters
parser.add_argument(
    "--task",
    type=str,
    default="two_moons",
    choices=["two_moons", "gaussian_mixture", "gaussian_linear_uniform"],
    help="Task from sbibm to perform the experiment on.",
)

parser.add_argument(
    "--t_stat_nt",
    action="store_true",
    help="Exp 1: Plot the test statistic as a function of N_train.",
)

# data parameters
parser.add_argument(
    "--n_cal",
    nargs="+",
    type=int,
    default=[10000],  # [50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
    help="Number of calibration samples to train (L)-C2ST on. Can be a list of integers.",
)

parser.add_argument(
    "--n_eval",
    type=int,
    default=10000,
    help="Number of evaluation samples for (L)-C2ST.",
)

# test parameters

# null distribution parameters
parser.add_argument(
    "--n_trials_null",
    "-nt",
    type=int,
    default=100,
    help="Number of trials to estimate the distribution of the test statistic under the null.",
)

print("==== VALIDATION METHOD COMPARISON for sbibm-tasks ====")
print()

# Parse arguments
args = parser.parse_args()
# define task and path
task = sbibm.get_task(args.task)
task_path = PATH_EXPERIMENT / args.task

# ==== sbi set-up for given task ==== #
# prior, simulator, inference algorithm
prior = task.get_prior()
simulator = task.get_simulator()
algorithm = "npe"
if algorithm != "npe":
    raise NotImplementedError("Only NPE is supported for now.")
print(f"Task: {args.task} / Algorithm: {algorithm}")
print()

# load observations
print(f"Loading observations {NUM_OBSERVATION_LIST}")
observation_list = [
    task.get_observation(num_observation=n_obs) for n_obs in NUM_OBSERVATION_LIST
]
print()
observation_dict = dict(zip(NUM_OBSERVATION_LIST, observation_list))

# ==== test set-up ==== #
# dataset sizes
N_cal_list = args.n_cal
N_eval = args.n_eval
dim_theta = prior(num_samples=1).shape[-1]
dim_x = simulator(prior(num_samples=1)).shape[-1]

# classifier parameters
c2st_clf_kwargs = sbibm_clf_kwargs(ndim=dim_theta)
lc2st_clf_kwargs = sbibm_clf_kwargs(ndim=dim_theta + dim_x)

# set-up path-params to save results for given test params
test_params = f"alpha_{ALPHA}_n_runs_{N_RUNS}_n_trials_null_{args.n_trials_null}"
eval_params = f"n_eval_{N_eval}_n_ensemble_{N_ENSEMBLE}_cross_val_{CROSS_VAL}"

# pre-compute / load test statistics for the null hypothesis
# they are independant of the estimator and the observation space (x)
for N_cal in N_cal_list:
    if not os.path.exists(task_path / "t_stats_null" / eval_params):
        os.makedirs(task_path / "t_stats_null" / eval_params)
    try:
        lc2st_stats_null = torch.load(
            task_path
            / "t_stats_null"
            / eval_params
            / f"lc2st_stats_null_nt_{N_TRIALS_PRECOMPUTE}_n_cal_{N_cal}.pkl"
        )
        c2st_stats_null = torch.load(
            task_path
            / "t_stats_null"
            / eval_params
            / f"c2st_stats_null_nt_{N_TRIALS_PRECOMPUTE}_n_cal_{N_cal}.pkl"
        )
    except FileNotFoundError:
        print(
            f"Pre-compute test statistics for (NF)-H_0 (N_cal={N_cal}, n_trials={N_TRIALS_PRECOMPUTE})"
        )
        P_dist_null = mvn(mean=torch.zeros(dim_theta), cov=torch.eye(dim_theta))
        list_P_null = [P_dist_null.rvs(N_cal) for _ in range(2 * N_TRIALS_PRECOMPUTE)]
        list_P_eval_null = [
            P_dist_null.rvs(N_cal) for _ in range(2 * N_TRIALS_PRECOMPUTE)
        ]

        lc2st_stats_null = t_stats_c2st(
            null_hypothesis=True,
            metrics=ALL_METRICS,  # proposed test statistics
            use_permutation=False,
            n_trials_null=N_TRIALS_PRECOMPUTE,  # we can use a lot because we pre-compute
            return_probas=False,
            # required kwargs for t_stats_c2st
            P=list_P_null[0],
            Q=list_P_null[1],
            # kwargs for c2st_scores
            cross_val=CROSS_VAL,
            n_ensemble=N_ENSEMBLE,
            single_class_eval=True,
            clf_kwargs=c2st_clf_kwargs,
            list_P_null=list_P_null,
            list_P_eval_null=list_P_eval_null,
        )
        c2st_stats_null = t_stats_c2st(
            null_hypothesis=True,
            metrics=ALL_METRICS,  # proposed test statistics
            use_permutation=False,
            n_trials_null=N_TRIALS_PRECOMPUTE,  # we can use a lot because we pre-compute
            return_probas=False,
            # required kwargs for t_stats_c2st
            P=list_P_null[0],
            Q=list_P_null[1],
            # kwargs for c2st_scores
            cross_val=CROSS_VAL,
            n_ensemble=N_ENSEMBLE,
            single_class_eval=False,
            clf_kwargs=c2st_clf_kwargs,
            list_P_null=list_P_null,
            list_P_eval_null=list_P_eval_null,
        )
        torch.save(
            lc2st_stats_null,
            task_path
            / "t_stats_null"
            / eval_params
            / f"lc2st_stats_null_nt_{N_TRIALS_PRECOMPUTE}_n_cal_{N_cal}.pkl",
        )
        torch.save(
            c2st_stats_null,
            task_path
            / "t_stats_null"
            / eval_params
            / f"c2st_stats_null_nt_{N_TRIALS_PRECOMPUTE}_n_cal_{N_cal}.pkl",
        )

# initialize the C2ST and L-C2ST test statistics function
t_stats_c2st_custom = partial(
    t_stats_c2st,
    n_trials_null=args.n_trials_null,
    # kwargs for c2st_scores
    cross_val=CROSS_VAL,
    n_ensemble=N_ENSEMBLE,
    clf_kwargs=c2st_clf_kwargs,
)

t_stats_lc2st_custom = partial(
    t_stats_lc2st,
    n_trials_null=args.n_trials_null,
    return_probas=False,
    # kwargs for lc2st_scores
    cross_val=CROSS_VAL,
    n_ensemble=N_ENSEMBLE,
    single_class_eval=True,
    clf_kwargs=lc2st_clf_kwargs,
)


# perform the experiment
# ==== EXP 1: test stats as a function of N_train (N_cal = max)==== #
if args.t_stat_nt:
    print(f"Experiment 1: test statistics as a function of N_train...")
    print(f"... for N_cal = {N_cal}")
    print()

    for N_cal in N_cal_list:
        avg_results = l_c2st_results_n_train(
            task,
            n_cal=N_cal,
            n_eval=N_eval,
            observation_dict=observation_dict,
            n_train_list=N_TRAIN_LIST,
            alpha=ALPHA,
            c2st_stats_fn=t_stats_c2st_custom,
            lc2st_stats_fn=t_stats_lc2st_custom,
            c2st_stats_null_nf=c2st_stats_null,
            lc2st_stats_null_nf=lc2st_stats_null,
            task_path=task_path,
            results_n_train_path=Path("results") / test_params / eval_params,
            methods=["c2st", "lc2st", "c2st_nf", "lc2st_nf"],
        )

        # path to save figures
        fig_path = task_path / "figures"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        test_results_dict = {
            "vanilla C2ST ": {k: v["accuracy"] for k, v in avg_results["c2st"].items()},
            "Reg-C2ST": {k: v["mse"] for k, v in avg_results["c2st"].items()},
            "Reg-L-C2ST": {k: v["mse"] for k, v in avg_results["lc2st"].items()},
            "Reg-L-C2ST (NF)": {
                k: v["mse"] for k, v in avg_results["lc2st_nf"].items()
            },
            "Max-L-C2ST": {k: v["div"] for k, v in avg_results["lc2st"].items()},
            "Max-L-C2ST (NF)": {
                k: v["div"] for k, v in avg_results["lc2st_nf"].items()
            },
        }

        colors = ["grey", "blue", "orange", "orange", "red", "red"]
        linestyles = ["-", "-", "-", "--", "-", "--"]

        markers = ["o", "o", "o", "*", "o", "*"]

        for k in avg_results["c2st"].keys():
            for method, color, linestyle, marker in zip(
                test_results_dict.keys(), colors, linestyles, markers
            ):
                plt.plot(
                    np.arange(len(N_TRAIN_LIST)),
                    test_results_dict[method][k],
                    label=method,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                )
                if "std" in k:
                    k_mean = k[:-4] + "_mean"
                    plt.fill_between(
                        np.arange(len(N_TRAIN_LIST)),
                        np.array(test_results_dict[method][k_mean])
                        - np.array(test_results_dict[method][k]),
                        np.array(test_results_dict[method][k_mean])
                        + np.array(test_results_dict[method][k]),
                        alpha=0.2,
                        color=color,
                    )
            if "p_value" in k:
                plt.plot(
                    np.arange(len(N_TRAIN_LIST)),
                    np.ones(len(N_TRAIN_LIST)) * 0.05,
                    "--",
                    color="black",
                    label="alpha-level",
                )
            if "t_stat" in k:
                plt.plot(
                    np.arange(len(N_TRAIN_LIST)),
                    np.ones(len(N_TRAIN_LIST)) * 0.5,
                    "--",
                    color="black",
                    label=r"$\mathcal{H}_0$",
                )
            if "std" not in k:
                plt.legend()
                plt.xticks(np.arange(len(N_TRAIN_LIST)), N_TRAIN_LIST)
                plt.xlabel("N_train")
                plt.ylabel(k)
                plt.savefig(
                    fig_path
                    / f"{k}_ntrain_n_cal_{N_cal}_{test_params}_{eval_params}.pdf"
                )
                plt.show()
