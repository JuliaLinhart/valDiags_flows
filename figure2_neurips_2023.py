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

import numpy as np
import matplotlib.pyplot as plt
import torch
import sbibm
from lc2st_sbibm_experiments import (
    compute_emp_power_l_c2st,
    l_c2st_results_n_train,
    precompute_t_stats_null,
)

from valdiags.vanillaC2ST import sbibm_clf_kwargs

from scipy.stats import multivariate_normal as mvn
from sklearn.neural_network import MLPClassifier

# set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# GLOBAL PARAMETERS
PATH_EXPERIMENT = Path("saved_experiments/neurips_2023/exp_2")

# numbers of the observations x_0 from sbibm to evaluate the tests at
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# test parameters
ALPHA = 0.05
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
    choices=["two_moons", "gaussian_mixture", "gaussian_linear_uniform", "slcp"],
    help="Task from sbibm to perform the experiment on.",
)

parser.add_argument(
    "--t_res_ntrain",
    action="store_true",
    help="Exp 1: Plot the test results as a function of N_train (at fixed N_cal).",
)

parser.add_argument(
    "--power_ntrain",
    action="store_true",
    help="Exp 2: Plot the empirical power / type 1 error as a function N_train (at fixed N_cal).",
)

parser.add_argument(
    "--power_ncal",
    action="store_true",
    help="Exp 2: Plot the the empirical power / type 1 error as a function N_cal (at fixed N_train).",
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

print()
print("=================================================")
print("  VALIDATION METHOD COMPARISON for sbibm-tasks")
print("=================================================")
print()

# Parse arguments
args = parser.parse_args()
# define task and path
task = sbibm.get_task(args.task)
task_path = PATH_EXPERIMENT / args.task / "seed_lc2st_null_joint"

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
test_params = f"alpha_{ALPHA}_n_trials_null_{args.n_trials_null}"
eval_params = f"n_eval_{N_eval}_n_ensemble_{N_ENSEMBLE}_cross_val_{CROSS_VAL}"

# kwargs for c2st_scores function
kwargs_c2st = {
    "cross_val": CROSS_VAL,
    "n_ensemble": N_ENSEMBLE,
    "clf_kwargs": c2st_clf_kwargs,
}
# kwargs for lc2st_scores function
kwargs_lc2st = {
    "cross_val": CROSS_VAL,
    "n_ensemble": N_ENSEMBLE,
    "single_class_eval": True,
    "clf_kwargs": lc2st_clf_kwargs,
}

# pre-compute / load test statistics for the C2ST-NF null hypothesis
# they are independant of the estimator and the observation space (x)
# N.B> L-C2ST is still dependent on the observation space (x)
# as its trained on the joint samples (theta, x)
t_stats_null_c2st_nf = {}
for N_cal in N_cal_list:
    t_stats_null_c2st_nf[N_cal] = precompute_t_stats_null(
        metrics=ALL_METRICS,
        n_cal=N_cal,
        n_eval=N_eval,
        dim_theta=dim_theta,
        n_trials_null=N_TRIALS_PRECOMPUTE,
        t_stats_null_path=task_path / "t_stats_null" / eval_params,
        methods=["c2st_nf"],
        kwargs_c2st=kwargs_c2st,
        save_results=True,
        load_results=True,
        # args for lc2st only
        kwargs_lc2st=None,
        x_cal=None,
        observation_dict=None,
    )["c2st_nf"]

# perform the experiment
# ==== EXP 1: test stats as a function of N_train (N_cal = max)==== #
if args.t_res_ntrain:
    print()
    print(f"Experiment 1: test statistics as a function of N_train...")
    print(f"... for N_cal = {N_cal}")
    print()

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

    METHODS = ["c2st", "lc2st", "c2st_nf", "lc2st_nf", "lc2st_nf_perm"]

    for N_cal in N_cal_list:
        avg_results, train_runtime = l_c2st_results_n_train(
            task,
            n_cal=N_cal,
            n_eval=N_eval,
            observation_dict=observation_dict,
            n_train_list=N_TRAIN_LIST,
            alpha=ALPHA,
            n_trials_null=args.n_trials_null,
            t_stats_null_c2st_nf=t_stats_null_c2st_nf[N_cal],
            n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
            kwargs_c2st=kwargs_c2st,
            kwargs_lc2st=kwargs_lc2st,
            task_path=task_path,
            t_stats_null_path=task_path / "t_stats_null" / eval_params,
            results_n_train_path=Path("results") / test_params / eval_params,
            methods=METHODS,
            test_stat_names=ALL_METRICS,
            seed=RANDOM_SEED,
        )

        # path to save figures
        fig_path = task_path / "figures" / eval_params / test_params
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        test_results_dict = {
            "vanilla C2ST ": {k: v["accuracy"] for k, v in avg_results["c2st"].items()},
            "Reg-C2ST": {k: v["mse"] for k, v in avg_results["c2st"].items()},
            "Reg-L-C2ST": {k: v["mse"] for k, v in avg_results["lc2st"].items()},
            "Reg-L-C2ST (NF)": {
                k: v["mse"] for k, v in avg_results["lc2st_nf"].items()
            },
            "Reg-L-C2ST (NF-Perm)": {
                k: v["mse"] for k, v in avg_results["lc2st_nf_perm"].items()
            },
            "Max-L-C2ST": {k: v["div"] for k, v in avg_results["lc2st"].items()},
            "Max-L-C2ST (NF)": {
                k: v["div"] for k, v in avg_results["lc2st_nf"].items()
            },
            "Max-L-C2ST (NF-Perm)": {
                k: v["div"] for k, v in avg_results["lc2st_nf_perm"].items()
            },
        }

        colors = ["grey", "blue", "orange", "orange", "gold", "red", "red", "coral"]
        linestyles = ["-", "-", "-", "--", "-.", "-", "--", "-."]

        markers = ["o", "o", "o", "*", "*", "o", "*", "*"]

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
                if "mean" in k:
                    k_std = k[:-5] + "_std"
                    plt.fill_between(
                        np.arange(len(N_TRAIN_LIST)),
                        np.array(test_results_dict[method][k])
                        - np.array(test_results_dict[method][k_std]),
                        np.array(test_results_dict[method][k])
                        + np.array(test_results_dict[method][k_std]),
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
            if "run_time" in k:
                plt.plot(
                    np.arange(len(N_TRAIN_LIST)),
                    train_runtime["lc2st"],
                    label="L-C2ST (pre-train)",
                    color="black",
                    linestyle="--",
                    marker="o",
                )
                plt.plot(
                    np.arange(len(N_TRAIN_LIST)),
                    train_runtime["lc2st_nf"],
                    label="L-C2ST-NF (pre-train)",
                    color="black",
                    linestyle="--",
                    marker="*",
                )
                k = k + " (1 trial)"

            if "std" not in k:
                plt.legend()
                plt.xticks(np.arange(len(N_TRAIN_LIST)), N_TRAIN_LIST)
                plt.xlabel("N_train")
                plt.ylabel(k)
                plt.savefig(fig_path / f"{k}_ntrain_n_cal_{N_cal}.pdf")
                plt.show()
            else:
                plt.close()

if args.power_ncal:
    print()
    print(f"Experiment 2: Empirical Power as a function of N_cal...")

    N_TRAIN = 100000
    print(f"... for N_train = {N_TRAIN} ...")
    print()

    METHODS = ["c2st", "lc2st", "lc2st_nf", "lc2st_nf_perm"]  # "c2st_nf",
    N_RUNS = 100

    result_path = task_path / f"npe_{N_TRAIN}" / "results" / test_params / eval_params
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    emp_power_dict, type_I_error_dict = {}, {}
    p_values_dict, p_values_h0_dict = {}, {}

    COMPUTE_FPR = True
    COMPUTE_TPR = False

    for N_cal in N_cal_list:
        try:
            if COMPUTE_TPR:
                emp_power_dict[N_cal] = torch.load(
                    result_path / f"emp_power_n_runs_{N_RUNS}_n_cal_{N_cal}.pkl",
                )
                p_values_dict[N_cal] = torch.load(
                    result_path / f"p_values_avg_n_runs_{N_RUNS}_n_cal{N_cal}.pkl",
                )
            if COMPUTE_FPR:
                type_I_error_dict[N_cal] = torch.load(
                    result_path / f"type_I_error_n_runs_{N_RUNS}_n_cal_{N_cal}.pkl",
                )
                p_values_h0_dict[N_cal] = torch.load(
                    result_path / f"p_values_h0_avg_n_runs_{N_RUNS}_n_cal{N_cal}.pkl",
                )
            print(f"Loaded Empirical Results for N_cal = {N_cal} ...")
        except FileNotFoundError:
            emp_power, type_I_error, p_values, p_values_h0 = compute_emp_power_l_c2st(
                n_runs=N_RUNS,
                alpha=ALPHA,
                task=task,
                n_train=N_TRAIN,
                observation_dict=observation_dict,
                n_cal=N_cal,
                n_eval=N_eval,
                n_trials_null=args.n_trials_null,
                kwargs_c2st=kwargs_c2st,
                kwargs_lc2st=kwargs_lc2st,
                t_stats_null_c2st_nf=t_stats_null_c2st_nf[N_cal],
                n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
                methods=METHODS,
                test_stat_names=ALL_METRICS,
                compute_emp_power=COMPUTE_TPR,
                compute_type_I_error=COMPUTE_FPR,
                task_path=task_path,
            )
            emp_power_dict[N_cal] = emp_power
            type_I_error_dict[N_cal] = type_I_error
            p_values_dict[N_cal] = p_values
            p_values_h0_dict[N_cal] = p_values_h0
            if COMPUTE_TPR:
                torch.save(
                    emp_power,
                    result_path / f"emp_power_n_runs_{N_RUNS}_n_cal_{N_cal}.pkl",
                )
                torch.save(
                    p_values,
                    result_path / f"p_values_avg_n_runs_{N_RUNS}_n_cal{N_cal}.pkl",
                )
            if COMPUTE_FPR:
                torch.save(
                    type_I_error,
                    result_path / f"type_I_error_n_runs_{N_RUNS}_n_cal{N_cal}.pkl",
                )
                torch.save(
                    p_values_h0,
                    result_path / f"p_values_h0_avg_n_runs_{N_RUNS}_n_cal{N_cal}.pkl",
                )
