# ==== Compare different validation methods (2 sample tests - 2ST) on SBIBM tasks (toy-examples) ==== #
## Toy Examples:
# - two moons (D=2)
# - gaussian linear uniform (D=10)
#
# Validation methods to compare:
# REFERENCE for toy-examples when the true posterior is known (not amortized)
#   - Oracle C2ST (vanilla) - permutation method (not analytic because depends on classifier)
#   - Oracle C2ST (Reg) - permutation method (not analytic because depends on classifier)
#   - Oracle HPD (Highest Posterior Density Regions) (analytically known)
# OUR METHOD: when the true posterior is not known (amortized and single-class-eval)
#   - L-C2ST / LC2ST-NF (Max) (with permutation / pre-computed)
#   - L-C2ST / LC2ST-NF (Reg) (with permutation / pre-computed)
#   - L-HPD [Zhao et al. 2018] (how null?)
# Experiments to evaluate / compare the methods (on average over all observations x_0 from sbibm tasks):
#   - exp 1: results as a function of N_train
#   - exp 2: power / type 1 error / runtime as a function of the number of n_cal (at fixed N_train)

import argparse
from pathlib import Path
import os

import matplotlib.pyplot as plt

import numpy as np
import torch
import sbibm

from valdiags.vanillaC2ST import sbibm_clf_kwargs

from precompute_test_statistics_null import precompute_t_stats_null
from experiment_utils_sbibm import (
    compute_emp_power_l_c2st,
    l_c2st_results_n_train,
)

from plots_neurips2023_new import plot_sbibm_results_n_train

# set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# GLOBAL PARAMETERS
PATH_EXPERIMENT = Path("saved_experiments/neurips_2023/exp_2")

METHODS_ACC = [
    r"oracle C2ST ($\hat{t}_{Acc}$)",
    r"L-C2ST ($\hat{t}_{Max0}$)",
    r"L-C2ST-NF ($\hat{t}_{Max0}$)",
    # r"L-C2ST-NF-perm ($\hat{t}_{Max0}$)",
]
METHODS_L2 = [
    r"oracle C2ST ($\hat{t}_{Reg}$)",
    r"L-C2ST ($\hat{t}_{Reg0}$)",
    r"L-C2ST-NF ($\hat{t}_{Reg0}$)",
    # r"L-C2ST-NF-perm ($\hat{t}_{Reg0}$)",
    "L-HPD",
]
METHODS_ALL = [
    r"oracle C2ST ($\hat{t}_{Acc}$)",
    r"oracle C2ST ($\hat{t}_{Reg}$)",
    r"L-C2ST ($\hat{t}_{Max0}$)",
    r"L-C2ST-NF ($\hat{t}_{Max0}$)",
    # r"L-C2ST-NF-perm ($\hat{t}_{Max0}$)",
    r"L-C2ST ($\hat{t}_{Reg0}$)",
    r"L-C2ST-NF ($\hat{t}_{Reg0}$)",
    # r"L-C2ST-NF-perm ($\hat{t}_{Reg0}$)",
    "L-HPD",
]

# numbers of the observations x_0 from sbibm to evaluate the tests at
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# test parameters
ALPHA = 0.05
N_TRIALS_PRECOMPUTE = 1000
NB_HPD_LEVELS = 11

# metrics / test statistics
ALL_METRICS = ["accuracy", "mse", "div"]

# classifier parameters
CROSS_VAL = False
N_ENSEMBLE = 1

# Parse arguments
parser = argparse.ArgumentParser()

# data parameters
parser.add_argument(
    "--n_cal",
    nargs="+",
    type=int,
    default=[
        10000
    ],  # Use default for exp 1. Use [100, 500, 1000, 2000, 5000, 10000] for exp 2.
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

# experiment parameters
parser.add_argument(
    "--task",
    type=str,
    default="two_moons",
    choices=["two_moons", "gaussian_mixture", "gaussian_linear_uniform", "slcp"],
    help="Task from sbibm to perform the experiment on.",
)

parser.add_argument(
    "--n_train",
    nargs="+",
    type=int,
    default=[
        100000
    ],  # Use default for exp 2. Use [100, 1_000, 10_000, 100_000] for exp 1.
    help="Number of training samples used to train the NPE. Can be a list of integers.",
)

parser.add_argument(
    "--t_res_ntrain",
    action="store_true",
    help="Exp 1: Results as a function of N_train (at fixed N_cal=10_000).",
)

parser.add_argument(
    "--power_ncal",
    action="store_true",
    help="Exp 2: Plot the the empirical power / type 1 error as a function N_cal (at fixed N_train=100_000).",
)

parser.add_argument(
    "--plot",
    "-p",
    action="store_true",
    help="Plot results only.",
)

# Parse arguments
args = parser.parse_args()

print()
print("=================================================")
print("  VALIDATION METHOD COMPARISON for sbibm-tasks")
print("=================================================")
print()

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
n_cal_list = args.n_cal
n_eval = args.n_eval

dim_theta = prior(num_samples=1).shape[-1]

# classifier parameters
sbibm_kwargs = sbibm_clf_kwargs(ndim=dim_theta)

# set-up path-params to save results for given test params
test_params = f"alpha_{ALPHA}_n_trials_null_{args.n_trials_null}"
eval_params = f"n_eval_{n_eval}_n_ensemble_{N_ENSEMBLE}_cross_val_{CROSS_VAL}"

# kwargs for c2st_scores function
kwargs_c2st = {
    "cross_val": CROSS_VAL,
    "n_ensemble": N_ENSEMBLE,
    "clf_kwargs": sbibm_kwargs,
}
# kwargs for lc2st_scores function
kwargs_lc2st = {
    "cross_val": CROSS_VAL,
    "n_ensemble": N_ENSEMBLE,
    "single_class_eval": True,
    "clf_kwargs": sbibm_kwargs,
}

# kwargs for lhpd_scores function
# same as in the original code from zhao et al https://github.com/zhao-david/CDE-diagnostics
lhpd_clf_kwargs = {"alpha": 0, "max_iter": 25000}
kwargs_lhpd = {
    "n_alphas": NB_HPD_LEVELS,
    "n_ensemble": N_ENSEMBLE,
    "clf_kwargs": lhpd_clf_kwargs,
}

# pre-compute / load test statistics for the C2ST-NF null hypothesis
# they are independant of the estimator and the observation space (x)
# N.B> L-C2ST is still dependent on the observation space (x)
# as its trained on the joint samples (theta, x)
t_stats_null_c2st_nf = {ncal: None for ncal in n_cal_list}
# if not args.plot:
#     t_stats_null_c2st_nf[n_cal] = precompute_t_stats_null(
#         metrics=ALL_METRICS,
#         n_cal=n_cal,
#         n_eval=n_eval,
#         dim_theta=dim_theta,
#         n_trials_null=N_TRIALS_PRECOMPUTE,
#         t_stats_null_path=task_path / "t_stats_null" / eval_params,
#         methods=["c2st_nf"],
#         kwargs_c2st=kwargs_c2st,
#         save_results=True,
#         load_results=True,
#         # args for lc2st only
#         kwargs_lc2st=None,
#         kwargs_lhpd=None,
#         x_cal=None,
#         observation_dict=None,
#     )["c2st_nf"]

# perform the experiment
# ==== EXP 1: test stats as a function of N_train (n_cal = max)==== #
if args.t_res_ntrain:
    n_cal = n_cal_list[0]
    n_train_list = args.n_train

    print()
    print(
        f"Experiment 1: test statistics as a function of N_train in {n_train_list} ..."
    )
    print(f"... for N_cal = {n_cal}")
    print()

    methods = [
        "c2st",
        "lc2st",
        "lc2st_nf",
        "lc2st_nf_perm",
        "lhpd",
    ]  # , "lhpd", "c2st_nf"]

    avg_results, train_runtime = l_c2st_results_n_train(
        task,
        n_cal=n_cal,
        n_eval=n_eval,
        observation_dict=observation_dict,
        n_train_list=n_train_list,
        alpha=ALPHA,
        n_trials_null=args.n_trials_null,
        t_stats_null_c2st_nf=t_stats_null_c2st_nf[n_cal],
        n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
        kwargs_c2st=kwargs_c2st,
        kwargs_lc2st=kwargs_lc2st,
        kwargs_lhpd=kwargs_lhpd,
        task_path=task_path,
        t_stats_null_path=task_path / "t_stats_null" / eval_params,
        results_n_train_path=Path(f"results") / test_params / eval_params,
        methods=methods,
        test_stat_names=ALL_METRICS,
        seed=RANDOM_SEED,
        plot_mode=args.plot,
    )

    # path to save figures
    fig_path = (
        task_path
        / "figures"
        # / f"nt_precompute_{N_TRIALS_PRECOMPUTE}"
        / eval_params
        / test_params
    )
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # plot_sbibm_results_n_train(
    #     avg_results=avg_results,
    #     train_runtime=train_runtime,
    #     fig_path=fig_path,
    #     n_train_list=n_train_list,
    #     n_cal=n_cal,
    #     methods=METHODS_ACC,
    #     t_stat_ext="t_acc_max",
    # )

    # plot_sbibm_results_n_train(
    #     avg_results=avg_results,
    #     train_runtime=train_runtime,
    #     fig_path=fig_path,
    #     n_train_list=n_train_list,
    #     n_cal=n_cal,
    #     methods=METHODS_L2,
    #     t_stat_ext="t_reg_lhpd",
    # )

    fig = plot_sbibm_results_n_train(
        avg_results=avg_results,
        train_runtime=train_runtime,
        fig_path=fig_path,
        n_train_list=n_train_list,
        n_cal=n_cal,
        methods_acc=METHODS_ACC,
        methods_reg=METHODS_L2,
        methods_all=METHODS_ALL,
        t_stat_ext="new",
    )
    plt.savefig(fig_path / f"results_ntrain_n_cal_{n_cal}.pdf")
    plt.show()


if args.power_ncal:
    n_train = args.n_train[0]

    print()
    print(f"Experiment 2: Empirical Power as a function of N_cal in {n_cal_list} ...")

    print(f"... for N_train = {n_train}")
    print()

    methods = ["c2st", "lc2st", "lc2st_nf", "lc2st_nf_perm"]  # , "lhpd"]  # "c2st_nf",
    n_runs = 100

    result_path = task_path / f"npe_{n_train}" / "results" / test_params / eval_params
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    emp_power_dict, type_I_error_dict = {}, {}
    p_values_dict, p_values_h0_dict = {}, {}

    compute_tpr = True
    compute_fpr = True

    for n_cal in n_cal_list:
        emp_power_dict[n_cal], type_I_error_dict[n_cal] = {}, {}
        p_values_dict[n_cal], p_values_h0_dict[n_cal] = {}, {}
        try:
            for m in methods:
                if compute_tpr:
                    emp_power_dict[n_cal][m] = torch.load(
                        result_path
                        / f"n_runs_{n_runs}"
                        / f"emp_power_{m}_n_runs_{n_runs}_n_cal_{n_cal}.pkl",
                    )
                    p_values_dict[n_cal][m] = torch.load(
                        result_path
                        / f"n_runs_{n_runs}"
                        / f"p_values_obs_per_run_{m}_n_runs_{n_runs}_n_cal_{n_cal}.pkl",
                    )
                if compute_fpr:
                    type_I_error_dict[n_cal][m] = torch.load(
                        result_path
                        / f"n_runs_{n_runs}"
                        / f"type_I_error_{m}_n_runs_{n_runs}_n_cal_{n_cal}.pkl",
                    )
                    p_values_h0_dict[n_cal][m] = torch.load(
                        result_path
                        / f"n_runs_{n_runs}"
                        / f"p_values_h0__obs_per_run_{m}_n_runs_{n_runs}_n_cal{n_cal}.pkl",
                    )
            print(f"Loaded Empirical Results for n_cal = {n_cal} ...")
        except FileNotFoundError:
            emp_power, type_I_error, p_values, p_values_h0 = compute_emp_power_l_c2st(
                n_runs=n_runs,
                alpha=ALPHA,
                task=task,
                n_train=n_train,
                observation_dict=observation_dict,
                n_cal=n_cal,
                n_eval=n_eval,
                n_trials_null=args.n_trials_null,
                kwargs_c2st=kwargs_c2st,
                kwargs_lc2st=kwargs_lc2st,
                kwargs_lhpd=kwargs_lhpd,
                t_stats_null_c2st_nf=t_stats_null_c2st_nf[n_cal],
                n_trials_null_precompute=N_TRIALS_PRECOMPUTE,
                methods=methods,
                test_stat_names=ALL_METRICS,
                compute_emp_power=compute_tpr,
                compute_type_I_error=compute_fpr,
                task_path=task_path,
                load_eval_data=True,
                result_path=result_path,
                t_stats_null_path=task_path / "t_stats_null" / eval_params,
                results_n_train_path=Path(f"results") / test_params / eval_params,
                n_run_load_results=0,
                # save_every_n_runs=10,
            )
            emp_power_dict[n_cal] = emp_power
            type_I_error_dict[n_cal] = type_I_error
            p_values_dict[n_cal] = p_values
            p_values_h0_dict[n_cal] = p_values_h0

    emp_power_mean_dict = {
        m: {t_stat_name: [] for t_stat_name in ALL_METRICS} for m in methods
    }
    emp_power_std_dict = {
        m: {t_stat_name: [] for t_stat_name in ALL_METRICS} for m in methods
    }
    for n_cal in n_cal_list:
        for m in methods:
            for t_stat_name in ALL_METRICS:
                emp_power_mean_dict[m][t_stat_name].append(
                    np.mean(emp_power_dict[n_cal][m][t_stat_name])
                )
                emp_power_std_dict[m][t_stat_name].append(
                    np.std(emp_power_dict[n_cal][m][t_stat_name])
                )

    # plot empirical power
    for m in methods:
        for t_stat_name in ALL_METRICS:
            if "lc2st" in m and t_stat_name == "accuracy":
                continue
            if not "lc2st" in m and t_stat_name == "div":
                continue
            plt.plot(
                np.arange(len(n_cal_list)),
                emp_power_mean_dict[m]["accuracy"],
                label=m + " " + t_stat_name,
            )
            plt.fill_between(
                np.arange(len(n_cal_list)),
                np.array(emp_power_mean_dict[m]["accuracy"])
                - np.array(emp_power_std_dict[m]["accuracy"]),
                np.array(emp_power_mean_dict[m]["accuracy"])
                + np.array(emp_power_std_dict[m]["accuracy"]),
                alpha=0.2,
            )
    plt.xticks(np.arange(len(n_cal_list)), n_cal_list)
    plt.xlabel("n_cal")
    plt.ylabel("Empirical Power")
    plt.legend()
    plt.show()
