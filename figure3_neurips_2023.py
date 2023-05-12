# # Inference and validation of the Jansen & Rit Neural Mass Model posterior

# 1. Posterior estimation via `sbi`-library
# 2. Global validation metrics include
#     - SBC (`sbi` implementation),
#     - HPD (`lampe` implementation adapted to `sbi` posterior objects)
#     - Global L-C2ST (ours) ???
# 3. Local validation metrics include
#     - L(ocal)-HPD (code for Zhao et al. 2020 paper)
#     - L(ocal)-C2ST (ours): cross-val score and hypothesis tests


import argparse
from pathlib import Path
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from tasks.jrnmm.prior import prior_JRNMM

from experiment_utils_jrnmm import (
    train_posterior_jrnmm,
    generate_observations,
    global_coverage_tests,
    local_coverage_tests,
)

from precompute_test_statistics_null import precompute_t_stats_null

from valdiags.graphical_valdiags import sbc_plot, confidence_region_null

from valdiags.localC2ST import sbibm_clf_kwargs

from plots_neurips2023 import (
    plot_pairgrid_with_groundtruth_and_proba_intensity_lc2st,
)

# set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# GLOBAL PARAMETERS
PATH_EXPERIMENT = Path("saved_experiments/neurips_2023/exp_3")

# Simulator parameters
SIM_PARAMETER_NAMES = [
    r"$\theta_1 = C$",
    r"$\theta_2 = \mu$",
    r"$\theta_3 = \sigma$",
    r"$\theta_4 = g$",
]

# number of training samples for the NPE
N_TRAIN = 50000

# test parameters
ALPHA = 0.05
NB_HPD_LEVELS = 11

# metrics / test statistics for L-C2ST
METRICS_LC2ST = ["mse", "div"]

# classifier parameters
CROSS_VAL = False
N_ENSEMBLE = 1

# Parse arguments
parser = argparse.ArgumentParser()

# data parameters
parser.add_argument(
    "--n_cal",
    type=int,
    default=10000,  # max value is 10000
    help="Number of calibration samples from the joint to compute the validation diagnostics.",
)

parser.add_argument(
    "--n_eval",
    type=int,
    default=10000,
    help="Number of evaluation samples for L-C2ST.",
)

# test parameters

# null distribution parameters
parser.add_argument(
    "--n_trials_null",
    "-nt",
    type=int,
    default=1000,
    help="Number of trials to estimate the distribution of the test statistic under the null.",
)

# experiment parameters

parser.add_argument(
    "--global_ct",
    "-gct",
    action="store_true",
    help="Exp 1: Global Tests results.",
)

parser.add_argument(
    "--local_ct_gain",
    "-lct_g",
    action="store_true",
    help="Exp 2: Local Tests results as a function of varying gain parameter.",
)

parser.add_argument(
    "--pp_plots",
    action="store_true",
    help="Exp 2: L-C2ST PP-Plots for every observation",
)

parser.add_argument(
    "--lc2st_interpretability",
    "-lc2st_i",
    action="store_true",
    help="EXP 3: L-C2ST interpretability plots.",
)

# Parse arguments
args = parser.parse_args()

print()
print("=================================================")
print("  L-C2ST: Appplication to the Jansen-Rit model")
print("=================================================")
print()

# ==== sbi set-up for given task ==== #
prior = prior_JRNMM(
    parameters=[
        ("C", 10.0, 250.0),
        ("mu", 50.0, 500.0),
        ("sigma", 100.0, 5000.0),
        ("gain", -20.0, +20.0),
    ]
)

# infer NPE : loading a pre-trained npe, the same as in Neurips 2022 WS paper
# --> if the file is not in the experiment path, it will train a new one
try:
    npe_jrnmm = torch.load(PATH_EXPERIMENT / "posterior_estimator_jrnmm.pkl")
except FileNotFoundError:
    npe_jrnmm = train_posterior_jrnmm(N_TRAIN, PATH_EXPERIMENT)
    torch.save(npe_jrnmm, PATH_EXPERIMENT / "posterior_estimator_jrnmm.pkl")

# ==== test set-up ==== #
# dataset sizes
n_cal = args.n_cal
n_eval = args.n_eval

# Load pre-computed simulations - data from the joint distribution
joint_dataset = torch.load(PATH_EXPERIMENT / "joint_data_jrnmm_cal.pkl")
theta_cal, x_cal = joint_dataset["theta"][:n_cal], joint_dataset["x"][:n_cal]

dim_theta = theta_cal.shape[-1]

# classifier parameters
sbibm_kwargs = sbibm_clf_kwargs(ndim=dim_theta)

# set-up path-params to save results for given test params
test_params = f"alpha_{ALPHA}_n_trials_null_{args.n_trials_null}"
eval_params = f"n_eval_{n_eval}_n_ensemble_{N_ENSEMBLE}_cross_val_{CROSS_VAL}"

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

if args.global_ct:
    print()
    print("EXP 1: Global Tests")
    print()

    global_rank_stats = global_coverage_tests(
        npe=npe_jrnmm,
        prior=prior,
        theta_cal=theta_cal,
        x_cal=x_cal,
        save_path=PATH_EXPERIMENT,
        methods=["sbc", "hpd"],
    )
    colors_sbc = ["#B9D6AF", "#B2C695", "#81B598", "#519D7A"]

    # # SBC
    # sbc_plot(
    #     global_rank_stats["sbc"],
    #     colors=colors_sbc,
    #     labels=SIM_PARAMETER_NAMES,
    #     conf_alpha=ALPHA / dim_theta,
    # )  # bonferonni correction
    # plt.savefig(PATH_EXPERIMENT / "global_tests/sbc_plot.pdf")
    # plt.show()

    # # HPD
    # confidence_region_null(np.linspace(0, 1, 100), N=n_cal, conf_alpha=ALPHA)
    # alphas = np.linspace(0.0, 1.0, len(global_rank_stats["hpd"]))
    # plt.plot(alphas, global_rank_stats["hpd"].numpy())
    # plt.title("Expected HPD")
    # plt.savefig(PATH_EXPERIMENT / "global_tests/hpd_plot.pdf")
    # plt.show()

    from plots_neurips2023 import global_coverage_pp_plots

    fig = global_coverage_pp_plots(
        multi_PIT_values=None,
        alphas=np.linspace(0, 1, 100),
        # colors_sbc=colors_sbc,
        labels_sbc=SIM_PARAMETER_NAMES,
        sbc_ranks=global_rank_stats["sbc"],
        hpd_ranks=global_rank_stats["hpd"],
    )
    plt.savefig(PATH_EXPERIMENT / "global_tests/global_consistency.pdf")
    plt.show()

if args.local_ct_gain:
    print()
    print("EXP 2: Local Tests for observations obtained with varying gain")
    print()

    # ground truth parameters used to generate observations x_0
    # fixed parameters
    c = 135.0
    mu = 220.0
    sigma = 2000.0
    # varying gain parameter
    gain_list = np.arange(-25, 26, 5)

    # generate observations
    try:
        # load observations
        x_obs_list = torch.load(PATH_EXPERIMENT / "observations/gain_experiment.pkl")[1]
    except FileNotFoundError:
        x_obs_list = generate_observations(
            c, mu, sigma, gain_list, load_path=PATH_EXPERIMENT / "observations"
        )
        torch.save(x_obs_list, PATH_EXPERIMENT / "observations/gain_experiment.pkl")

    observation_dict = {g: x[None, :] for g, x in zip(gain_list, x_obs_list[:, :, 0])}

    # pre-compute / load test statistics for the C2ST-NF null hypothesis
    # they are independant of the estimator and the observation space (x)
    # N.B> L-C2ST is still dependent on the observation space (x)
    # as its trained on the joint samples (theta, x)

    try:
        lct_stats_null = torch.load(
            PATH_EXPERIMENT / "t_stats_null" / eval_params / "lct_stats_null_dict.pkl"
        )
        probas_null = torch.load(
            PATH_EXPERIMENT / "t_stats_null" / eval_params / "probas_null_dict.pkl"
        )
    except FileNotFoundError:
        lct_stats_null, probas_null = precompute_t_stats_null(
            x_cal=x_cal[:, :, 0],
            n_cal=n_cal,
            n_eval=n_eval,
            observation_dict=observation_dict,
            dim_theta=dim_theta,
            n_trials_null=args.n_trials_null,
            t_stats_null_path=PATH_EXPERIMENT / "t_stats_null" / eval_params,
            methods=["lc2st_nf"],  # , "lhpd"],
            metrics=METRICS_LC2ST,
            kwargs_lc2st=kwargs_lc2st,
            kwargs_lhpd=kwargs_lhpd,
            save_results=True,
            load_results=True,
            return_predicted_probas=True,
            # args for lc2st only
            kwargs_c2st=None,
        )

        # save results
        torch.save(
            lct_stats_null,
            PATH_EXPERIMENT / "t_stats_null" / eval_params / "lct_stats_null_dict.pkl",
        )
        torch.save(
            probas_null,
            PATH_EXPERIMENT / "t_stats_null" / eval_params / "probas_null_dict.pkl",
        )

    # local test
    (
        results_dict,
        train_runtime,
        probas_obs_dict,
        trained_clfs_dict,
    ) = local_coverage_tests(
        alpha=ALPHA,
        npe=npe_jrnmm,
        theta_cal=theta_cal,
        x_cal=x_cal,
        n_eval=n_eval,
        observation_dict=observation_dict,
        t_stats_null_lc2st=lct_stats_null["lc2st_nf"],
        t_stats_null_lhpd=None,  # lct_stats_null["lhpd"],
        kwargs_lc2st=kwargs_lc2st,
        kwargs_lhpd=kwargs_lhpd,
        data_path=PATH_EXPERIMENT,
        result_path=PATH_EXPERIMENT / "local_tests" / test_params / eval_params,
        methods=["lc2st_nf"],  # , "lhpd"],
        test_stat_names=METRICS_LC2ST,
        return_predicted_probas=True,
        return_trained_clfs=True,
    )

    # print p_values
    for method in results_dict.keys():
        print()
        print(method)
        print()
        print(f"{results_dict[method]['p_value']}")

    if args.pp_plots:
        from valdiags.graphical_valdiags import pp_plot_c2st

        for g in gain_list:
            pp_plot_c2st(
                probas=[probas_obs_dict["lc2st_nf"][g]],
                probas_null=probas_null["lc2st_nf"][g],
                labels=[r"$g_0 = $" + f"{g}"],
                colors=["orange"],
            )
            plt.savefig(PATH_EXPERIMENT / f"local_tests/pp_plot_g_{g}.pdf")
            plt.show()

    if not args.lc2st_interpretability:
        # plot results
        from plots_neurips2023 import local_coverage_gain_plots

        fig = local_coverage_gain_plots(
            gain_dict={g: i for i, g in enumerate(gain_list)},
            t_stats_obs={k: v["t_stat"] for k, v in results_dict.items()},
            t_stats_obs_null=lct_stats_null,
            gain_list_pp_plots=[-20, 0, 20],
            probas_obs=probas_obs_dict,
            probas_obs_null=probas_null,
            p_values_obs={k: v["p_value"] for k, v in results_dict.items()},
            methods=[
                r"L-C2ST-NF ($\hat{t}_{Reg0}$)",
                # r"L-C2ST-NF ($\hat{t}_{Max0}$)",
                # "L-HPD",
            ],  # , "lhpd"],
        )
        plt.savefig(PATH_EXPERIMENT / "local_tests/local_consistency.pdf")
        plt.show()

    else:
        # interpretability
        dict_obs_g = {g: i for i, g in enumerate(gain_list)}
        for g in gain_list:
            observation = x_obs_list[dict_obs_g[g]][None, :, :]
            theta_gt = np.array([c, mu, sigma, g])

            from plots_neurips2023 import (
                plot_pairgrid_with_groundtruth_and_proba_intensity_lc2st,
            )
            from functools import partial
            from valdiags.localC2ST import lc2st_scores

            fig = plot_pairgrid_with_groundtruth_and_proba_intensity_lc2st(
                npe_jrnmm,
                theta_gt=theta_gt,
                observation=observation,
                trained_clfs_lc2st=trained_clfs_dict["lc2st_nf"],
                scores_fn_lc2st=partial(lc2st_scores, **kwargs_lc2st),
                probas_null_obs_lc2st=probas_null["lc2st_nf"][g],
                n_samples=n_eval,
                n_bins=20,
            )
            plt.savefig(
                PATH_EXPERIMENT / f"local_tests/pairplot_with_intensity_g_{g}.pdf"
            )
            plt.show()
