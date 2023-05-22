# =============================================================================

#       SCRIPT TO REPRODUCE FIGURES 3 and 4 IN NEURIPS 2023 SUBMISSION
#         (+ additional illustations in appendix A.6)

# =============================================================================

# DESCRIPTION: Inference and validation of the Jansen & Rit Neural Mass Model posterior.
# > Model: Jansen & Rit Neural Mass Model as used in [Rodrigues et al. (2021)](https://arxiv.org/abs/2102.06477)
#
# > Posterior Estimation via `sbi`-library (NPE)
#
# > Global Validation Methods
#   - SBC (`sbi` implementation),
#   - HPD (`lampe` implementation adapted to `sbi` posterior objects)
#
# > Local Validation
#   - L-C2ST-NF (ours): local C2ST for normalizing flow-based posterior estimators
#  (- local-HPD [Zhao et al. (2021)]: implemented but not used in paper)


# USAGE:
# >> python figure3_neurips_2023.py --global_ct
# >> python figure3_neurips_2023.py --local_ct_gain
# >> python figure3_neurips_2023.py --plots
# >> python figure3_neurips_2023.py --plots --lc2st_interpretability

# ====== Imports ======

import argparse
from pathlib import Path
import os
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

from valdiags.localC2ST import sbibm_clf_kwargs, lc2st_scores

from tasks.jrnmm.prior import prior_JRNMM

from precompute_test_statistics_null import precompute_t_stats_null
from experiment_utils_jrnmm import (
    train_posterior_jrnmm,
    generate_observations,
    global_coverage_tests,
    local_coverage_tests,
)
from plots_neurips2023 import (
    global_vs_local_tstats,
    plot_pairgrid_with_groundtruth_and_proba_intensity_lc2st,
    local_pp_plot,
)

# ====== GLOBAL PARAMETERS ======

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Path to save results
PATH_EXPERIMENT = Path("saved_experiments/neurips_2023/exp_3")

# Number of training samples for the NPE
N_TRAIN = 50000

# Test parameters
ALPHA = 0.05  # significance level for the tests
NB_HPD_LEVELS = 11

# Test statistics for L-C2ST
METRICS_LC2ST = ["mse"]  # , "div"]

# classifier parameters
CROSS_VAL = False
N_ENSEMBLE = 1

# ====== Parse arguments ======

parser = argparse.ArgumentParser()

# Data parameters
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

# Test parameters
parser.add_argument(
    "--n_trials_null",
    "-nt",
    type=int,
    default=1000,
    help="Number of trials to estimate the distribution of the test statistic under the null.",
)

# Experiment parameters
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
    "--lc2st_interpretability",
    "-lc2st_i",
    action="store_true",
    help="EXP 3: L-C2ST interpretability plots.",
)

parser.add_argument(
    "--plot",
    "-p",
    action="store_true",
    help="Plot final figures only.",
)

# ====== EXPERIMENTS ======

# Parse arguments
args = parser.parse_args()

print()
print("=================================================")
print("  L-C2ST: Appplication to the Jansen-Rit model")
print("=================================================")
print()

# SBI set-up for given task: prior
prior = prior_JRNMM(
    parameters=[
        ("C", 10.0, 250.0),
        ("mu", 50.0, 500.0),
        ("sigma", 100.0, 5000.0),
        ("gain", -20.0, +20.0),
    ]
)

# Infer NPE : loading a pre-trained npe, the same as in Neurips 2022 WS paper
# --> if the file is not in the experiment path, it will train a new one
try:
    npe_jrnmm = torch.load(PATH_EXPERIMENT / "posterior_estimator_jrnmm.pkl")
except FileNotFoundError:
    npe_jrnmm = train_posterior_jrnmm(N_TRAIN, PATH_EXPERIMENT)
    torch.save(npe_jrnmm, PATH_EXPERIMENT / "posterior_estimator_jrnmm.pkl")

# Ground truth parameters used to generate observations x_0
# fixed parameters
c = 135.0
mu = 220.0
sigma = 2000.0
# varying gain parameter
gain_list = np.arange(-25, 26, 5)  # [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]

try:
    # Load observations
    x_obs_list = torch.load(PATH_EXPERIMENT / "observations/gain_experiment.pkl")[1]
except FileNotFoundError:
    # Generate observations
    x_obs_list = generate_observations(
        c, mu, sigma, gain_list, load_path=PATH_EXPERIMENT / "observations"
    )
    torch.save(x_obs_list, PATH_EXPERIMENT / "observations/gain_experiment.pkl")

observation_dict = {g: x[None, :] for g, x in zip(gain_list, x_obs_list[:, :, 0])}

# Test set-up
n_cal = args.n_cal  # number of calibration samples
n_eval = args.n_eval  # number of evaluation samples

# Load pre-computed simulations - data from the joint distribution
joint_dataset = torch.load(PATH_EXPERIMENT / "joint_data_jrnmm_cal.pkl")
theta_cal, x_cal = joint_dataset["theta"][:n_cal], joint_dataset["x"][:n_cal]

dim_theta = theta_cal.shape[-1]  # dimension of the parameter space

# Classifier parameters
sbibm_kwargs = sbibm_clf_kwargs(ndim=dim_theta)  # same as from `sbibm`-library

# Path-params to save results for given test params
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

if args.local_ct_gain:
    print()
    print("EXP 2: Local Tests for observations obtained with varying gain")
    print()

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


# ====== PLOTS ONLY ======

if args.plot:
    # Path to save figures
    fig_path = PATH_EXPERIMENT / "figures"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Load Global Results
    global_rank_stats = global_coverage_tests(
        npe=npe_jrnmm,
        prior=prior,
        theta_cal=theta_cal,
        x_cal=x_cal,
        save_path=PATH_EXPERIMENT,
        methods=["sbc", "hpd"],
    )

    # Load Local Results
    # Method
    method = "lc2st_nf"

    lct_stats_null = torch.load(
        PATH_EXPERIMENT / "t_stats_null" / eval_params / "lct_stats_null_dict.pkl"
    )

    results_dict = torch.load(
        PATH_EXPERIMENT
        / "local_tests"
        / test_params
        / eval_params
        / f"{method}_results_n_eval_{n_eval}_n_cal_{x_cal.shape[0]}.pkl"
    )

    probas_obs_dict = torch.load(
        PATH_EXPERIMENT
        / "local_tests"
        / test_params
        / eval_params
        / f"{method}_probas_obs_n_eval_{n_eval}_n_cal_{x_cal.shape[0]}.pkl"
    )

    probas_null = torch.load(
        PATH_EXPERIMENT
        / "t_stats_null"
        / eval_params
        / "lc2st_nf_probas_nt_1000_n_cal_10000.pkl"
    )

    trained_clfs_dict = torch.load(
        PATH_EXPERIMENT
        / "local_tests"
        / test_params
        / eval_params
        / f"trained_clfs_{method}_n_cal_{x_cal.shape[0]}.pkl"
    )

    # Plot Global vs. Local Results
    fig = global_vs_local_tstats(
        sbc_alphas=np.linspace(0, 1, 100),
        sbc_ranks=global_rank_stats["sbc"],
        hpd_ranks=global_rank_stats["hpd"],
        gain_dict={
            g: i for i, g in enumerate(gain_list)
        },  # no -25 and 25 (outside of the prior)
        t_stats_obs={method: results_dict["t_stat"]},
        t_stats_obs_null=lct_stats_null,
        methods=[r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)"],
        labels=[r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)"],
        alpha=ALPHA,
        n_trials=args.n_trials_null,
    )
    plt.savefig(fig_path / "global_vs_local_tstats.pdf")
    plt.show()

    # Graphical L-C2ST Diagnostics
    if args.lc2st_interpretability:
        print("L-C2ST Interpretability")

        dict_obs_g = {g: i for i, g in enumerate(gain_list)}

        for g in gain_list:
            observation = x_obs_list[dict_obs_g[g]][None, :, :]
            theta_gt = np.array([c, mu, sigma, g])

            # Pairplot with ground truth and Predicted Probability (PP) intensity
            fig = plot_pairgrid_with_groundtruth_and_proba_intensity_lc2st(
                npe_jrnmm,
                theta_gt=theta_gt,
                observation=observation,
                trained_clfs_lc2st=trained_clfs_dict,
                scores_fn_lc2st=partial(lc2st_scores, **kwargs_lc2st),
                probas_null_obs_lc2st=None,
                n_samples=n_eval,
                n_bins=20,
            )
            plt.savefig(
                PATH_EXPERIMENT / f"local_tests/pairplot_with_intensity_g_{g}.pdf"
            )
            plt.show()

            # Local PP-Plot
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
            ax = local_pp_plot(
                probas_obs=[probas_obs_dict[g]],
                probas_obs_null=probas_null[g],
                method=r"$\ell$-C2ST-NF ($\hat{t}_{\mathrm{MSE}_0}$)",
                text=rf"$g_0 = {g}$",
            )
            plt.title("Local PP-Plot")
            plt.savefig(PATH_EXPERIMENT / f"local_tests/pp_plot_g_{g}.pdf")
            plt.show()
