import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tueplots import fonts, axes
import matplotlib.gridspec as gridspec

import pandas as pd
import torch

from scipy.stats import binom, uniform

from valdiags.graphical_valdiags import PP_vals

# ======== FIGURE 2 ========== #

METHODS_DICT = {
    r"oracle C2ST ($\hat{t}_{Acc}$)": {
        "test_name": "c2st",
        "t_stat_name": "accuracy",
        "color": "grey",
        "linestyle": "-",
        "marker": "o",
        "markersize": 6,
    },
    r"oracle C2ST ($\hat{t}_{Reg}$)": {
        "test_name": "c2st",
        "t_stat_name": "mse",
        "color": "darkgrey",
        "linestyle": "-",
        "marker": "o",
        "markersize": 6,
    },
    r"L-C2ST ($\hat{t}_{Reg0}$)": {
        "test_name": "lc2st",
        "t_stat_name": "mse",
        "color": "orange",
        "linestyle": "-",
        "marker": "o",
        "markersize": 6,
    },
    r"L-C2ST-NF ($\hat{t}_{Reg0}$)": {
        "test_name": "lc2st_nf",
        "t_stat_name": "mse",
        "color": "orange",
        "linestyle": "-.",
        "marker": "*",
        "markersize": 10,
    },
    r"L-C2ST-NF-perm ($\hat{t}_{Reg0}$)": {
        "test_name": "lc2st_nf_perm",
        "t_stat_name": "mse",
        "color": "darkorange",
        "linestyle": "-.",
        "marker": "o",
        "markersize": 6,
    },
    r"L-C2ST ($\hat{t}_{Max0}$)": {
        "test_name": "lc2st",
        "t_stat_name": "div",
        "color": "blue",
        "linestyle": "-",
        "marker": "o",
        "markersize": 6,
    },
    r"L-C2ST-NF ($\hat{t}_{Max0}$)": {
        "test_name": "lc2st_nf",
        "t_stat_name": "div",
        "color": "blue",
        "linestyle": "-.",
        "marker": "*",
        "markersize": 10,
    },
    r"L-C2ST-NF-perm ($\hat{t}_{Max0}$)": {
        "test_name": "lc2st_nf_perm",
        "t_stat_name": "div",
        "color": "darkblue",
        "linestyle": "-.",
        "marker": "o",
        "markersize": 6,
    },
    "L-HPD": {
        "test_name": "lhpd",
        "t_stat_name": "mse",
        "color": "#3BA071",
        "linestyle": "-",
        "marker": "x",
        "markersize": 6,
    },
}

avg_result_keys = {
    "TPR": "reject",
    "p_value_mean": "p_value",
    "p_value_std": "p_value",
    "t_stat_mean": "t_stat",
    "t_stat_std": "t_stat",
    "run_time_mean": "run_time",
    "run_time_std": "run_time",
}


def plot_sbibm_results_n_train(
    avg_results,
    train_runtime,
    methods,
    n_train_list,
    n_cal,
    fig_path,
    t_stat_ext="t_all",
):
    # plt.rcParams.update(figsizes.neurips2022(nrows=1, ncols=3, height_to_width_ratio=1))
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams.update(fonts.neurips2022())
    plt.rcParams.update(axes.color(base="black"))
    plt.rcParams["legend.fontsize"] = 23.0
    plt.rcParams["xtick.labelsize"] = 23.0
    plt.rcParams["ytick.labelsize"] = 23.0
    plt.rcParams["axes.labelsize"] = 23.0
    plt.rcParams["font.size"] = 23.0
    plt.rcParams["axes.titlesize"] = 27.0
    # ==== t_stats of L-C2ST(-NF) w.r.t to oracle ====

    # plot theoretical H_0 value
    plt.plot(
        np.arange(len(n_train_list)),
        np.ones(len(n_train_list)) * 0.5,
        "--",
        color="black",
        label=r"theoretical $t \mid \mathcal{H}_0$",
    )
    # plot estimated T values
    for method in methods:
        if "perm" in method or "lhpd" in method:
            continue

        test_name = METHODS_DICT[method]["test_name"]
        t_stat_name = METHODS_DICT[method]["t_stat_name"]
        plt.plot(
            np.arange(len(n_train_list)),
            avg_results[test_name]["t_stat_mean"][t_stat_name],
            label=method,
            color=METHODS_DICT[method]["color"],
            linestyle=METHODS_DICT[method]["linestyle"],
            marker=METHODS_DICT[method]["marker"],
            markersize=METHODS_DICT[method]["markersize"],
            alpha=0.8,
        )
        err = np.array(avg_results[test_name]["t_stat_std"][t_stat_name])
        plt.fill_between(
            np.arange(len(n_train_list)),
            np.array(avg_results[test_name]["t_stat_mean"][t_stat_name]) - err,
            np.array(avg_results[test_name]["t_stat_mean"][t_stat_name]) + err,
            alpha=0.2,
            color=METHODS_DICT[method]["color"],
        )
    plt.legend()
    plt.xticks(np.arange(len(n_train_list)), n_train_list)
    plt.xlabel("N_train")
    plt.ylabel("test statistic (mean +/- std)")
    plt.savefig(fig_path / f"t_stats_{t_stat_ext}_ntrain_n_cal_{n_cal}.pdf")
    plt.show()

    # ==== p-value of all methods w.r.t to oracle ===

    # plot alpha-level
    plt.plot(
        np.arange(len(n_train_list)),
        np.ones(len(n_train_list)) * 0.05,
        "--",
        color="black",
        label="alpha-level",
    )

    # plot estimated p-values
    for method in methods:
        test_name = METHODS_DICT[method]["test_name"]
        t_stat_name = METHODS_DICT[method]["t_stat_name"]
        plt.plot(
            np.arange(len(n_train_list)),
            avg_results[test_name]["p_value_mean"][t_stat_name],
            label=method,
            color=METHODS_DICT[method]["color"],
            linestyle=METHODS_DICT[method]["linestyle"],
            marker=METHODS_DICT[method]["marker"],
            markersize=METHODS_DICT[method]["markersize"],
            alpha=0.8,
        )
        low = np.array(avg_results[test_name]["p_value_min"][t_stat_name])
        high = np.array(avg_results[test_name]["p_value_max"][t_stat_name])
        plt.fill_between(
            np.arange(len(n_train_list)),
            low,
            high,
            alpha=0.2,
            color=METHODS_DICT[method]["color"],
        )
    plt.legend()
    plt.xticks(np.arange(len(n_train_list)), n_train_list)
    plt.xlabel("N_train")
    plt.ylabel("p-value (mean / min-max)")
    plt.savefig(fig_path / f"p_values_{t_stat_ext}_ntrain_n_cal_{n_cal}.pdf")
    plt.show()

    # plot rejection rate of all methods w.r.t to oracle
    for method in methods:
        test_name = METHODS_DICT[method]["test_name"]
        t_stat_name = METHODS_DICT[method]["t_stat_name"]
        plt.plot(
            np.arange(len(n_train_list)),
            avg_results[test_name]["TPR"][t_stat_name],
            label=method,
            color=METHODS_DICT[method]["color"],
            linestyle=METHODS_DICT[method]["linestyle"],
            marker=METHODS_DICT[method]["marker"],
            markersize=METHODS_DICT[method]["markersize"],
            alpha=0.8,
        )
    plt.legend()
    plt.xticks(np.arange(len(n_train_list)), n_train_list)
    plt.xlabel("N_train")
    plt.ylabel("rejection rate")
    plt.savefig(fig_path / f"rejection_rate_{t_stat_ext}_ntrain_n_cal_{n_cal}.pdf")
    plt.show()

    # plot run time of amortized methods w.r.t to oracle
    for method in methods:
        if "perm" in method:
            continue
        test_name = METHODS_DICT[method]["test_name"]
        t_stat_name = METHODS_DICT[method]["t_stat_name"]
        plt.plot(
            np.arange(len(n_train_list)),
            avg_results[test_name]["run_time_mean"][t_stat_name],
            label=method,
            color=METHODS_DICT[method]["color"],
            linestyle=METHODS_DICT[method]["linestyle"],
            marker=METHODS_DICT[method]["marker"],
            markersize=METHODS_DICT[method]["markersize"],
            alpha=0.8,
        )
        err = np.array(avg_results[test_name]["run_time_std"][t_stat_name])
        plt.fill_between(
            np.arange(len(n_train_list)),
            np.array(avg_results[test_name]["run_time_mean"][t_stat_name]) - err,
            np.array(avg_results[test_name]["run_time_mean"][t_stat_name]) + err,
            alpha=0.2,
            color=METHODS_DICT[method]["color"],
        )
        if not "oracle" in method:
            plt.plot(
                np.arange(len(n_train_list)),
                train_runtime[METHODS_DICT[method]["test_name"]],
                label=f"{method} (pre-train)",
                color="black",
                linestyle=METHODS_DICT[method]["linestyle"],
                marker=METHODS_DICT[method]["marker"],
                markersize=METHODS_DICT[method]["markersize"],
            )

    plt.legend()
    plt.xticks(np.arange(len(n_train_list)), n_train_list)
    plt.xlabel("N_train")
    plt.ylabel("runtime (s) (mean +/- std)")
    plt.savefig(fig_path / f"runtime_{t_stat_ext}_ntrain_n_cal_{n_cal}.pdf")
    plt.show()


### ======== FIGURE 3 ========== ###


def multi_global_consistency(
    multi_PIT_values,
    alphas,
    sbc_ranks,
    labels_sbc,
    labels_pit,
    colors_sbc,
    colors_pit,
    ylabel_pit=r"empirical $r_{i,\alpha} = \mathbb{P}(P_{i}\leq \alpha)$",
    ylabel_sbc=r"empirical CDF",
    confidence_int=True,
    conf_alpha=0.05,
    n_trials=1000,
    hpd_ranks=None,
):
    # plt.rcParams.update(figsizes.neurips2022(nrows=1, ncols=3, height_to_width_ratio=1))
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams.update(fonts.neurips2022())
    plt.rcParams.update(axes.color(base="black"))
    plt.rcParams["legend.fontsize"] = 23.0
    plt.rcParams["xtick.labelsize"] = 23.0
    plt.rcParams["ytick.labelsize"] = 23.0
    plt.rcParams["axes.labelsize"] = 23.0
    plt.rcParams["font.size"] = 23.0
    plt.rcParams["axes.titlesize"] = 27.0

    if multi_PIT_values is None:
        n_cols = 2
        ax_sbc = 0
        ax_hpd = 1
        plt.rcParams["figure.figsize"] = (10, 5)
    else:
        n_cols = 3
        ax_sbc = 1
        ax_hpd = 2
        plt.rcParams["figure.figsize"] = (15, 5)

    fig, axs = plt.subplots(
        nrows=1, ncols=n_cols, sharex=True, sharey=True, constrained_layout=False
    )

    for i, ax in enumerate(axs):
        # plot identity function
        lims = [np.min([0, 0]), np.max([1, 1])]
        ax.plot(lims, lims, "--", color="black", alpha=0.75)
        if confidence_int:
            if i == 0:
                conf_alpha = conf_alpha / len(sbc_ranks[0])  # bonferonni correction
            # Construct uniform histogram.
            N = len(sbc_ranks)
            u_pp_values = {}
            for t in range(n_trials):
                u_samples = uniform().rvs(N)
                u_pp_values[t] = pd.Series(PP_vals(u_samples, alphas))
            lower_band = pd.DataFrame(u_pp_values).quantile(q=conf_alpha / 2, axis=1)
            upper_band = pd.DataFrame(u_pp_values).quantile(
                q=1 - conf_alpha / 2, axis=1
            )

            ax.fill_between(alphas, lower_band, upper_band, color="grey", alpha=0.3)

        ax.set_aspect("equal")

    # global pit
    if multi_PIT_values is not None:
        for i, Z in enumerate(multi_PIT_values):
            # compute quantiles P_{target}(PIT_values <= alpha)
            pp_vals = PP_vals(Z, alphas)
            # Plot the quantiles as a function of alpha
            axs[0].plot(
                alphas, pp_vals, color=colors_pit[i], label=labels_pit[i], linewidth=2
            )

        axs[0].set_yticks([0.0, 0.5, 1.0])
        axs[0].set_ylabel(ylabel_pit)
        axs[0].set_xlabel(r"$\alpha$")
        axs[0].set_title("Global PIT")
        axs[0].legend(loc="upper left")

    # sbc ranks
    for i in range(len(sbc_ranks[0])):
        sbc_cdf = np.histogram(sbc_ranks[:, i], bins=len(alphas))[0].cumsum()
        axs[ax_sbc].plot(
            alphas,
            sbc_cdf / sbc_cdf.max(),
            color=colors_sbc[i],
            label=labels_sbc[i],
            linewidth=2,
        )

    axs[ax_sbc].set_ylabel(ylabel_sbc)
    axs[ax_sbc].set_ylim(0, 1)
    axs[ax_sbc].set_xlim(0, 1)
    axs[ax_sbc].set_xlabel(r"posterior rank $\theta_i$")
    axs[ax_sbc].set_title("SBC")
    axs[ax_sbc].legend(loc="upper left")

    # hpd_values
    if hpd_ranks is not None:
        alphas = torch.linspace(0.0, 1.0, len(hpd_ranks))
        axs[ax_hpd].plot(
            alphas, hpd_ranks, color="#1f77b4", label=r"$HPD(\mathbf{\theta})$"
        )
        axs[ax_hpd].set_ylabel(r"MC-est. $\mathbb{P}(HPD \leq \alpha)$")
        axs[ax_hpd].set_ylim(0, 1)
        axs[ax_hpd].set_xlim(0, 1)
        axs[ax_hpd].set_xlabel(r"$\alpha$")
        axs[ax_hpd].set_title("Global HPD")
        axs[ax_hpd].legend(loc="upper left")

    return fig
