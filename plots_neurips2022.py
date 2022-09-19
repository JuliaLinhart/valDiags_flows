import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, fontsizes

plt.rcParams.update(fontsizes.neurips2022())
plt.rcParams.update(fonts.neurips2022())

import numpy as np
import pandas as pd
import torch

from scipy.stats import binom

from diagnostics.pp_plots import PP_vals
from diagnostics.multi_local_test import get_lct_results


def multi_global_consistency(
    multi_PIT_values,
    alphas,
    sbc_ranks,
    labels,
    colors,
    ylabel_pit=r"empirical $r_{i,\alpha}$",
    ylabel_sbc="empirical CDF",
):
    plt.rcParams.update(figsizes.neurips2022(nrows=1, ncols=2, height_to_width_ratio=1))
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    for ax in axes:
        # plot identity function
        lims = [np.min([0, 0]), np.max([1, 1])]
        ax.plot(lims, lims, "--", color="black", alpha=0.75)

        # Construct uniform histogram.
        N = len(multi_PIT_values[0])
        nbins = len(alphas)
        hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
        hbb = hb.cumsum() / hb.sum()
        # avoid last value being exactly 1
        hbb[-1] -= 1e-9

        lower = [binom(N, p=p).ppf(0.005) for p in hbb]
        upper = [binom(N, p=p).ppf(0.995) for p in hbb]

        # Plot grey area with expected ECDF.
        ax.fill_between(
            x=np.linspace(0, 1, nbins),
            y1=np.repeat(lower / np.max(lower), 1),
            y2=np.repeat(upper / np.max(lower), 1),
            color="grey",
            alpha=0.3,
        )

    # sbc ranks
    for i in range(len(sbc_ranks[0])):
        sbc_cdf = np.histogram(sbc_ranks[:, i], bins=len(alphas))[0].cumsum()
        axes[0].plot(alphas, sbc_cdf / sbc_cdf.max(), color=colors[i], label=labels[i])

    axes[0].set_ylabel(ylabel_sbc, fontsize=15)
    axes[0].set_xlabel("ranks", fontsize=15)
    axes[0].set_title("SBC", fontsize=18)
    axes[0].legend()

    # global pit
    for i, Z in enumerate(multi_PIT_values):
        # compute quantiles P_{target}(PIT_values <= alpha)
        pp_vals = PP_vals(Z, alphas)
        # Plot the quantiles as a function of alpha
        axes[1].plot(alphas, pp_vals, color=colors[i], label=labels[i])

    axes[1].set_ylabel(ylabel_pit, fontsize=15)
    axes[1].set_xlabel(r"$\alpha$", fontsize=15)
    axes[1].set_title("Global PIT", fontsize=18)
    return fig


def multi_local_consistency(
    lct_path_list,
    gain_list,
    colors,
    labels,
    colors_g0=["#32327B", "#3838E2", "#52A9F5"],
):

    plt.rcParams.update(
        figsizes.neurips2022(nrows=2, ncols=3, height_to_width_ratio=1,)
    )

    fig = plt.figure()
    subfigs = fig.subfigures(nrows=2, ncols=1)

    axes1 = subfigs[0].subplots(nrows=1, ncols=3)
    for j in [0, 2]:
        axes1[j].remove()

    # test statistics
    df_lct_results = get_lct_results(lct_path_list, pvalues=False)
    df_lct_results.index = gain_list
    for i in range(1, 5):
        axes1[1].plot(
            gain_list,
            df_lct_results[f"dim_{i}"],
            marker="o",
            markersize=1,
            color=colors[i - 1],
            label=labels[i - 1],
            linewidth=1,
        )

    axes1[1].yaxis.set_tick_params(which="both", labelleft=True)
    axes1[1].set_xticks([gain_list[i] for i in [0, 2, 4, 6, 8]])
    axes1[1].set_yticks(
        np.round(np.linspace(0, np.max(df_lct_results.values), 5, endpoint=False), 2)
    )

    axes1[1].set_xlabel(r"$g_0$")
    axes1[1].set_ylabel(r"$T_i(x_0)$")
    handles = axes1[1].get_legend_handles_labels()
    subfigs[0].legend(
        handles=handles[0],
        bbox_to_anchor=axes1[1].get_position().get_points()[1] + np.array([0.2, -0.08]),
    )

    axes1[1].set_title("Local Test statistics")

    # pp-plots
    axes = subfigs[1].subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    for n, (x0, g0) in enumerate(zip([0, 4, 8], [-20, 0, 20])):
        r_alpha_x0 = torch.load(lct_path_list[x0])["r_alpha_learned"]
        handles_new = []
        for i in range(1, 5):
            axes[n].plot(
                np.linspace(0, 1, 100),
                pd.Series(r_alpha_x0[f"dim_{i}"]),
                color=colors[i - 1],
                marker="o",
                markersize=1,
                linestyle="",
            )
        axes[n].set_xlabel(r"$\alpha$")
        if n == 1:
            axes[n].set_xlabel(r"$\alpha$" + "\n" + "Local PP-plots")
        axes[n].set_title(r"$g_0=$" + f"{g0}")
        plt.setp(axes[n].spines.values(), color=colors_g0[n])
    axes[0].set_ylabel(r"$\hat{r}_{i,\alpha}(x_0)$")

    return fig
