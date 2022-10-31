import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, fontsizes, axes
import matplotlib.gridspec as gridspec


import numpy as np
import pandas as pd
import torch
import seaborn as sns

from scipy.stats import binom

from diagnostics.pp_plots import PP_vals
from diagnostics.multi_local_test import get_lct_results


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
    hpd_values = None,
):
    # plt.rcParams.update(figsizes.neurips2022(nrows=1, ncols=3, height_to_width_ratio=1))
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.rcParams.update(fonts.neurips2022())
    plt.rcParams.update(axes.color(base="black"))
    plt.rcParams["legend.fontsize"] = 23.0
    plt.rcParams["xtick.labelsize"] = 23.0
    plt.rcParams["ytick.labelsize"] = 23.0
    plt.rcParams["axes.labelsize"] = 23.0
    plt.rcParams["font.size"] = 23.0
    plt.rcParams["axes.titlesize"] = 27.0

    fig, axs = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, constrained_layout=False
    )

    for i, ax in enumerate(axs):
        # plot identity function
        lims = [np.min([0, 0]), np.max([1, 1])]
        ax.plot(lims, lims, "--", color="black", alpha=0.75)
        if confidence_int:
            if i != 2:
                conf_alpha = conf_alpha/len(multi_PIT_values)
            # Construct uniform histogram.
            N = len(multi_PIT_values[0])
            nbins = len(alphas)
            hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
            hbb = hb.cumsum() / hb.sum()
            # avoid last value being exactly 1
            hbb[-1] -= 1e-9

            lower = [binom(N, p=p).ppf(conf_alpha / 2) for p in hbb]
            upper = [binom(N, p=p).ppf(1 - conf_alpha / 2) for p in hbb]

            # Plot grey area with expected ECDF.
            ax.fill_between(
                x=np.linspace(0, 1, nbins),
                y1=np.repeat(lower / np.max(lower), 1),
                y2=np.repeat(upper / np.max(lower), 1),
                color="grey",
                alpha=0.3,
            )
        ax.set_aspect("equal")

    # global pit
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
        axs[1].plot(
            alphas,
            sbc_cdf / sbc_cdf.max(),
            color=colors_sbc[i],
            label=labels_sbc[i],
            linewidth=2,
        )

    axs[1].set_ylabel(ylabel_sbc)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, 1)
    axs[1].set_xlabel(r"posterior rank $\theta_i$")
    axs[1].set_title("SBC")
    axs[1].legend(loc="upper left")

    # hpd_values 
    if hpd_values is not None:
        alphas = torch.linspace(0.0, 1.0, len(hpd_values))
        axs[2].plot(alphas, hpd_values, color='#1f77b4', label=r'$HPD(\mathbf{\theta})$')
        axs[2].set_ylabel(r'MC-est. $\mathbb{P}(HPD \leq \alpha)$')
        axs[2].set_ylim(0, 1)
        axs[2].set_xlim(0, 1)
        axs[2].set_xlabel(r"$\alpha$")
        axs[2].set_title("Global HPD")
        axs[2].legend(loc="upper left")
        
    return fig


def global_histograms(data, color, nbins=10, conf_alpha=0.05, ylim=1300):
    plt.rcParams.update(axes.color(base="white"))
    N = len(data)
    fig = plt.figure(figsize=(5, 3))
    low_lim = binom.ppf(q=conf_alpha / 2, n=N, p=1 / nbins)
    upp_lim = binom.ppf(q=1 - conf_alpha / 2, n=N, p=1 / nbins)
    plt.axhline(y=low_lim, color="darkgrey")
    plt.axhline(y=upp_lim, color="darkgrey")
    plt.axhline(y=N / nbins, linestyle="--", label="Uniform Average", color="black")

    plt.hist(data, bins=nbins, color=color)
    plt.xticks([])
    plt.xlim(0, 1)
    plt.ylim(0, ylim)
    plt.yticks([])
    plt.fill_between(
        x=np.linspace(0, 1, nbins),
        y1=np.repeat(low_lim, nbins),
        y2=np.repeat(upp_lim, nbins),
        color="darkgrey",
        alpha=0.5,
    )
    return fig


def multi_local_consistency(
    lct_path_list,
    gain_list,
    colors,
    labels,
    colors_g0=["#32327B", "#3838E2", "#52A9F5"],
    apply_permutation=None,
    r_alpha_null_list=None,
    conf_alpha=0.05,
):

    # plt.rcParams.update(
    #     figsizes.neurips2022(nrows=2, ncols=3, height_to_width_ratio=1,)
    # )
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams.update(fonts.neurips2022())
    plt.rcParams.update(axes.color(base="black"))
    plt.rcParams["legend.fontsize"] = 23.0
    plt.rcParams["xtick.labelsize"] = 23.0
    plt.rcParams["ytick.labelsize"] = 23.0
    plt.rcParams["axes.labelsize"] = 23.0
    plt.rcParams["font.size"] = 23.0
    plt.rcParams["axes.titlesize"] = 27.0

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3)

    ax = fig.add_subplot(gs[0, :])

    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[1, 2], sharex=ax0)
    axes1 = [ax0, ax1, ax2]
    # ax0.set_ylabel('YLabel1')
    for ax1 in [ax1, ax2]:
        ax1.set_yticklabels([])
        ax1.set_xticks([0.0, 0.5, 1.0])

    id = list(range(4))
    iperm = id
    if apply_permutation:
        iperm = apply_permutation(id)

    # test statistics
    df_lct_results = get_lct_results(lct_path_list, pvalues=False)
    df_lct_results.index = gain_list
    for i, ip in zip(id, iperm):
        ax.plot(
            gain_list,
            df_lct_results[f"dim_{ip+1}"],
            marker="o",
            markersize=3,
            color=colors[i],
            label=labels[i],
            linewidth=2,
        )

    ax.yaxis.set_tick_params(which="both", labelleft=True)
    ax.set_xticks(gain_list)
    ax.set_yticks(
        np.round(np.linspace(0, np.max(df_lct_results.values), 5, endpoint=False), 2)
    )

    ax.set_xlabel(r"$g_0$")
    ax.set_ylabel(r"$T_i(x_0)$")
    handles = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[0],
        title="1D-plots for",
        loc="upper right",
        # bbox_to_anchor=ax.get_position().get_points()[1]
        # + np.array([1.6, -0.08]),
    )

    ax.set_title("Local Test statistics")

    # pp-plots

    if r_alpha_null_list is not None:
        for ax1, r_alpha_null in zip(axes1, r_alpha_null_list):
            alphas = list(r_alpha_null[0].index)
            lower_band = pd.DataFrame(r_alpha_null).quantile(q=conf_alpha / 2, axis=1)
            upper_band = pd.DataFrame(r_alpha_null).quantile(
                q=1 - conf_alpha / 2, axis=1
            )

            ax1.fill_between(alphas, lower_band, upper_band, color="grey", alpha=0.2)

    for n, (ax1, x0, g0) in enumerate(zip(axes1, [0, 4, 8], [-20, 0, 20])):
        r_alpha_x0 = torch.load(lct_path_list[x0])["r_alpha_learned"]
        # plot identity function
        lims = [np.min([0, 0]), np.max([1, 1])]
        ax1.plot(lims, lims, "--", color="black", alpha=0.75)
        # plot pp-plots
        for i, ip in zip(id, iperm):
            ax1.plot(
                np.linspace(0, 1, 100),
                pd.Series(r_alpha_x0[f"dim_{ip+1}"]),
                color=colors[i],
                marker="o",
                markersize=1.5,
                linestyle="",
            )
        ax1.set_xlabel(r"$\alpha$", fontsize=23)
        if n == 1:
            ax1.set_title("Local PP-plots")
        ax1.text(0.01, 0.93, r"$g_0=$" + f"{g0}", fontsize=23)
        plt.setp(ax1.spines.values(), color=colors_g0[n])
    axes1[0].set_ylabel(r"$\hat{r}_{_i,\alpha}(x_0)$")
    axes1[0].set_yticks([0.0, 0.5, 1.0])

    # plt.subplots_adjust(wspace=None, hspace=0.4)

    for ax1 in axes1:
        ax1.set_aspect("equal")
    # ax.set_aspect("equal")
    # for j in [0, 2]:
    #     axes[0][j].set_visible(False)
    ax.set_xlim(-20.1, 20.1)
    # fig.align_ylabels()

    return fig


def plot_pairgrid_with_groundtruth(
    posteriors, theta_gt, color_dict, handles, context, n_samples=10000, title=None
):
    plt.rcParams["figure.figsize"] = (9, 9)
    plt.rcParams.update(fonts.neurips2022())
    plt.rcParams.update(axes.color(base="black"))
    plt.rcParams["legend.fontsize"] = 23.0
    plt.rcParams["xtick.labelsize"] = 23.0
    plt.rcParams["ytick.labelsize"] = 23.0
    plt.rcParams["axes.labelsize"] = 23.0
    plt.rcParams["font.size"] = 23.0
    plt.rcParams["axes.titlesize"] = 27.0

    modes = list(posteriors.keys())
    dfs = []
    for n in range(len(posteriors)):
        posterior = posteriors[modes[n]]
        if modes[n] == "prior":
            samples = posterior.sample(n_samples, context=context[modes[n]])
        else:
            samples = posterior.sample(n_samples, context=context[modes[n]])
        df = pd.DataFrame(
            samples.detach().numpy(), columns=[r"$C$", r"$\mu$", r"$\sigma$", r"$g$"]
        )
        df["mode"] = modes[n]
        dfs.append(df)

    joint_df = pd.concat(dfs, ignore_index=True)

    g = sns.PairGrid(
        joint_df, hue="mode", palette=color_dict, diag_sharey=False, corner=True
    )
    g.fig.set_size_inches(8, 8)

    g.map_lower(sns.kdeplot, linewidths=1)
    g.map_diag(sns.kdeplot, shade=True, linewidths=1)

    g.axes[1][0].set_xlim(10.0, 300.0)  # C
    g.axes[1][0].set_ylim(50.0, 500.0)  # mu
    g.axes[1][0].set_yticks([200, 400])

    g.axes[2][0].set_xlim(10.0, 300.0)  # C
    g.axes[2][0].set_ylim(100.0, 5000.0)  # sigma
    g.axes[2][0].set_yticks([1000, 3500])

    g.axes[2][1].set_xlim(50.0, 500.0)  # mu
    g.axes[2][1].set_ylim(100.0, 5000.0)  # sigma
    # g.axes[2][1].set_xticks([])

    g.axes[3][0].set_xlim(10.0, 300.0)  # C
    g.axes[3][0].set_ylim(-22.0, 22.0)  # gain
    g.axes[3][0].set_yticks([-20, 0, 20])
    g.axes[3][0].set_xticks([100, 250])

    g.axes[3][1].set_xlim(50.0, 500.0)  # mu
    g.axes[3][1].set_ylim(-22.0, 22.0)  # gain
    g.axes[3][1].set_xticks([200, 400])

    g.axes[3][2].set_xlim(100.0, 5000.0)  # sigma
    g.axes[3][2].set_ylim(-22.0, 22.0)  # gain
    g.axes[3][2].set_xticks([1000, 3500])

    g.axes[3][3].set_xlim(-22.0, 22.0)  # gain

    if theta_gt is not None:
        # get groundtruth parameters
        for gt in theta_gt:
            C, mu, sigma, gain = gt

            # plot points
            g.axes[1][0].scatter(C, mu, color="black", zorder=2, s=8)
            g.axes[2][0].scatter(C, sigma, color="black", zorder=2, s=8)
            g.axes[2][1].scatter(mu, sigma, color="black", zorder=2, s=8)
            g.axes[3][0].scatter(C, gain, color="black", zorder=2, s=8)
            g.axes[3][1].scatter(mu, gain, color="black", zorder=2, s=8)
            g.axes[3][2].scatter(sigma, gain, color="black", zorder=2, s=8)

            # plot dirac
            g.axes[0][0].axvline(x=C, ls="--", c="black", linewidth=1)
            g.axes[1][1].axvline(x=mu, ls="--", c="black", linewidth=1)
            g.axes[2][2].axvline(x=sigma, ls="--", c="black", linewidth=1)
            g.axes[3][3].axvline(x=gain, ls="--", c="black", linewidth=1)

    plt.legend(
        handles=handles,
        title=title,
        bbox_to_anchor=(1.1, 4.3),
        # loc="upper right",
    )
    g.fig.suptitle("Local pair-plots", y=1.02)

    return g
