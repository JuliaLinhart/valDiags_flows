# Graphical diagnostics for the validation of conditional density estimators,
# in particular in the context of SBI. They help interpret the results of the
# following test:
#
# 1. Simulation Based Calibration (SBC) [Talts et al. (2018)]
# 2. (Local) Classifier Two Sample Test (C2ST) (can be used for any SBI-algorithm)
#    - [Lopez et al. (2016)](https://arxiv.org/abs/1602.05336))
#    - [Lee et al. (2018)](https://arxiv.org/abs/1812.08927))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributions as D
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from scipy.stats import hmean, uniform


# ==== Functions applicable for both tests ====


def PP_vals(RV_values, alphas):
    pp_vals = [np.mean(RV_values <= alpha) for alpha in alphas]
    return pp_vals


def confidence_region_null(alphas, N=1000, conf_alpha=0.05, n_trials=1000):
    u_pp_values = {}
    for t in range(n_trials):
        u_samples = uniform().rvs(N)
        u_pp_values[t] = pd.Series(PP_vals(u_samples, alphas))
    lower_band = pd.DataFrame(u_pp_values).quantile(q=conf_alpha / 2, axis=1)
    upper_band = pd.DataFrame(u_pp_values).quantile(q=1 - conf_alpha / 2, axis=1)

    plt.fill_between(alphas, lower_band, upper_band, color="grey", alpha=0.2)


def box_plot_lc2st(
    scores, scores_null, labels, colors, title=r"Box plot", conf_alpha=0.05
):
    import matplotlib.cbook as cbook

    data = scores_null
    stats = cbook.boxplot_stats(data)[0]
    stats["q1"] = np.quantile(data, conf_alpha)
    stats["q3"] = np.quantile(data, 1 - conf_alpha)
    stats["whislo"] = min(data)
    stats["whishi"] = max(data)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    bp = ax.bxp([stats], widths=0.1, vert=False, showfliers=False, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgray")
    ax.set_label(r"95% confidence interval$")
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(stats["whislo"] - np.std(data), max(scores) + np.std(data))

    for s, l, c in zip(scores, labels, colors):
        plt.text(s, 0.9, l, color=c)
        plt.scatter(s, 1, color=c, zorder=10)

    fig.set_size_inches(5, 2)
    plt.title(title)


# ==== 1. SBC: PP-plot for SBC validation method =====


def sbc_plot(
    sbc_ranks,
    colors,
    labels,
    alphas=np.linspace(0, 1, 100),
    confidence_int=True,
    conf_alpha=0.05,
    title="SBC",
):
    """PP-plot for SBC validation method:
    Empirical distribution of the SBC ranks computed for every parameter seperately.

    inputs:
    - sbc_ranks: numpy array, size: (N, dim)
        For example one can use the output of sbi.analysis.sbc.run_sbc computed on
        N samples of the joint (Theta, X).
    - colors: list of strings, length: dim
    - labels: list of strings, length: dim
    - alphas: numpy array, size: (K,)
        Default is np.linspace(0,1,100).
    - confidence_int: bool
        Whether to show the confidence region (acceptance of the null hypothesis).
        Default is True.
    - conf_alpha: alpha level of the (1-conf-alpha)-confidence region.
        Default is 0.05, for a confidence level of 0.95.
    - title: sting
        Title of the plot.
    """
    lims = [np.min([0, 0]), np.max([1, 1])]
    plt.plot(lims, lims, "--", color="black", alpha=0.75)

    for i in range(len(sbc_ranks[0])):
        sbc_cdf = np.histogram(sbc_ranks[:, i], bins=len(alphas))[0].cumsum()
        plt.plot(alphas, sbc_cdf / sbc_cdf.max(), color=colors[i], label=labels[i])

    if confidence_int:
        # Construct uniform histogram.
        N = len(sbc_ranks)
        confidence_region_null(alphas=alphas, N=N, conf_alpha=conf_alpha)

    plt.ylabel("empirical CDF", fontsize=15)
    plt.xlabel("ranks", fontsize=15)
    plt.title(title, fontsize=18)
    plt.legend()


# ==== 2. (Local) Classifier Two Sample Test (C2ST) ====

# PP-plot of clasifier predicted class probabilities


def pp_plot_c2st(
    probas, probas_null, labels, colors, pp_vals_null=None, ax=None, **kwargs
):
    if ax == None:
        ax = plt.gca()
    alphas = np.linspace(0, 1, 100)
    pp_vals_dirac = PP_vals([0.5] * len(probas), alphas)
    ax.plot(
        alphas,
        pp_vals_dirac,
        "--",
        color="black",
    )

    if pp_vals_null is None:
        pp_vals_null = {}
        for t in range(len(probas_null)):
            pp_vals_null[t] = pd.Series(PP_vals(probas_null[t], alphas))

    low_null = pd.DataFrame(pp_vals_null).quantile(0.05 / 2, axis=1)
    up_null = pd.DataFrame(pp_vals_null).quantile(1 - 0.05 / 2, axis=1)
    ax.fill_between(
        alphas,
        low_null,
        up_null,
        color="grey",
        alpha=0.2,
        # label="95% confidence region",
    )

    for p, l, c in zip(probas, labels, colors):
        pp_vals = pd.Series(PP_vals(p, alphas))
        ax.plot(alphas, pp_vals, label=l, color=c, **kwargs)
    return ax


# Interpretability plots for C2ST: regions of high/low predicted class probabilities


def eval_space_with_proba_intensity(
    probas,
    probas_null,
    P_eval,
    dim=1,
    z_space=True,
    thresholding=False,
    n_bins=50,
    ax=None,
    show_colorbar=True,
    scatter=True,
):
    if ax is None:
        ax = plt.gca()
    df = pd.DataFrame({"probas": probas})

    if probas_null is not None:
        # define low and high thresholds w.r.t to null (95% confidence region)
        low = np.quantile(np.mean(probas_null, axis=0), q=0.05)
        high = np.quantile(np.mean(probas_null, axis=0), q=0.95)

    if thresholding:
        # high/low proba regions
        df["intensity"] = ["uncertain"] * len(df)
        df.loc[df["probas"] > high, "intensity"] = (
            r"high ($p \geq$ " + f"{np.round(high,2)})"
        )
        df.loc[df["probas"] < low, "intensity"] = (
            r"low ($p \leq$ " + f"{np.round(low,2)})"
        )

    if dim == 1:
        from matplotlib import cm

        df["z"] = P_eval[:, 0]

        if thresholding:
            df.pivot(columns="intensity", values="z").plot.hist(
                bins=n_bins, color=["red", "blue", "grey"], alpha=0.3
            )
        else:
            counts, bins, patches = ax.hist(df.z, n_bins, density=True, color="green")
            # bins[-1] = 10
            df["bins"] = np.select(
                [df.z <= i for i in bins[1:]], list(range(n_bins))
            )  # , 1000)

            weights = df.groupby(["bins"]).mean().probas
            id = list(set(range(n_bins)) - set(df.bins))
            patches = np.delete(patches, id)
            counts = np.delete(counts, id)
            bins = np.delete(bins, id)
            counts = np.array(counts) * np.diff(bins)
            counts = (counts - min(counts)) / (max(counts) - min(counts))

            cmap = plt.cm.get_cmap("coolwarm")
            norm = Normalize(vmin=0, vmax=1)

            for w, c, p in zip(weights, counts, patches):
                p.set_facecolor(cmap(w))
                p.set_alpha(c)
            if show_colorbar:
                ax.colorbar(
                    cm.ScalarMappable(cmap=cmap, norm=norm),
                    label=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$",
                )
        if z_space:
            xlabel = r"$z$"
        else:
            xlabel = r"$\theta$"
        # plt.xlabel(xlabel)

    elif dim == 2:
        if z_space:
            xlabel = r"$Z_1$"
            ylabel = r"$Z_2$"
            legend = r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$"
        else:
            xlabel = r"$\Theta_1$"
            ylabel = r"$\Theta_2$"
            legend = r"$\hat{p}(\Theta\sim q_{\phi}(\theta \mid x_0) \mid x_0)$"

        df["z_1"] = P_eval[:, 0]
        df["z_2"] = P_eval[:, 1]

        if not thresholding:
            if scatter:
                ax.scatter(df.z_1, df.z_2, c=df.probas, cmap="bwr", alpha=0.3)
            else:
                # h_prob = np.histogram2d(probas, probas, bins=n_bins)[0]
                h, x, y = np.histogram2d(df.z_1, df.z_2, bins=n_bins)
                h = h / np.sum(h)
                h = (h - np.min(h)) / (np.max(h) - np.min(h))

                df["bins_x"] = np.select(
                    [df.z_1 <= i for i in x[1:]], list(range(n_bins))
                )
                df["bins_y"] = np.select(
                    [df.z_2 <= i for i in y[1:]], list(range(n_bins))
                )
                prob_mean = df.groupby(["bins_x", "bins_y"]).mean().probas
                weights = np.zeros((n_bins, n_bins))
                for i in range(n_bins):
                    for j in range(n_bins):
                        try:
                            weights[i, j] = prob_mean.loc[i].loc[j]

                        except KeyError:
                            weights[i, j] = 0.5

                cmap = plt.get_cmap("coolwarm")
                norm = Normalize(vmin=0, vmax=1)
                # ax.pcolormesh(x, y, weights.T, cmap="coolwarm", norm=norm)
                for i in range(len(x) - 1):
                    for j in range(len(y) - 1):
                        rect = Rectangle(
                            (x[i], y[j]),
                            x[i + 1] - x[i],
                            y[j + 1] - y[j],
                            facecolor=cmap(norm(weights.T[j, i])),
                            alpha=h.T[j, i],
                            edgecolor='none',
                        )
                        ax.add_patch(rect)

            if show_colorbar:
                plt.colorbar(label=legend)
        else:
            cdict = {
                "uncertain": "grey",
                r"high ($p \geq$ " + f"{np.round(high,2)})": "red",
                r"low ($p \leq$ " + f"{np.round(low,2)})": "blue",
            }
            groups = df.groupby("intensity")

            _, ax = plt.subplots()
            for name, group in groups:
                x = group.z_1
                y = group.z_2
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="",
                    alpha=0.3,
                    label=name,
                    color=cdict[name],
                )
            plt.legend(title=legend)

        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)

    else:
        print("Not implemented.")

    return ax
