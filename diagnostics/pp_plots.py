import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import torch

import sys

import torch.distributions as D

from matplotlib.lines import Line2D

from scipy.stats import hmean, binom


sys.path.append("../")
from nde.flows import cdf_flow


def multi_cde_pit_values(
    samples_theta, samples_x, flow, feature_transform, local=False
):
    dim = samples_theta.shape[-1]
    pit_values = []
    for i in range(dim):
        if local:
            conditional_transform_1d = (
                D.Normal(0, 1)
                .cdf(
                    flow._transform(
                        samples_theta, context=feature_transform(samples_x)
                    )[0][:, i]
                )
                .detach()
                .numpy()
            )
        else:
            conditional_transform_1d = (
                D.Normal(0, 1)
                .cdf(
                    flow._transform(
                        samples_theta, context=feature_transform(samples_x)
                    )[0][:, i]
                )
                .detach()
                .numpy()
            )

        pit_values.append(conditional_transform_1d)

    return pit_values


def cde_pit_values(samples_theta, samples_x, flow, feature_transform, local=False):
    # Compute PIT-values of the flows F_{Q|X_i}(Theta_i)
    if local:
        pit_values = (
            cdf_flow(samples_theta, context=feature_transform(samples_x), flow=flow)
            .detach()
            .numpy()
        )
    else:
        pit_values = np.array(
            [
                cdf_flow(samples_theta[i][None], context=x, flow=flow).detach().numpy()
                for i, x in enumerate(feature_transform(samples_x))
            ]
        )
    return pit_values


def PP_vals(PIT_values, alphas):
    """Compute alpha quantiles of the PIT-distribution P(PIT_values <= alpha):
    where P is the distribution of the target distribution (empirical approx)

    inputs:
    - PIT_values: numpy array of PIT values of a given estimator
    computed for samples of the target distribution
    - alphas: numpy array of values to evaluate the PP-vals
    """
    z = [np.mean(PIT_values <= alpha) for alpha in alphas]
    return z


def PP_plot_1D(
    PIT_values,
    alphas,
    r_alpha_learned=None,
    colors=["blue"],
    colors_r_alpha=["red"],
    labels=["Target"],
    title="PIT-distribution",
    ylabel=r"$r_{\alpha}(x_0)$",
    xlabel=r"$\alpha$",
    pvalue=None,
    sbc_ranks=None,
    confidence_int=False,
):
    """1D PP-plot: distribution of the PIT vs. uniform distribution
        It shows the deviation to the identity function and thus
        allows to evaluate how well the given (estimated) distribution
        matches the samples from the target distribution.

    inputs:
        - PIT_values: list of numpy arrays of PIT values of given estimators
            computed for samples of the target distribution
        - alphas: numpy array of values to evaluate the PP-vals
        - r_alpha_learned: regressed quantile values for local PIT
    """
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["mathtext.fontset"] = "cm"
    # plot identity function
    fig = plt.figure()
    lims = [np.min([0, 0]), np.max([1, 1])]
    plt.plot(lims, lims, "--", color="black", alpha=0.75)

    if PIT_values is not None:
        for i, Z in enumerate(PIT_values):
            # compute quantiles P_{target}(PIT_values <= alpha)
            pp_vals = PP_vals(Z, alphas)
            # Plot the quantiles as a function of alpha
            plt.plot(alphas, pp_vals, color=colors[i], label=labels[i])
            if sbc_ranks is not None:
                sbc_cdf = np.histogram(sbc_ranks[:, i], bins=len(alphas))[0].cumsum()
                plt.plot(
                    alphas,
                    sbc_cdf / sbc_cdf.max(),
                    "--",
                    color=colors[i],
                    label=labels[i],
                )

    handles_new = []
    if r_alpha_learned is not None:
        for i, r_alpha in enumerate(r_alpha_learned):
            label = labels[i]
            style = "o"
            if PIT_values is not None:
                if len(r_alpha_learned) == 1:
                    label = "Learned"
                else:
                    handles_new.append(
                        Line2D(
                            [],
                            [],
                            color=colors_r_alpha[i],
                            marker=".",
                            linestyle="solid",
                            label=labels[i],
                        )
                    )

            _ = pd.Series(r_alpha).plot(
                style=style, color=colors_r_alpha[i], markersize=3, label=label
            )

    handles, _ = plt.gca().get_legend_handles_labels()
    if handles_new:
        handles = handles_new

    if pvalue is not None:
        plt.text(0.9, 0.1, f"pvalue = {pvalue}", horizontalalignment="center")

    if confidence_int:
        # Construct uniform histogram.
        N = 10000
        nbins = len(alphas)
        hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
        hbb = hb.cumsum() / hb.sum()
        # avoid last value being exactly 1
        hbb[-1] -= 1e-9

        lower = [binom(N, p=p).ppf(0.05/2) for p in hbb]
        upper = [binom(N, p=p).ppf(1-0.05/2) for p in hbb]

        # Plot grey area with expected ECDF.
        plt.fill_between(
            x=np.linspace(0, 1, nbins),
            y1=np.repeat(lower / np.max(lower), 1),
            y2=np.repeat(upper / np.max(lower), 1),
            color="grey",
            alpha=0.3,
        )

    plt.legend(handles=handles)
    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.title(title, fontsize=18)
    plt.show()


def compare_pp_plots_regression(
    r_alpha_learned, true_pit_values, x_evals, nb_train_samples, labels
):
    for j, x_eval in enumerate(x_evals):
        # plot identity function
        lims = [np.min([0, 0]), np.max([1, 1])]
        plt.plot(lims, lims, "--", color="black", alpha=0.75)

        # compute quantiles P_{target}(PIT_values <= alpha)
        alphas = np.linspace(0, 0.999, 100)
        pp_vals = PP_vals(true_pit_values[j], alphas)
        # Plot the true quantiles as a function of alpha
        plt.plot(alphas, pp_vals, color="blue", label="True")

        colors = ["green", "purple", "orange", "red", "pink", "yellow"]
        for i, r_alpha in enumerate(r_alpha_learned[j][: len(labels)]):
            if labels[i] in labels:
                fig = pd.Series(r_alpha).plot(
                    style=".",
                    color=colors[i],
                    markersize=5,
                    label=labels[i] + f" (n={nb_train_samples})",
                )

        plt.legend()
        plt.ylabel(r"$\alpha$-quantile $r_{\alpha}(x_0)$")
        plt.xlabel(r"$\alpha$")
        plt.title(
            r"Local PIT-distribution of the flow at $x_0$ = " + str(x_eval.numpy())
        )
        plt.show()


def multi_pp_plots(
    lct_paths,
    x_eval_names,
    param_names,
    pvalues=True,
    title=r"PP-plot at $x_0$",
    xlabel=r"$\alpha$",
    ylabel=r"$r_{i,\alpha}(x_0)$",
    confidence_int=False,
):

    for i, x_eval_name in enumerate(x_eval_names):
        for k in range(len(lct_paths)):
            lct_dict = torch.load(lct_paths[k][i])

            r_alpha_learned = lct_dict["r_alpha_learned"]
            hmean_pvalue = None
            labels = param_names
            if pvalues:
                pvalues = lct_dict["pvalues"]
                labels = [
                    param_names[i - 1] + f", pvalue={pvalues[f'dim_{i}']}"
                    for i in range(1, len(param_names) + 1)
                ]
                hmean_pvalue = np.round(hmean(list(pvalues.values())), decimals=3)

            PP_plot_1D(
                PIT_values=None,
                alphas=np.linspace(0, 1, 21),
                r_alpha_learned=[
                    r_alpha_learned["dim_1"],
                    r_alpha_learned["dim_2"],
                    r_alpha_learned["dim_3"],
                    r_alpha_learned["dim_4"],
                ],
                colors=["orange", "red"],
                colors_r_alpha=["orange", "red", "purple", "blue"],
                labels=labels,
                title=title + f"{x_eval_name}",
                pvalue=hmean_pvalue,
                xlabel=xlabel,
                ylabel=ylabel,
                confidence_int=confidence_int,
            )


def sbc_plot(
    sbc_ranks,
    colors,
    labels,
    alphas=np.linspace(0, 1, 100),
    confidence_int=True,
    title="SBC",
):
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["mathtext.fontset"] = "cm"

    lims = [np.min([0, 0]), np.max([1, 1])]
    plt.plot(lims, lims, "--", color="black", alpha=0.75)

    for i in range(len(sbc_ranks[0])):
        sbc_cdf = np.histogram(sbc_ranks[:, i], bins=len(alphas))[0].cumsum()
        plt.plot(alphas, sbc_cdf / sbc_cdf.max(), color=colors[i], label=labels[i])

    if confidence_int:
        # Construct uniform histogram.
        N = len(sbc_ranks)
        nbins = len(alphas)
        hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
        hbb = hb.cumsum() / hb.sum()
        # avoid last value being exactly 1
        hbb[-1] -= 1e-9

        lower = [binom(N, p=p).ppf(0.05/2) for p in hbb]
        upper = [binom(N, p=p).ppf(1-0.05/2) for p in hbb]

        # Plot grey area with expected ECDF.
        plt.fill_between(
            x=np.linspace(0, 1, nbins),
            y1=np.repeat(lower / np.max(lower), 1),
            y2=np.repeat(upper / np.max(lower), 1),
            color="grey",
            alpha=0.3,
        )

    plt.ylabel("empirical CDF", fontsize=15)
    plt.xlabel("ranks", fontsize=15)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.show()
