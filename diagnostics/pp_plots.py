import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import torch
import torch.distributions as D
from matplotlib.lines import Line2D

from scipy.stats import hmean, binom

import sys

sys.path.append("../")
from nde.flows import cdf_flow
from data.feature_transforms import identity


def multi_cde_pit_values(
    samples_theta, samples_x, flow, feature_transform=identity,
):
    """ Compute PIT-values for multivaraite target data,
    computed on N samples (theta, x) from the joint.

    inputs:
    - samples_theta: torch.Tensor, size: (N, dim)
    - samples_x: torch.Tensor, size: (N, nb_features, 1)
    - flow: class based on pyknos.nflows.distributions.base.Distribution
        Pytorch neural network defining our Normalizing Flow, 
        hence conditional (posterior) density estimator.
    - feature_transform: function 
        Default is "identity": no feature transform on x.
    
    outputs:
    - pit_values: list of length dim with numpy arrays of size (N, )
        List of pit-values for each dimension.
    """
    dim = samples_theta.shape[-1]
    pit_values = []
    for i in range(dim):
        conditional_transform_1d = (
            D.Normal(0, 1)
            .cdf(
                flow._transform(samples_theta, context=feature_transform(samples_x))[0][
                    :, i
                ]
            )
            .detach()
            .numpy()
        )
        pit_values.append(conditional_transform_1d)
    return pit_values


def cde_pit_values(
    samples_theta, samples_x, flow, feature_transform=identity, local=False
):
    """ Compute global (resp. local) PIT-values for univariate (1D) target data,
    on N samples (theta, x) from the joint (resp. (theta, x_0) from the true posterior at x_0).

    inputs:
    - samples_theta: torch.Tensor, size: (N, dim)
    - samples_x: torch.Tensor, size: (N, nb_features, 1)
    - flow: class based on pyknos.nflows.distributions.base.Distribution
        Pytorch neural network defining our Normalizing Flow, 
        hence conditional (posterior) density estimator.
    - feature_transform: function 
        Default is "identity": no feature transform on x.
    -local: bool
        If True, compute the true local PIT values:
        - samples_theta are true posterior samples at x_0,
        - samples_x is a tensor of x_0 repeated N times.
        Default is False.
    
    outputs:
    - pit_values: numpy arrays, size: (N, )
    """
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
    """Compute the 1D PIT distribution: the c.d.f. of the 1D PIT in every alpha: 
        P(PIT <= alpha)
    where P is empirically approximated.

    inputs:
    - PIT_values: numpy array, size: (N,)
        1D PIT-values of a given estimator (e.g. output of "cde_pit_values")
    - alphas: numpy array, size: (K,) 
        Values to evaluate the PIT-c.d.f in.

    outputs:
    - pp_vals: list of length K
        List of 1D PIT-c.d.f values for every alpha.
    """
    pp_vals = [np.mean(PIT_values <= alpha) for alpha in alphas]
    return pp_vals


def PP_plot_1D(
    PIT_values,
    alphas,
    r_alpha_learned=None,
    colors=["blue"],
    colors_r_alpha=["red"],
    labels=["Target"],
    title=r"Local PIT-distribution at $x_0$",
    ylabel=r"$r_{\alpha}(x_0)$",
    xlabel=r"$\alpha$",
    pvalue=None,
    confidence_int=False,
):
    """1D PP-plot: c.d.f of the 1D PIT vs. c.d.f of the uniform distribution.
        It shows the deviation to the identity function and thus allows to 
        visualize deviances of the estimated distribution w.r.t the target  
        distribution and determine their nature (bias / dispersion).

    inputs:
    - PIT_values: numpy array, size: (N,)
        1D PIT-values of a given estimator (e.g. output of "cde_pit_values")
    - alphas: numpy array, size: (K,) 
        Values to evaluate the PIT-c.d.f in.
    - r_alpha_learned: dict, keys: alphas
        Regressed local c.d.f values (for local PIT).
        Output from "localPIT_regression..." defined in localPIT_regression.py.
        Default is None: we use 1D PIT_values.
    - colors: string
        Color for the empirical (global or local) PIT-distribution.
        Default is blue.
    - colors_r_alpha: string
        Color for the regressedl local PIT-distribution.
        Default is red.
    - labels: list of strings
        Labels for the different distribution-plots that will appear in the legend.
    - title: string
        Title of the plot.
    - ylabel: string
        Label of the y-axis.
    - xlabel: string
        Label of the x-axis.
    - pvalue: float
        pvalue of the corresponding test.
        Default is None.
    - confidence_int: bool
        Whether to show the confidence region (acceptance of the null hypothesis).
        Default is False.
    """
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

        lower = [binom(N, p=p).ppf(0.05 / 2) for p in hbb]
        upper = [binom(N, p=p).ppf(1 - 0.05 / 2) for p in hbb]

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


def multi_pp_plots(
    lct_paths,
    x_eval_names,
    param_names,
    pvalues=False,
    title=r"PP-plot at $x_0$",
    xlabel=r"$\alpha$",
    ylabel=r"$r_{i,\alpha}(x_0)$",
    confidence_int=False,
):
    """PP-plot for multivariate target data: 
    C.d.f of every 1D element of the multivariate PIT (one for each dimension) 
    vs. c.d.f of the uniform distribution.

    inputs:
    - lct_paths: list of lists of strings
        One list for each regression method that includes paths for 
        lct results of every x_eval. The lct-results are dicts as outputted 
        by the function "multivariate_lct" defined in multi_local_test.py.
    - x_eval_names: list of strings
        Names of the observations that will appear in the title of the plot.
    - param_names: list of strings
        Names of the simulator parameters that will appear in the legend.
    pvalues: bool
        Whether to show the (harmonic mean) pvalue of the corresponding 
        multivariate LCT.
        Default is False.
    - title: string
        Title of the plot.
    - xlabel: string
        Label of the x-axis.
    - ylabel: string
        Label of the y-axis.
    - confidence_int: bool
        Whether to show the confidence region (acceptance of the null hypothesis).
        Default is False.
    """
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
    """ PP-plot for SBC validation method: 
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
        nbins = len(alphas)
        hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
        hbb = hb.cumsum() / hb.sum()
        # avoid last value being exactly 1
        hbb[-1] -= 1e-9

        lower = [binom(N, p=p).ppf(0.05 / 2) for p in hbb]
        upper = [binom(N, p=p).ppf(1 - 0.05 / 2) for p in hbb]

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


# ======================================================================
# Script for local PIT regression method comparison
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
