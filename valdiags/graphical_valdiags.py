# Graphical diagnostics for the validation of conditional density estimators,
# in particular in the context of SBI. They help interpret the results of the
# following tests:
#
# 1. Multivaraite (Local) Coverage Tests based on PIT-values for conditional Normalizing Flows
#    - [Zhao et al. (2021)](https://arxiv.org/abs/2102.10473))
# 2. (Local) Classifier Two Sample Test (C2ST) (can be used for any SBI-algorithm)
#    - [Lopez et al. (2016)](https://arxiv.org/abs/1602.05336))
#    - [Lee et al. (2018)](https://arxiv.org/abs/1812.08927))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributions as D
from matplotlib.lines import Line2D

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


# ==== 1. (Local) Coverage Tests based on PIT-values for conditional Normalizing Flows ====


# CDF function of a (conditional) flow (nflows) evaluated in x: F_{Q|context}(x)


def cdf_flow(x, context, flow, base_dist=D.Normal(0, 1)):
    return base_dist.cdf(flow._transform(x, context=context)[0])


# PIT values for 1D and multi-D target data


def cde_pit_values(samples_theta, samples_x, flow, local=False):
    """Compute global (resp. local) PIT-values for univariate (1D) target data,
    on N samples (theta, x) from the joint (resp. (theta, x_0) from the true posterior at x_0).

    inputs:
    - samples_theta: torch.Tensor, size: (N, dim)
    - samples_x: torch.Tensor, size: (N, nb_features, 1)
    - flow: class based on pyknos.nflows.distributions.base.Distribution
        Pytorch neural network defining our Normalizing Flow,
        hence conditional (posterior) density estimator.

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
            cdf_flow(samples_theta, context=samples_x, flow=flow).detach().numpy()
        )
    else:
        pit_values = np.array(
            [
                cdf_flow(samples_theta[i][None], context=x, flow=flow).detach().numpy()
                for i, x in enumerate(samples_x)
            ]
        )
    return pit_values


def multi_cde_pit_values(
    samples_theta,
    samples_x,
    flow,
):
    """Compute PIT-values for multivaraite target data,
    computed on N samples (theta, x) from the joint.

    inputs:
    - samples_theta: torch.Tensor, size: (N, dim)
    - samples_x: torch.Tensor, size: (N, nb_features, 1)
    - flow: class based on pyknos.nflows.distributions.base.Distribution
        Pytorch neural network defining our Normalizing Flow,
        hence conditional (posterior) density estimator.

    outputs:
    - pit_values: list of length dim with numpy arrays of size (N, )
        List of pit-values for each dimension.
    """
    dim = samples_theta.shape[-1]
    pit_values = []
    for i in range(dim):
        conditional_transform_1d = (
            D.Normal(0, 1)
            .cdf(flow._transform(samples_theta, context=samples_x)[0][:, i])
            .detach()
            .numpy()
        )
        pit_values.append(conditional_transform_1d)
    return pit_values


# same functions adaped to flows from zuko


def cdf_flow_zuko_1d(target, context, flow, base_dist=D.Normal(0, 1)):
    return base_dist.cdf(flow(context).transform(target))


def cde_pit_values_zuko(target, context, flow):
    pit_values = cdf_flow_zuko_1d(target, context, flow).detach().numpy()
    return pit_values


def multi_cde_pit_values_zuko(
    samples_theta,
    samples_x,
    flow,
):
    """Compute PIT-values for multivaraite target data,
    computed on N samples (theta, x) from the joint.

    inputs:
    - samples_theta: torch.Tensor, size: (N, dim)
    - samples_x: torch.Tensor, size: (N, nb_features, 1)
    - flow: class based on zuko.distributions.FlowModule
        Pytorch neural network defining our Normalizing Flow,
        hence conditional (posterior) density estimator.

    outputs:
    - pit_values: list of length dim with numpy arrays of size (N, )
        List of pit-values for each dimension.
    """
    dim = samples_theta.shape[-1]
    pit_values = []
    for i in range(dim):
        conditional_transform_1d = (
            D.Normal(0, 1)
            .cdf(flow(samples_x).transform(samples_theta)[:, i])
            .detach()
            .numpy()
        )
        pit_values.append(conditional_transform_1d)
    return pit_values


# PP-plot for 1D and multi-D target data


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
    conf_alpha=0.05,
    N=None,
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
    - conf_alpha: alpha level of the (1-conf-alpha)-confidence region.
        Default is 0.05, for a confidence level of 0.95.
    """
    # plot identity function
    fig = plt.figure()
    lims = [np.min([0, 0]), np.max([1, 1])]
    plt.plot(lims, lims, "--", color="black", alpha=0.75)

    if PIT_values is not None:
        N = len(PIT_values[0])
        for i, Z in enumerate(PIT_values):
            # compute quantiles P_{target}(PIT_values <= alpha)
            pp_vals = PP_vals(Z, alphas)
            # Plot the quantiles as a function of alpha
            plt.plot(alphas, pp_vals, color=colors[i], label=labels[i])

    handles_new = []
    if r_alpha_learned is not None:
        N = len(r_alpha_learned[0])
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
        N = N
        confidence_region_null(alphas=alphas, N=N, conf_alpha=conf_alpha)

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
    conf_alpha=0.05,
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
    - conf_alpha: alpha level of the (1-conf-alpha)-confidence region.
        Default is 0.05, for a confidence level of 0.95.
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
                conf_alpha=conf_alpha,
            )


# PP-plot for SBC validation method:


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


# ==== 2. (Local) Classifier Two Sample Test (C2ST) ====

# PP-plot of clasifier predicted class probabilities


def pp_plot_c2st(probas, probas_null, labels, colors):
    alphas = np.linspace(0, 1, 100)
    pp_vals_dirac = PP_vals([0.5] * len(probas), alphas)
    plt.plot(alphas, pp_vals_dirac, "--", color="black")

    pp_vals_null = {}
    for t in range(len(probas_null)):
        pp_vals_null[t] = pd.Series(PP_vals(probas_null[t], alphas))

    low_null = pd.DataFrame(pp_vals_null).quantile(0.05 / 2, axis=1)
    up_null = pd.DataFrame(pp_vals_null).quantile(1 - 0.05 / 2, axis=1)
    plt.fill_between(
        alphas,
        low_null,
        up_null,
        color="grey",
        alpha=0.3,
        label="95% confidence region",
    )

    for p, l, c in zip(probas, labels, colors):
        pp_vals = pd.Series(PP_vals(p, alphas))
        plt.plot(alphas, pp_vals, label=l, color=c)

    plt.legend()


# Interpretability plots for C2ST: regions of high/low predicted class probabilities


def z_space_with_proba_intensity(
    probas, probas_null, P_eval, theta_space=None, dim=1, thresholding=False
):
    df = pd.DataFrame({"probas": probas})

    # define low and high thresholds w.r.t to null (95% confidence region)
    low = np.quantile(np.mean(probas_null, axis=0), q=0.05)
    high = np.quantile(np.mean(probas_null, axis=0), q=0.95)
    # high/low proba regions for bad NF
    df["intensity"] = ["uncertain"] * len(df)
    df.loc[df["probas"] > high, "intensity"] = (
        r"high ($p \geq$ " + f"{np.round(high,2)})"
    )
    df.loc[df["probas"] < low, "intensity"] = r"low ($p \leq$ " + f"{np.round(low,2)})"

    if dim == 1:
        from matplotlib import cm

        df["z"] = P_eval[:, 0]
        values = "z"
        xlabel = r"$z$"
        x = df.z
        if theta_space is not None:
            df["theta"] = theta_space
            values = "theta"
            xlabel = r"$\theta$"
            x = df.theta

        if thresholding:
            df.pivot(columns="intensity", values=values).plot.hist(
                bins=50, color=["red", "blue", "grey"], alpha=0.3
            )
        else:
            _, bins, patches = plt.hist(x, 50, density=True, color="green")
            bins[-1] = 10
            df["bins"] = np.select([x <= i for i in bins[1:]], list(range(50)), 1000)

            weights = df.groupby(["bins"]).mean().probas
            id = list(set(range(50)) - set(df.bins))
            patches = np.delete(patches, id)

            cmap = plt.cm.get_cmap("bwr")
            for c, p in zip(weights, patches):
                plt.setp(p, "facecolor", cmap(c))
            plt.colorbar(
                cm.ScalarMappable(cmap=cmap),
                label=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$",
            )
        plt.xlabel(xlabel)

    elif dim == 2:
        df["z_1"] = P_eval[:, 0]
        df["z_2"] = P_eval[:, 1]
        x, y = df.z_1, df.z_2
        xlabel = r"$Z_1$"
        ylabel = r"$Z_2$"
        if theta_space is not None:
            df["theta_1"] = theta_space[:, 0]
            df["theta_2"] = theta_space[:, 1]
            x, y = df.theta_1, df.theta_2
            xlabel = r"$\Theta_1 = T_{\phi,1}(Z; x_0)$"
            ylabel = r"$\Theta_2 = T_{\phi,2}(Z; x_0)$"
        if not thresholding:
            plt.scatter(x, y, c=df.probas, cmap="bwr", alpha=0.3)
            plt.colorbar(label=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$")
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
                if theta_space is not None:
                    x = group.theta_1
                    y = group.theta_2
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="",
                    alpha=0.3,
                    label=name,
                    color=cdict[name],
                )
            plt.legend(title=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    else:
        print("Not implemented.")


def eval_space_with_proba_intensity(
    probas, probas_null, P_eval, dim=1, z_space=True, thresholding=False
):
    df = pd.DataFrame({"probas": probas})

    # define low and high thresholds w.r.t to null (95% confidence region)
    low = np.quantile(np.mean(probas_null, axis=0), q=0.05)
    high = np.quantile(np.mean(probas_null, axis=0), q=0.95)
    # high/low proba regions for bad NF
    df["intensity"] = ["uncertain"] * len(df)
    df.loc[df["probas"] > high, "intensity"] = (
        r"high ($p \geq$ " + f"{np.round(high,2)})"
    )
    df.loc[df["probas"] < low, "intensity"] = r"low ($p \leq$ " + f"{np.round(low,2)})"

    if dim == 1:
        from matplotlib import cm

        df["z"] = P_eval[:, 0]

        if thresholding:
            df.pivot(columns="intensity", values="z").plot.hist(
                bins=50, color=["red", "blue", "grey"], alpha=0.3
            )
        else:
            _, bins, patches = plt.hist(df.z, 50, density=True, color="green")
            bins[-1] = 10
            df["bins"] = np.select([x <= i for i in bins[1:]], list(range(50)), 1000)

            weights = df.groupby(["bins"]).mean().probas
            id = list(set(range(50)) - set(df.bins))
            patches = np.delete(patches, id)

            cmap = plt.cm.get_cmap("bwr")
            for c, p in zip(weights, patches):
                plt.setp(p, "facecolor", cmap(c))
            plt.colorbar(
                cm.ScalarMappable(cmap=cmap),
                label=r"$\hat{p}(Z\sim\mathcal{N}(0,1)\mid x_0)$",
            )
        if z_space:
            xlabel = r"$z$"
        else:
            xlabel = r"$\theta$"
        plt.xlabel(xlabel)

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
            plt.scatter(df.z_1, df.z_2, c=df.probas, cmap="bwr", alpha=0.3)
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

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    else:
        print("Not implemented.")
