from turtle import color
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch.distributions as D
import seaborn as sns


def plot_pdf_1D(x_samples, x_i, x_f, target_dist=None, flows=None, context=None):
    """`flows` must be a dict with the plot-label as key_name and 3 elements: (flow, context, plot-color)."""
    eval_x = torch.linspace(x_i, x_f, 100).reshape(-1, 1)
    if context is not None:
        context = context.repeat(eval_x.size(0), 1)

    _ = plt.figure(figsize=(6, 2))
    plt.plot(
        x_samples,
        np.zeros_like(x_samples),
        "bx",
        alpha=0.5,
        markerfacecolor="none",
        markersize=6,
    )
    labels = ["Samples"]

    if target_dist is not None:
        labels += ["True"]
        try:
            if context is not None:
                p_x_true = torch.exp(target_dist.log_prob(eval_x, context=context))
            else:
                p_x_true = torch.exp(target_dist.log_prob(eval_x))

            plt.plot(eval_x.numpy(), p_x_true.detach().numpy(), "--", color="blue")
        except ValueError:  # in case of exponential distribution
            eval_x_pos = torch.linspace(0.01, x_f).reshape(-1, 1)
            p_x_true = torch.exp(target_dist.log_prob(eval_x_pos, context=context))
            plt.plot(eval_x_pos.numpy(), p_x_true.detach().numpy(), "--", color="blue")

    if flows is not None:
        for flow, ct, col in list(flows.values()):
            if ct is not None:
                ct = ct.repeat(eval_x.size(0), 1)
                p_x_learned = torch.exp(flow.log_prob(eval_x, context=ct))
            else:
                p_x_learned = torch.exp(flow.log_prob(eval_x))

            plt.plot(eval_x.numpy(), p_x_learned.detach().numpy(), color=col)
        labels += list(flows.keys())

    plt.legend(labels)

    _ = plt.xlim([x_i, x_f])
    _ = plt.ylim([-0.12, 1.2])

    plt.show()


def plot_cdf_1D(target_dist, flow, x_i, x_f, base_dist=D.Normal(0, 1)):
    x_eval = torch.linspace(x_i, x_f).reshape(-1, 1)

    try:
        cdf_estimate = cdf_flow(x_eval, flow, base_dist)
        plt.plot(x_eval, target_dist.cdf(x_eval), color="blue")
    except ValueError:  # in case of exponential distribution
        x_eval_pos = torch.linspace(0.01, x_f).reshape(-1, 1)
        cdf_estimate = cdf_flow(x_eval_pos, flow, base_dist)
        plt.plot(x_eval, target_dist.cdf(x_eval_pos), color="blue")

    plt.plot(x_eval, cdf_estimate.detach().numpy(), color="orange")
    plt.legend(["True", "Learned"], loc="upper left")
    plt.show()


def PP_plot_1D(target_dist, x_samples, flow, base_dist=D.Normal(0, 1)):
    alphas = np.linspace(0, 1)
    cdf = lambda x: cdf_flow(x, flow, base_dist)
    z_true = PP_vals(target_dist.cdf, x_samples, alphas)
    z_estimate = PP_vals(cdf, x_samples, alphas)
    plt.plot(alphas, z_true, color="blue")
    plt.plot(alphas, z_estimate, color="orange")
    plt.legend(["True", "Learned"], loc="upper left")
    plt.show()


def PP_vals(cdf, x_samples, alphas):
    """Return estimated vs true quantiles between pdist and samples.
    inputs:
    - cdf: cdf of the estimate (flow)
    - x_samples: torch.tensor of samples of the target distribution
    - alphas: array of values to evaluate the PP-vals
    """
    F = cdf(x_samples).detach().numpy()
    z = [np.mean(F < alpha) for alpha in alphas]
    return z


def cdf_flow(x, flow, base_dist=D.Normal(0, 1)):
    """Return the cdf of the flow evaluated in x
    input:
    - x: torch.tensor
    - flow: nflows.Flow
    - base_dist: torch.distributions object
    """
    return base_dist.cdf(flow._transform(x)[0])


from matplotlib.lines import Line2D

### 2D ###
# 2D distributions: Evaluate the learnt transformation by plotting the learnt 2D pdf-contours
# against the true pdf-contours and the training samples (drawn from the true distribution)
def plot_2d_pdf_contours(
    target_dist, flows, x_samples=None, title=None, n=500, gaussian=False
):
    """`flows` must be a dict with the plot-label as key_name and 3 elements: 
    (flow, context, plot-color).
    """

    if x_samples is not None:
        plt.scatter(
            x=x_samples[:, 0], y=x_samples[:, 1], color="blue", label="Samples"
        )  # Plot training samples
    x_true = target_dist.sample((n,))  # Sample from groundtruth
    sns.kdeplot(x=x_true[:, 0], y=x_true[:, 1], color="blue")  # Plot true distribution
    handles_ext = [Line2D([0], [0], color="blue", label="True")]

    labels = list(flows.keys())
    for i, (flow, context, col) in enumerate(list(flows.values())):
        x_learned = (
            flow.sample(n, context=context).detach().numpy()
        )  # Sample from learned
        if context is not None:
            x_learned = x_learned[0]
        sns.kdeplot(
            x=x_learned[:, 0], y=x_learned[:, 1], color=col
        )  # Plot learned distribution
        handles_ext.append(Line2D([0], [0], color=col, label=labels[i]))

    handles, _ = plt.gca().get_legend_handles_labels()

    handles.extend(handles_ext)

    plt.legend(handles=handles)

    if gaussian:
        means_learned = np.mean(x_learned, axis=0)  # Learned mean
        plt.scatter(x=target_dist.mean[0], y=target_dist.mean[1], color="cyan")
        plt.scatter(x=means_learned[0], y=means_learned[1], color="magenta")

    plt.title(title)
    plt.show()


def get_grid(low, high, n_samples=1000):
    """Return grid for evaluation.

    Parameters
    ----------
    n_samples : int
        Number of samples, number of points in linspace.

    Returns
    -------
    samples : torch.Tensor, shape (n_samples*n_samples, 2)
    XX, YY : torch.Tensor, shape (n_samples, n_samples)
    """
    t = torch.linspace(low, high, n_samples)
    XX, YY = torch.meshgrid(t, t)
    samples = torch.cat([XX.reshape(-1, 1), YY.reshape(-1, 1)], dim=1)

    return samples, XX, YY


def plot_2d_pdf_on_grid(pdf, low, high):
    """Plot analytic and learned posterior on grid returned by the function get_grid() above.

    Parameters
    ----------
    pdf : pdf function
        Computes the probability of a given point
    low, high: lower and upper border of the 2d-grid

    Returns
    -------
    fig, ax : plt.figure and axes
    """
    samples, XX, YY = get_grid(low=low, high=high)
    probas_on_grid = pdf(samples)
    # Plot true distribution
    plt.contour(
        XX.numpy(),
        YY.numpy(),
        probas_on_grid.reshape(1000, 1000),
        cmap="Blues",
        zorder=0,
    )
    # plt.show()

def plot_pairgrid_with_groundtruth(
    posteriors, theta_gt, color_dict, handles, context, n_samples=10000, title=None, fixed_gain=False
):
    plt.rcParams["figure.figsize"] = (9, 9)
    # plt.rcParams["legend.fontsize"] = 23.0
    # plt.rcParams["xtick.labelsize"] = 23.0
    # plt.rcParams["ytick.labelsize"] = 23.0
    # plt.rcParams["axes.labelsize"] = 23.0
    # plt.rcParams["font.size"] = 23.0
    # plt.rcParams["axes.titlesize"] = 27.0

    dim = theta_gt.shape[-1]
    modes = list(posteriors.keys())
    dfs = []
    for n in range(len(posteriors)):
        posterior = posteriors[modes[n]]
        samples = posterior.sample(n_samples, context=context[modes[n]])
        df = pd.DataFrame(
            samples.detach().numpy(), columns=[r"$C$", r"$\mu$", r"$\sigma$", r"$g$"][:dim]
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

    if not fixed_gain:

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
            C, mu, sigma = gt[:3]
            # plot points
            g.axes[1][0].scatter(C, mu, color="black", zorder=2, s=8)
            g.axes[2][0].scatter(C, sigma, color="black", zorder=2, s=8)
            g.axes[2][1].scatter(mu, sigma, color="black", zorder=2, s=8)
            # plot dirac
            g.axes[0][0].axvline(x=C, ls="--", c="black", linewidth=1)
            g.axes[1][1].axvline(x=mu, ls="--", c="black", linewidth=1)
            g.axes[2][2].axvline(x=sigma, ls="--", c="black", linewidth=1)

            if not fixed_gain:
                gain = gt[3]
                # plot points
                g.axes[3][0].scatter(C, gain, color="black", zorder=2, s=8)
                g.axes[3][1].scatter(mu, gain, color="black", zorder=2, s=8)
                g.axes[3][2].scatter(sigma, gain, color="black", zorder=2, s=8)
                # plot dirac
                g.axes[3][3].axvline(x=gain, ls="--", c="black", linewidth=1)

    plt.legend(
        handles=handles,
        title=title,
        bbox_to_anchor=(1.1, 3.3),
        # loc="upper right",
    )
    g.fig.suptitle("Local pair-plots", y=1.02)

    return g
