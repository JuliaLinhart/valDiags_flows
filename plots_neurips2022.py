import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import binom

from diagnostics.pp_plots import PP_vals


def multi_global_consistency(
    multi_PIT_values,
    alphas,
    sbc_ranks,
    labels,
    colors,
    ylabel_pit=r"empirical $r_{i,\alpha}$",
):
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["mathtext.fontset"] = "cm"

    # plot identity function
    fig = plt.figure()
    lims = [np.min([0, 0]), np.max([1, 1])]
    plt.plot(lims, lims, "--", color="black", alpha=0.75)

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
    plt.fill_between(
        x=np.linspace(0, 1, nbins),
        y1=np.repeat(lower / np.max(lower), 1),
        y2=np.repeat(upper / np.max(lower), 1),
        color="grey",
        alpha=0.3,
    )

    for i, Z in enumerate(multi_PIT_values):
        # compute quantiles P_{target}(PIT_values <= alpha)
        pp_vals = PP_vals(Z, alphas)
        # Plot the quantiles as a function of alpha
        plt.plot(alphas, pp_vals, color=colors[i], label=labels[i])

    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles)
    plt.ylabel(ylabel_pit, fontsize=15)
    plt.xlabel(r"$\alpha$", fontsize=15)
    plt.title("Global PIT", fontsize=18)
