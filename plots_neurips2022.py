import matplotlib as mpl
import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, fontsizes

plt.rcParams.update(fontsizes.neurips2022())
plt.rcParams.update(fonts.neurips2022())

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
    ylabel_sbc="empirical CDF",
):
    plt.rcParams.update(
        figsizes.neurips2022(nrows=1, ncols=2, height_to_width_ratio=0.8)
    )
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    for ax in axes:
        # plot identity function
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
