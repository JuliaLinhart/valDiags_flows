import matplotlib.pyplot as plt

from lampe.plots import corner


def multi_corner_plots(samples_list, legends, colors, title, **kwargs):
    fig = None
    for s, l, c in zip(samples_list, legends, colors):
        fig = corner(s, legend=l, color=c, figure=fig, smooth=2, **kwargs)
        plt.title(title)


def plot_distributions(dist_list, colors, labels, dim=1, hist=False):
    if dim == 1:
        for d, c, l in zip(dist_list, colors, labels):
            plt.hist(
                d,
                bins=100,
                color=c,
                alpha=0.3,
                density=True,
                label=l,
            )

    elif dim == 2:
        for d, c, l in zip(dist_list, colors, labels):
            if not hist:
                plt.scatter(
                    d[:, 0],
                    d[:, 1],
                    color=c,
                    alpha=0.3,
                    label=l,
                )
            else:
                plt.hist2d(
                    d[:, 0].numpy(),
                    d[:, 1].numpy(),
                    bins=100,
                    cmap=c,
                    alpha=0.7,
                    density=True,
                    label=l,
                )
    else:
        print("Not implemented.")
