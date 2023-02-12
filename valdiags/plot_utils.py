import matplotlib.pyplot as plt


def plot_distributions(dist_list, colors, labels, dim=1):
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
            plt.scatter(
                d[:, 0],
                d[:, 1],
                color=c,
                alpha=0.3,
                label=l,
            )

    else:
        print("Not implemented.")
