import numpy as np
import matplotlib.pyplot as plt

METHODS_DICT = {
    "vanilla C2ST": {
        "test_name": "c2st",
        "t_stat_name": "accuracy",
        "color": "grey",
        "linestyle": "-",
        "marker": "o",
    },
    "vanilla C2ST (NF)": {
        "test_name": "c2st_nf",
        "t_stat_name": "accuracy",
        "color": "grey",
        "linestyle": "--",
        "marker": "*",
    },
    "Reg-C2ST": {
        "test_name": "c2st",
        "t_stat_name": "mse",
        "color": "blue",
        "linestyle": "-",
        "marker": "o",
    },
    "Reg-L-C2ST": {
        "test_name": "lc2st",
        "t_stat_name": "mse",
        "color": "orange",
        "linestyle": "-",
        "marker": "o",
    },
    "Reg-L-C2ST (NF)": {
        "test_name": "lc2st_nf",
        "t_stat_name": "mse",
        "color": "orange",
        "linestyle": "--",
        "marker": "*",
    },
    "Reg-L-C2ST (NF-perm)": {
        "test_name": "lc2st_nf_perm",
        "t_stat_name": "mse",
        "color": "darkorange",
        "linestyle": "-.",
        "marker": "*",
    },
    "Max-L-C2ST": {
        "test_name": "lc2st",
        "t_stat_name": "div",
        "color": "red",
        "linestyle": "-",
        "marker": "o",
    },
    "Max-L-C2ST (NF)": {
        "test_name": "lc2st_nf",
        "t_stat_name": "div",
        "color": "red",
        "linestyle": "--",
        "marker": "*",
    },
    "Max-L-C2ST (NF-perm)": {
        "test_name": "lc2st_nf_perm",
        "t_stat_name": "div",
        "color": "darkred",
        "linestyle": "-.",
        "marker": "*",
    },
    "L-HPD": {
        "test_name": "lhpd",
        "t_stat_name": "mse",
        "color": "lightgreen",
        "linestyle": "-",
        "marker": "x",
    },
}


def plot_sbibm_results_n_train(
    avg_results,
    train_runtime,
    methods,
    n_train_list,
    n_cal,
    fig_path,
    t_stat_ext="t_all",
):
    for k in avg_results["c2st"].keys():
        for method in methods:
            if (
                "t_stat" in k and "perm" in method
            ):  # skip permuted test statistics (same as non-permuted)
                continue
            test_name = METHODS_DICT[method]["test_name"]
            t_stat_name = METHODS_DICT[method]["t_stat_name"]
            plt.plot(
                np.arange(len(n_train_list)),
                avg_results[test_name][k][t_stat_name],
                label=method,
                color=METHODS_DICT[method]["color"],
                linestyle=METHODS_DICT[method]["linestyle"],
                marker=METHODS_DICT[method]["marker"],
            )
            if "mean" in k:
                k_std = k[:-5] + "_std"
                plt.fill_between(
                    np.arange(len(n_train_list)),
                    np.array(avg_results[test_name][k][t_stat_name])
                    - np.array(avg_results[test_name][k_std][t_stat_name]),
                    np.array(avg_results[test_name][k][t_stat_name])
                    + np.array(avg_results[test_name][k_std][t_stat_name]),
                    alpha=0.2,
                    color=METHODS_DICT[method]["color"],
                )
                # k = k[:-5]
        if "p_value" in k:
            plt.plot(
                np.arange(len(n_train_list)),
                np.ones(len(n_train_list)) * 0.05,
                "--",
                color="black",
                label="alpha-level",
            )
        if "t_stat" in k:
            plt.plot(
                np.arange(len(n_train_list)),
                np.ones(len(n_train_list)) * 0.5,
                "--",
                color="black",
                label=r"$\mathcal{H}_0$",
            )
        if "run_time" in k:
            for m in methods:
                if "L" in m:
                    plt.plot(
                        np.arange(len(n_train_list)),
                        train_runtime[METHODS_DICT[m]["test_name"]],
                        label=f"{m} (pre-train)",
                        color="black",
                        linestyle=METHODS_DICT[m]["linestyle"],
                        marker=METHODS_DICT[m]["marker"],
                    )

        if "std" not in k:
            plt.legend()
            plt.xticks(np.arange(len(n_train_list)), n_train_list)
            plt.xlabel("N_train")
            plt.ylabel(k)
            plt.savefig(fig_path / f"{k}_{t_stat_ext}_ntrain_n_cal_{n_cal}.pdf")
            plt.show()
        else:
            plt.close()
