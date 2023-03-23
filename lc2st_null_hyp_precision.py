# ==== Presicion of test under the null hypothesis for L-C2ST ====

# For different dataset sizes we compute the expexted (over different observations x_0)
# test statistics and stds of the predicted proabilities under the null hypothesis.

# We relax the problem from a dirac to a gaussian distribution with std of 0.05 or 0.1 (smooth dirac):
# - we compute the theoretical test-statistics of those two gaussians
# - we compare the obtained test-statistics and stds of the cross-validation to these theoretical values


import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier

from valdiags.vanillaC2ST import c2st_scores

import time


def eval_null_lc2st(
    x_samples,
    null_dist_samples,
    test_stats=["probas_mean"],
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    clf_name="mlp_base",
    n_samples=1000,
    n_folds=10,
    single_class_eval=True,
):

    scores = {}
    for m in test_stats:
        scores[m] = []

    P, Q = null_dist_samples
    x_samples_shuffled = np.random.permutation(x_samples)

    joint_P_x = np.concatenate([P, x_samples], axis=1)
    joint_Q_x = np.concatenate([Q, x_samples_shuffled], axis=1)

    start = time.time()

    scores = c2st_scores(
        P=joint_P_x,
        Q=joint_Q_x,
        metrics=test_stats + ["probas_std"],
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        cross_val=True,
        n_folds=n_folds,
        single_class_eval=single_class_eval,
    )
    total_cv_time = time.time() - start

    times = [total_cv_time] * n_folds
    n_samples_list = [n_samples] * n_folds
    classifier = [clf_name] * n_folds

    df = pd.DataFrame(
        {
            f"nb_samples": n_samples_list,
            "total_cv_time": times,
            "classifier": classifier,
        }
    )
    for m in test_stats + ["probas_std"]:
        df[m] = scores[m]

    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    from tasks.toy_examples.data_generators import ConditionalGaussian1d

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )

    from valdiags.graphical_valdiags import PP_vals
    from scipy.stats import wasserstein_distance, norm

    # Theoretical test-stat values for chosen stds

    def smooth_dirac(eps=0.1):
        return norm(loc=0.5, scale=eps)

    alphas = np.linspace(0, 1, 100)
    pp_vals_dirac = PP_vals([0.5] * 1000, alphas)
    for e in [0.05, 0.1]:
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(alphas, [smooth_dirac(eps=e).pdf(x) for x in alphas])

        pp_vals_dirac = PP_vals([0.5] * 1000, alphas)

        test_stats = []
        w_dist = []
        for t in range(10):
            samples = smooth_dirac(eps=e).rvs(1000)
            pp_vals = pd.Series(PP_vals(samples, alphas))
            test_stats.append(((pp_vals - pp_vals_dirac) ** 2).sum() / len(alphas))
            w_dist.append(wasserstein_distance([0.5] * 1000, samples))
            axs[1].plot(alphas, pp_vals_dirac, "--", color="black")
            axs[1].plot(alphas, pp_vals)
        plt.suptitle(
            f"Gauss-std: {e}, mean TV_dist: {np.round(np.mean(test_stats),3)}, mean w_dist: {np.round(np.mean(w_dist),3)}"
        )
        plt.show()

    # experiment set-up
    N_LIST = [1000, 2000, 5000]  # , 10000]

    data_gen = ConditionalGaussian1d()
    x_samples = {}
    for n in N_LIST:
        x_samples[n], ref_samples = data_gen.get_joint_data(n=n)

    null_dist = norm

    # Models
    ndim = x_samples[1000].shape[-1] + ref_samples.shape[-1]
    clf_classes = {
        "mlp_base": MLPClassifier,
        "mlp_sbi": MLPClassifier,
        "rf": RandomForestClassifier,
        "logreg": LogisticRegression,
        "lda": LinearDiscriminantAnalysis,
        "qda": QuadraticDiscriminantAnalysis,
    }
    clf_kwargs_dict = {
        "mlp_base": {"alpha": 0, "max_iter": 25000},
        "mlp_sbi": {
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        },
        "rf": {},
        "logreg": {},
        "lda": {},
        "qda": {},
    }

    # compute the test-statistics and stds under the null hypothesis
    clf_names = ["mlp_base", "mlp_sbi", "rf", "lda", "qda"]
    dfs = []
    for clf in clf_names:
        for n in N_LIST:
            null_dist_samples = [null_dist.sample((n,)), null_dist.sample((n,))]
            df = eval_null_lc2st(
                x_samples[n],
                null_dist_samples,
                test_stats=["w_dist", "TV"],
                n_samples=n,
                n_folds=50,
                clf_class=clf_classes[clf],
                clf_kwargs=clf_kwargs_dict[clf],
                clf_name=clf,
            )
            dfs.append(df)

    # plot the results
    df = pd.concat(dfs, ignore_index=True)
    for T, y1, y2 in zip(["TV", "w_dist"], [0.023, 0.08], [0.012, 0.04]):
        g = sns.relplot(
            data=df,
            x="nb_samples",
            y=T,
            hue="classifier",
            style="classifier",
            kind="line",
        )
        g.map(
            plt.axhline,
            y=y1,
            color=".7",
            dashes=(2, 1),
            zorder=0,
            label="norm with std 0.1",
        )
        g.map(
            plt.axhline,
            y=y2,
            color=".5",
            dashes=(2, 1),
            zorder=0,
            label="norm with std 0.05",
        )
        plt.legend()
        plt.show()

    clf_name = "mlp_base"
    df_clf = df[df["classifier"] == clf_name]
    for T, y1, y2 in zip(["TV", "w_dist"], [0.023, 0.08], [0.012, 0.04]):
        g = sns.relplot(
            data=df_clf,
            x="probas_std",
            y=T,
            hue="nb_samples",
            style="nb_samples",
            kind="scatter",
        )
        g.map(
            plt.axhline,
            y=y1,
            color=".7",
            dashes=(2, 1),
            zorder=0,
            label="norm with std 0.1",
        )
        g.map(
            plt.axhline,
            y=y2,
            color=".5",
            dashes=(2, 1),
            zorder=0,
            label="norm with std 0.05",
        )
        plt.legend()
        plt.show()
        print(
            df_clf[df_clf[T] <= y1]["probas_std"].max(),
            df_clf[df_clf[T] <= y2]["probas_std"].max(),
        )

