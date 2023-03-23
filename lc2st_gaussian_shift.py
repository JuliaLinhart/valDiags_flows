# ==== Gaussian Shift experiment: Classifier choice for L-C2ST ====
#
# We compute the expected lc2st scores (vanilla c2st on the joint) between:
#
#   - normal: P ~ N(0,1)
#   - gaussian with shifted mean and / or scale: Q ~ N(m,1) or Q ~ N(0,s)
#     (with m > 0 and s > 1)
#
# for several classifiers trained on data of the joint disributions
# [z_i, x_i], with labels 0 and 1, where:
#       - z_i | 0 ~ P, z_i | 1 ~ Q
#       - x_i correspond to data from p(x), here independant of z
# (here obtained from conditional 1D gaussian toy-model, in sbi these are simulated observations)
#
# The objectively best discriminator is obtained via LDA (resp. QDA) in the mean-shift
# (resp. scale-shift) experiment.
#
# This enables us to evaluate whether a given the classifier has the same behavior than lda/qda
# for the shift transformations when trained on features concatenated with x and / or how much
# data is needed to obtain the same behavior.


import numpy as np
import time

from valdiags.vanillaC2ST import c2st_scores


def eval_classifier_for_lc2st(
    x_samples,
    ref_samples,
    shifted_samples,
    shifts,
    clf_class,
    clf_kwargs,
    metrics=["probas_mean"],
    n_folds=10,
    single_class_eval=False,
):
    shift_list = []
    scores = {}
    accuracies = []
    for m in metrics:
        scores[m] = []
    times = []
    for s_samples, s in zip(shifted_samples, shifts):

        x_samples_shuffled = np.random.permutation(x_samples)

        joint_P_x = np.concatenate([ref_samples, x_samples], axis=1)
        joint_Q_x = np.concatenate([s_samples, x_samples_shuffled], axis=1)

        start = time.time()

        score = c2st_scores(
            P=joint_P_x,
            Q=joint_Q_x,
            metrics=metrics,
            clf_class=clf_class,
            clf_kwargs=clf_kwargs,
            cross_val=True,
            n_folds=n_folds,
            single_class_eval=single_class_eval,
        )

        for m in metrics:
            scores[m] = np.concatenate([scores[m], score[m]])

        accuracies = np.concatenate([accuracies, score["accuracy"]])

        total_cv_time = time.time() - start

        for _ in range(n_folds):
            shift_list.append(s)
            times.append(total_cv_time)
    return shift_list, scores, accuracies, times


if __name__ == "__main__":
    import torch
    import torch.distributions as D
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )

    from tasks.toy_examples.data_generators import ConditionalGaussian1d

    data_gen = ConditionalGaussian1d()

    N_list = [1000, 2000, 5000]
    mean_shifts = [0, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 5, 10]
    scale_shifts = np.linspace(1, 20, 10)

    # Datasets
    x_samples = {}
    ref_samples = {}
    for n in N_list:
        x_samples[n] = data_gen.sample_x(n)
        ref_samples[n] = D.MultivariateNormal(
            loc=torch.zeros(1), covariance_matrix=torch.eye(1)
        ).rsample((n,))

    # shifted gaussian samples for class 1
    mean_shifted_samples = {}
    scale_shifted_samples = {}
    for n in N_list:
        mean_shifted_samples[n] = [
            D.MultivariateNormal(
                loc=torch.FloatTensor([m]), covariance_matrix=torch.eye(1)
            ).rsample((n,))
            for m in mean_shifts
        ]
        scale_shifted_samples[n] = [
            D.MultivariateNormal(
                loc=torch.zeros(1), covariance_matrix=torch.eye(1) * s
            ).rsample((n,))
            for s in scale_shifts
        ]

    # Models
    ndim = x_samples[1000].shape[-1] + ref_samples[1000].shape[-1]
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

    # mean shift
    clf_names = ["lda", "mlp_sbi", "mlp_base", "rf"]

    dfs_mean = {}
    for n in N_list:
        print(n)
        dfs = []
        for clf_name in clf_names:
            shift_list, scores, accuracies, times = eval_classifier_for_lc2st(
                x_samples[n],
                ref_samples[n],
                shifted_samples=mean_shifted_samples[n],
                shifts=mean_shifts,
                clf_class=clf_classes[clf_name],
                clf_kwargs=clf_kwargs_dict[clf_name],
                metrics=["accuracy", "probas_mean"],
                single_class_eval=True,
            )
            clf_method = [clf_name] * len(shift_list)
            dfs.append(
                pd.DataFrame(
                    {
                        "mean_shift": shift_list,
                        "accuracy": accuracies,
                        "probas_mean": scores["probas_mean"],
                        "total_cv_time": times,
                        "classifier": clf_method,
                    }
                )
            )
        dfs_mean[n] = pd.concat(dfs, ignore_index=True)

    torch.save(
        dfs_mean, "saved_experiments/Gaussian1d/lc2st_eval_clfs/dfs_mean_exp_lc2st.pkl"
    )

    for n in N_list:
        # sns.relplot(
        #     data=dfs_mean[n], x="mean_shift", y="accuracy",
        #     hue="classifier", style="classifier", kind="line",
        # )
        # plt.title(f'NB samples: {n}')
        # plt.show()

        sns.relplot(
            data=dfs_mean[n],
            x="mean_shift",
            y="probas_mean",
            hue="classifier",
            style="classifier",
            kind="line",
        )
        plt.title(f"NB samples: {n}")
        plt.show()

    # scale shift
    clf_names = ["qda", "mlp_sbi", "mlp_base", "rf"]

dfs_scale = {}
for n in N_list:
    dfs = []
    for clf_name in clf_names:
        shift_list, scores, accuracies, times = eval_classifier_for_lc2st(
            x_samples[n],
            ref_samples[n],
            shifted_samples=scale_shifted_samples[n],
            shifts=scale_shifts,
            clf_class=clf_classes[clf_name],
            clf_kwargs=clf_kwargs_dict[clf_name],
            metrics=["accuracy", "probas_mean"],
            single_class_eval=True,
        )
        clf_method = [clf_name] * len(shift_list)
        dfs.append(
            pd.DataFrame(
                {
                    "scale_shift": shift_list,
                    "accuracy": accuracies,
                    "probas_mean": scores["probas_mean"],
                    "total_cv_time": times,
                    "classifier": clf_method,
                }
            )
        )
    dfs_scale[n] = pd.concat(dfs, ignore_index=True)

    torch.save(
        dfs_scale,
        "saved_experiments/Gaussian1d/lc2st_eval_clfs/dfs_scale_exp_lc2st.pkl",
    )

    for n in N_list:
        # sns.relplot(
        #     data=dfs_scale[n], x="scale_shift", y="accuracy",
        #     hue="classifier", style="classifier", kind="line",
        # )
        # plt.title(f'NB samples: {n}')
        # plt.show()

        sns.relplot(
            data=dfs_scale[n],
            x="scale_shift",
            y="probas_mean",
            hue="classifier",
            style="classifier",
            kind="line",
        )
        plt.title(f"NB samples: {n}")
        plt.show()

