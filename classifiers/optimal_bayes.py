import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import t

from valdiags.localC2ST import compute_metric
from tqdm import tqdm


class AnalyticGaussianLQDA:
    def __init__(self, dim, mu=0, sigma=1) -> None:
        self.dist_c1 = mvn(mean=np.array([mu] * dim), cov=np.eye(dim) * sigma)
        self.dist_c0 = mvn(mean=np.zeros(dim))

    def predict(self, x):
        return np.argmax([self.dist_c0.pdf(x), self.dist_c1.pdf(x)], axis=0)

    def predict_proba(self, x):
        d = (self.dist_c0.pdf(x) / (self.dist_c0.pdf(x) + self.dist_c1.pdf(x))).reshape(
            -1, 1
        )
        return np.concatenate([d, 1 - d], axis=1,)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


class AnalyticStudentClassifier:
    def __init__(self, df=2, mu=0, sigma=1) -> None:
        self.dist_c1 = t(df, loc=mu, scale=sigma)
        self.dist_c0 = t(df, loc=0, scale=1)

    def predict(self, x):
        return np.argmax([self.dist_c0.pdf(x), self.dist_c1.pdf(x)], axis=0)

    def predict_proba(self, x):
        d = (self.dist_c0.pdf(x) / (self.dist_c0.pdf(x) + self.dist_c1.pdf(x))).reshape(
            -1, 1
        )
        return np.concatenate([d, 1 - d], axis=1,)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


def opt_bayes_scores(
    P, Q, clf, metrics=["accuracy", "probas_mean", "div", "mse"], single_class_eval=True
):
    N_SAMPLES = len(P)
    if single_class_eval:
        X_val = P
        y_val = np.array([0] * (N_SAMPLES))
    else:
        X_val = np.concatenate([P, Q], axis=0)
        y_val = np.array([0] * N_SAMPLES + [1] * N_SAMPLES)

    accuracy = clf.score(X_val, y_val)

    proba = clf.predict_proba(P)[:, 0]
    if not single_class_eval:
        proba_1 = clf.predict_proba(Q)[:, 1]
        proba = np.concatenate([proba, proba_1], axis=0)

    scores = dict(zip(metrics, [None] * len(metrics)))
    for m in metrics:
        if m == "accuracy":
            scores["accuracy"] = accuracy
        else:
            scores[m] = compute_metric(proba, metrics=[m])[m]

    return scores


def t_stats_opt_bayes(
    P, Q, null_samples_list, metrics, clf_data, clf_null, verbose=True, **kwargs
):
    t_stat_data = {}
    t_stats_null = dict(zip(metrics, [[] for _ in range(len(metrics))]))

    scores_data = opt_bayes_scores(P=P, Q=Q, metrics=metrics, clf=clf_data, **kwargs)
    for m in metrics:
        t_stat_data[m] = np.mean(scores_data[m])
    for i in tqdm(
        range(len(null_samples_list)),
        desc="Testing under the null",
        disable=(not verbose),
    ):
        scores_null = opt_bayes_scores(
            P=P, Q=null_samples_list[i], metrics=metrics, clf=clf_null, **kwargs,
        )
        for m in metrics:
            t_stats_null[m].append(np.mean(scores_null[m]))

    return t_stat_data, t_stats_null


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    N_SAMPLES = 10_000
    DIM = 2

    shifts = np.array([0, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 5, 10])
    shifts = np.concatenate([-1 * shifts, shifts[1:]])
    # # uncomment this to do the scale-shift experiment
    # shifts = np.linspace(0.01, 10, 20)

    # ref norm samples
    ref_samples = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)

    shifted_samples = [
        mvn(mean=np.array([s] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES) for s in shifts
    ]

    # # uncomment this to do the scale-shift experiment
    # shifted_samples = [
    #     mvn(mean=np.zeros(DIM), cov=np.eye(DIM) * s).rvs(N_SAMPLES) for s in shifts
    # ]

    # # uncomment this to do the student mean-shift experiment
    # # ref student samples
    # ref_samples = t(df=2, loc=0, scale=1).rvs(N_SAMPLES)
    # shifted_samples = [t(df=2, loc=s, scale=1).rvs(N_SAMPLES) for s in shifts]

    accuracies = []
    probas_mean = []
    div = []
    mse = []
    single_class = []
    shift_list = []

    for s, s_samples in zip(shifts, shifted_samples):
        # # uncomment this to do the mean-shift experiment
        clf = AnalyticGaussianLQDA(dim=DIM, mu=s)
        # # uncomment this to do the scale-shift experiment
        # clf = AnalyticGaussianLQDA(dim=DIM, sigma=s)

        # uncomment this to do the student mean-shift experiment
        # clf = AnalyticStudentClassifier(mu=s)

        for b in [True, False]:
            single_class.append(b)
            shift_list.append(s)

            scores = opt_bayes_scores(
                P=ref_samples, Q=s_samples, clf=clf, single_class_eval=b
            )

            accuracies.append(scores["accuracy"])
            probas_mean.append(scores["probas_mean"])
            div.append(scores["div"])
            mse.append(scores["mse"])

    df = pd.DataFrame(
        {
            "mean_shift": shift_list,
            "accuracy": accuracies,
            "probas_mean": probas_mean,
            "div": div,
            "mse": mse,
            "single_class_eval": single_class,
        }
    )

    for metric in ["accuracy", "probas_mean", "div", "mse"]:
        sns.relplot(
            data=df,
            x="mean_shift",
            y=metric,
            hue="single_class_eval",
            style="single_class_eval",
            kind="line",
        )
        plt.savefig(f"lqda_mean_shift_n_{N_SAMPLES}_{metric}.pdf")
        plt.show()

    # # uncomment this to do the scale-shift experiment
    # df = pd.DataFrame(
    #     {
    #         "scale_shift": shift_list,
    #         "accuracy": accuracies,
    #         "probas_mean": probas_mean,
    #         "div": div,
    #         "mse": mse,
    #         "single_class_eval": single_class_eval,
    #     }
    # )

    # for metric in ["accuracy", "probas_mean", "div", "mse"]:
    #     sns.relplot(
    #         data=df,
    #         x="scale_shift",
    #         y=metric,
    #         hue="single_class_eval",
    #         style="single_class_eval",
    #         kind="line",
    #     )
    #     plt.savefig(f"lqda_scale_shift_n_{N_SAMPLES}_{metric}.pdf")
    #     plt.show()

    # # uncomment this to do the student mean-shift experiment
    # df = pd.DataFrame(
    #     {
    #         "mean_shift": shift_list,
    #         "accuracy": accuracies,
    #         "probas_mean": probas_mean,
    #         "div": div,
    #         "mse": mse,
    #         "single_class_eval": single_class_eval,
    #     }
    # )

    # for metric in ["accuracy", "probas_mean", "div", "mse"]:
    #     sns.relplot(
    #         data=df,
    #         x="mean_shift",
    #         y=metric,
    #         hue="single_class_eval",
    #         style="single_class_eval",
    #         kind="line",
    #     )
    #     plt.savefig(f"student_mean_shift_n_{N_SAMPLES}_{metric}.pdf")
    #     plt.show()

