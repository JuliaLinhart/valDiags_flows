import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import t


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


if __name__ == "__main__":
    from valdiags.localC2ST import compute_metric

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    N_SAMPLES = 10_000
    DIM = 2

    shifts = np.array([0, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 5, 10])
    shifts = np.concatenate([-1 * shifts, shifts[1:]])
    # # uncomment this to do the scale-shift experiment
    # shifts = np.linspace(0.01, 10, 20)

    # # ref norm samples
    # ref_samples = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)

    # shifted_samples = [
    #     mvn(mean=np.array([s] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES) for s in shifts
    # ]

    # # uncomment this to do the scale-shift experiment
    # shifted_samples = [
    #     mvn(mean=np.zeros(DIM), cov=np.eye(DIM) * s).rvs(N_SAMPLES) for s in shifts
    # ]

    # uncomment this to do the student mean-shift experiment
    # ref student samples
    ref_samples = t(df=2, loc=0, scale=1).rvs(N_SAMPLES)
    shifted_samples = [t(df=2, loc=s, scale=1).rvs(N_SAMPLES) for s in shifts]

    accuracies = []
    probas_mean = []
    div = []
    single_class_eval = []
    shift_list = []

    for s, s_samples in zip(shifts, shifted_samples):
        # # uncomment this to do the mean-shift experiment
        # clf = AnalyticGaussianLQDA(dim=DIM, mu=s)
        # # uncomment this to do the scale-shift experiment
        # clf = AnalyticGaussianLQDA(dim=DIM, sigma=s)

        # uncomment this to do the student mean-shift experiment
        clf = AnalyticStudentClassifier(mu=s)

        for b in [True, False]:
            single_class_eval.append(b)
            shift_list.append(s)

            if b:
                X_val = ref_samples
                y_val = np.array([0] * (N_SAMPLES))
            else:
                X_val = np.concatenate([ref_samples, s_samples], axis=0)
                y_val = np.array([0] * N_SAMPLES + [1] * N_SAMPLES)

            accuracy = clf.score(X_val, y_val)

            proba = clf.predict_proba(ref_samples)[:, 0]
            if not b:
                proba_1 = clf.predict_proba(s_samples)[:, 1]
                proba = np.concatenate([proba, proba_1], axis=0)

            scores = compute_metric(proba, metrics=["probas_mean", "div"])

            accuracies.append(accuracy)
            probas_mean.append(scores["probas_mean"])
            div.append(scores["div"])

    # df = pd.DataFrame(
    #     {
    #         "mean_shift": shift_list,
    #         "accuracy": accuracies,
    #         "probas_mean": probas_mean,
    #         "div": div,
    #         "single_class_eval": single_class_eval,
    #     }
    # )

    # for metric in ["accuracy", "probas_mean", "div"]:
    #     sns.relplot(
    #         data=df,
    #         x="mean_shift",
    #         y=metric,
    #         hue="single_class_eval",
    #         style="single_class_eval",
    #         kind="line",
    #     )
    #     plt.savefig(f"lqda_mean_shift_n_{N_SAMPLES}_{metric}.pdf")
    #     plt.show()

    # # uncomment this to do the scale-shift experiment
    # df = pd.DataFrame(
    #     {
    #         "scale_shift": shift_list,
    #         "accuracy": accuracies,
    #         "probas_mean": probas_mean,
    #         "div": div,
    #         "single_class_eval": single_class_eval,
    #     }
    # )

    # for metric in ["accuracy", "probas_mean", "div"]:
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

    # uncomment this to do the student mean-shift experiment
    df = pd.DataFrame(
        {
            "mean_shift": shift_list,
            "accuracy": accuracies,
            "probas_mean": probas_mean,
            "div": div,
            "single_class_eval": single_class_eval,
        }
    )

    for metric in ["accuracy", "probas_mean", "div"]:
        sns.relplot(
            data=df,
            x="mean_shift",
            y=metric,
            hue="single_class_eval",
            style="single_class_eval",
            kind="line",
        )
        plt.savefig(f"student_mean_shift_n_{N_SAMPLES}_{metric}.pdf")
        plt.show()

