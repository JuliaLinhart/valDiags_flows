# Implementation of the optimal Bayes classifier for
# - Gaussian L/QDA
# - Student t distributions with fixed df=2

import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import t

from valdiags.c2st_utils import compute_metric, t_stats_c2st
from valdiags.vanillaC2ST import eval_c2st

from tqdm import tqdm
from functools import partial


class OptimalBayesClassifier:
    """Base classs for an optimal (binary) Bayes classifier to discriminate 
    between data from two distributions associated to the classes c0 and c1:
        
        - X|c0 ~ P 
        - X|c1 ~ Q
    
    The optimal Bayes classifier is given by:
                    
        f(x) = argmax_{p(c0|x), p(c1|x)} \in {0,1} 

    with p(c0|x) = p(x|c0) / (p(x|c0) + p(x|c1)) and p(c1|x) = 1 - p(c0|x).

    Methods:
        fit: fit the classifier to data from P and Q.
            This method is empty as the optimal Bayes classifier is deterministic
            and does not need to be trained.
        predict: predict the class of a given sample.
            returns a numpy array of size (n_samples,).
        predict_proba: predict the probability of the sample to belong to class 0/1.
            returns a numpy array of size (n_samples, 2) with the probabilities.
        score: compute the accuracy of the classifier on a given dataset.
            returns a float.

    """

    def __init__(self) -> None:
        self.dist_c0 = None
        self.dist_c1 = None

    def fit(self, P, Q):
        pass

    def predict(self, x):
        return np.argmax([self.dist_c0.pdf(x), self.dist_c1.pdf(x)], axis=0)

    def predict_proba(self, x):
        d = (self.dist_c0.pdf(x) / (self.dist_c0.pdf(x) + self.dist_c1.pdf(x))).reshape(
            -1, 1
        )
        return np.concatenate([d, 1 - d], axis=1,)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


class AnalyticGaussianLQDA(OptimalBayesClassifier):
    """`OptimalBayesClassifier` for the Gaussian Linear Quadratic Discriminant Analysis (LQDA).
    The two classes are multivariate Gaussians of size `dim`:

        - c0: N(0, I)
        - c1: N(mu, sigma^2*I) with mu and sigma^2 to be specified.
    
    """

    def __init__(self, dim, mu=0, sigma=1) -> None:
        super().__init__()
        self.dist_c0 = mvn(mean=np.zeros(dim))
        self.dist_c1 = mvn(mean=np.array([mu] * dim), cov=np.eye(dim) * sigma)


class AnalyticStudentClassifier(OptimalBayesClassifier):
    """`OptimalBayesClassifier` for Student t distributions.
    The two classes are Student t distributions with fixed df=2:
    
        - c0: t(df=2, loc=0, scale=1)
        - c1: t(df=2, loc=mu, scale=sigma) with mu and sigma to be specified.
    """

    def __init__(self, mu=0, sigma=1) -> None:
        super().__init__()
        self.dist_c0 = t(df=2, loc=0, scale=1)
        self.dist_c1 = t(df=2, loc=mu, scale=sigma)


def opt_bayes_scores(
    P,
    Q,
    clf,
    metrics=["accuracy", "probas_mean", "div", "mse"],
    single_class_eval=True,
    cross_val=False,
):
    """Compute the scores of the optimal Bayes classifier on the data from P and Q.
    These scores can be used as test statistics for the C2ST test.

    Args:
        P (np.array): data drawn from P (c0)
            of size (n_samples, dim).
        Q (np.array): data drawn from Q (c1)
            of size (n_samples, dim).
        clf (OptimalBayesClassifier): the initialized classifier to use.
            needs to have a `score` and `predict_proba` method.
        metrics (list, optional): list of metric names (strings) to compute. 
            Defaults to ["accuracy", "div", "mse"].
        single_class_eval (bool, optional): if True, the classifier is evaluated on P only.
            Defaults to True.
        cross_val (bool, optional): never used. Defaults to False.

    Returns:
        dict: dictionary of scores for each metric.
    """
    # evaluate the classifier on the data
    accuracy, proba = eval_c2st(P=P, Q=Q, clf=clf, single_class_eval=single_class_eval)

    # compute the scores / metrics
    scores = dict(zip(metrics, [None] * len(metrics)))
    for m in metrics:
        if m == "accuracy":
            scores["accuracy"] = accuracy  # already computed
        else:
            scores[m] = compute_metric(
                proba, metrics=[m], single_class_eval=single_class_eval
            )[m]

    return scores


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
    mse = []
    single_class = []
    shift_list = []

    for s, s_samples in zip(shifts, shifted_samples):
        # # uncomment this to do the mean-shift experiment
        # clf = AnalyticGaussianLQDA(dim=DIM, mu=s)
        # # uncomment this to do the scale-shift experiment
        # clf = AnalyticGaussianLQDA(dim=DIM, sigma=s)

        # uncomment this to do the student mean-shift experiment
        clf = AnalyticStudentClassifier(mu=s)

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

    # df = pd.DataFrame(
    #     {
    #         "mean_shift": shift_list,
    #         "accuracy": accuracies,
    #         # "probas_mean": probas_mean,
    #         "div": div,
    #         "mse": mse,
    #         "single_class_eval": single_class,
    #     }
    # )

    # for metric in ["accuracy", "div", "mse"]:
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
    #         # "probas_mean": probas_mean,
    #         "div": div,
    #         "mse": mse,
    #         "single_class_eval": single_class,
    #     }
    # )

    # for metric in ["accuracy", "div", "mse"]:
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
        plt.savefig(f"student_mean_shift_n_{N_SAMPLES}_{metric}.pdf")
        plt.show()

