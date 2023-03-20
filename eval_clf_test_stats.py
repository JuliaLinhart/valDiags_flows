import numpy as np
from scipy.stats import multivariate_normal as mvn

from valdiags.test_utils import empirical_error_htest

from classifiers.optimal_bayes import AnalyticGaussianLQDA, t_stats_opt_bayes

from valdiags.vanillaC2ST import t_stats_c2st, c2st_scores
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
from tqdm import tqdm

### =========== empirical power as in [Lee et al. (2018)] ============

N_SAMPLES = 100

DIM = 5
mu = np.sqrt(0.05)
# DIM = 20
# mu = np.sqrt(0.01)

metrics = ["accuracy", "div", "mse"]
n_alpha = 20
n_runs = 300

# ESTIMATED LDA
power = dict(zip(metrics, [[] for _ in range(len(metrics))]))
for i, alpha in enumerate(np.linspace(0, 1, n_alpha)):
    print(f"Computing empirical error as the success rate over {n_runs} runs:")
    print(f"Significance level alpha {i+1}/{n_alpha} = {alpha}")
    power_a = dict(zip(metrics, [0] * len(metrics)))

    for _ in tqdm(range(n_runs)):
        ref_samples = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
        shift_samples = mvn(mean=np.array([mu] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
        null_samples_list = [
            mvn(mean=np.array([0] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
            for _ in range(100)
        ]

        success_rate = empirical_error_htest(
            t_stats_estimator=t_stats_c2st,
            metrics=metrics,
            conf_alpha=alpha,
            P=ref_samples,
            Q=shift_samples,
            null_samples_list=null_samples_list,
            clf_class=LinearDiscriminantAnalysis,
            clf_kwargs={},
            single_class_eval=True,
            # n_runs=300,
            verbose=False,
            n_folds=2,
        )

        for m in metrics:
            power_a[m] += success_rate[m] / n_runs

    for m in metrics:
        power[m].append(power_a[m])

for m in metrics:
    plt.plot(np.linspace(0, 1, n_alpha), power[m], label=str(m), marker="o")
plt.legend()
plt.savefig(f"emp_power_lqda_single_class_dim_{DIM}.pdf")
plt.show()

# # OPIMAL BAYES LDA
# clf_mu = AnalyticGaussianLQDA(dim=DIM, mu=mu)
# power = dict(zip(metrics, [[] for _ in range(len(metrics))]))
# for i, alpha in enumerate(np.linspace(0, 1, n_alpha)):
#     print(f"alpha {i+1}/{n_alpha} = {alpha}")
#     print(f"Computing empirical error as the success rate over {n_runs} runs:")
#     power_a = dict(zip(metrics, [0] * len(metrics)))

#     for _ in tqdm(range(n_runs)):
#         ref_samples = mvn(mean=np.zeros(DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
#         shift_samples = mvn(mean=np.array([mu] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
#         null_samples_list = [
#             mvn(mean=np.array([0] * DIM), cov=np.eye(DIM)).rvs(N_SAMPLES)
#             for _ in range(100)
#         ]

#         success_rate = empirical_error_htest(
#         t_stats_estimator=t_stats_opt_bayes,
#         metrics=metrics,
#         conf_alpha=alpha,
#         P=ref_samples,
#         Q=shift_samples,
#         null_samples_list=null_samples_list,
#         clf_data=clf_mu,
#         clf_null=AnalyticGaussianLQDA(dim=DIM),
#         single_class_eval=True,
#         # n_runs=300,
#         verbose=False,
#         )

#         for m in metrics:
#             power_a[m] += success_rate[m] / n_runs

#     for m in metrics:
#         power[m].append(power_a[m])

# for m in metrics:
#     plt.plot(
#         np.linspace(0, 1, n_alpha), power[m], label=str(m),
#     )
# plt.legend()
# plt.show()

