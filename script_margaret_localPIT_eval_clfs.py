import submitit
import time

import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import math
from functools import partial
from scipy.stats import norm
from diagnostics.localPIT_regression import (
    localPIT_regression_baseline,
    infer_r_alphas_baseline,
    localPIT_regression_sample,
    infer_r_alphas_amortized,
)
from diagnostics.pp_plots import PP_vals
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import KFold

from data.data_generators import ConditionalGaussian1d

# EXPERIMENT = 'Gaussian1d'
EXPERIMENT = "JR-NMM"
gauss1d_data = torch.load("saved_experiments/Gaussian1d/datasets.pkl")
jrnmm_data = torch.load("saved_experiments/JR-NMM/datasets_naive.pkl")

# data DIM
DIM = 4
N_LIST = [1000, 2000, 5000, 10000]  # calibration dataset size


# Simulated data for clf eval
# data_gen = ConditionalGaussian1d()
# x_samples = {}
# for n in N_LIST:
#     x_samples[n], _ = data_gen.get_joint_data(n=n)

x_samples = {}
x_samples[1000] = jrnmm_data["B_double_prime"]["x"][:, :, 0]
x_samples[2000] = jrnmm_data[2000]["x"][:, :, 0]
x_samples[5000] = jrnmm_data[5000]["x"][:, :, 0]
x_samples[10000] = jrnmm_data["B_prime"]["x"][:, :, 0]


# Reference samples (base distribution)
# P = norm()
P = D.MultivariateNormal(loc=torch.zeros(DIM), covariance_matrix=torch.eye(DIM))

# ALPHAS
alphas = np.linspace(0, 0.99, 100)

# clfs
# clf_hist = HistGradientBoostingClassifier(monotonic_cst=[0, 0, 1], max_iter=70)
clf_hist = HistGradientBoostingClassifier(monotonic_cst=[0 for _ in range(33)]+[1], max_iter=90)
clf_mlp = MLPClassifier(alpha=0, max_iter=25000)

def get_executor_marg(job_name, timeout_hour=60, n_cpus=40):

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f"{timeout_hour}:00:00",
        slurm_additional_parameters={
            "ntasks": 1,
            "cpus-per-task": n_cpus,
            "distribution": "block:block",
        },
    )
    return executor


def eval_classifier_for_localPIT(
    x_samples,
    shifts,
    shift_object="mean",
    reg_methods=["mlp_base", "mlp_sample50", "hist_sample50"],
    n_trials=10,
):
    nb_samples = len(x_samples)

    reg = partial(localPIT_regression_baseline, classifier=clf_mlp, alphas=alphas)
    infer = infer_r_alphas_baseline

    reg_method = []
    shift = []
    scores = []
    times = []
    if shift_object == "mean":
        for method in reg_methods:
            print(method)
            if 'sample' in method:
                infer = partial(infer_r_alphas_amortized, alphas=alphas)
                if 'mlp' in method:
                    reg = partial(localPIT_regression_sample, nb_samples=50, classifier=clf_mlp)
                else:
                    reg = partial(localPIT_regression_sample, nb_samples=50, classifier=clf_hist)

            for m in shifts:
                # for _ in range(n_trials):
                # Q_samples = norm().cdf(norm(loc=m).rvs(nb_samples))
                Q_samples = (
                    D.MultivariateNormal(
                        loc=torch.ones(DIM) * m, covariance_matrix=torch.eye(DIM)
                    )
                    .rsample((nb_samples,))
                    .numpy()
                )
                pp_vals_true = PP_vals(Q_samples, alphas)
                kf = KFold(n_splits=n_trials, shuffle=True, random_state=1)
                start = time.time()
                for i, (train_index, test_index) in enumerate(kf.split(x_samples)):
                    x_samples_train = x_samples[train_index]
                    Q_samples_train = Q_samples[train_index].ravel()
                    x_samples_test = x_samples[test_index]
                    Q_samples_test = Q_samples[test_index]
                    print(Q_samples_train.shape, x_samples_train.shape)
                    clfs = reg(
                        pit_values_train=Q_samples_train, x_train=x_samples_train
                    )
                    euc_dist_norm = 0
                    for x in x_samples_test:
                        x = x[None, :]
                        r_alpha_test = pd.Series(infer(x_eval=x, clfs=clfs))
                        euc_dist = math.dist(pp_vals_true, np.array(r_alpha_test))
                        # max = [0]*len(alphas)
                        # max_dist = math.dist(alphas, max)
                        euc_dist_norm += euc_dist / len(x_samples_test)
                    scores.append(euc_dist_norm)
                    reg_method.append(method)
                    shift.append(m)
                total_cv_time = start - time.time()
                for _ in range(n_trials):
                    times.append(total_cv_time)      
    else:
        for method in reg_methods:
            print(method)
            if 'sample' in method:
                infer = partial(infer_r_alphas_amortized, alphas=alphas)
                if 'mlp' in method:
                    reg = partial(localPIT_regression_sample, nb_samples=50, classifier=clf_mlp)
                else:
                    reg = partial(localPIT_regression_sample, nb_samples=50, classifier=clf_hist)

            for s in shifts:
                # for _ in range(n_trials):
                # Q_samples = norm().cdf(norm(loc=m).rvs(nb_samples))
                Q_samples = (
                    D.MultivariateNormal(
                        loc=torch.zeros(DIM), covariance_matrix=torch.eye(DIM)*s
                    )
                    .rsample((nb_samples,))
                    .numpy()
                )
                pp_vals_true = PP_vals(Q_samples, alphas)
                kf = KFold(n_splits=n_trials, shuffle=True, random_state=1)
                start = time.time()
                for i, (train_index, test_index) in enumerate(kf.split(x_samples)):
                    x_samples_train = x_samples[train_index]
                    Q_samples_train = Q_samples[train_index].ravel()
                    x_samples_test = x_samples[test_index]
                    Q_samples_test = Q_samples[test_index]
                    print(Q_samples_train.shape, x_samples_train.shape)
                    clfs = reg(
                        pit_values_train=Q_samples_train, x_train=x_samples_train
                    )
                    euc_dist_norm = 0
                    for x in x_samples_test:
                        x = x[None, :]
                        r_alpha_test = pd.Series(infer(x_eval=x, clfs=clfs))
                        euc_dist = math.dist(pp_vals_true, np.array(r_alpha_test))
                        # max = [0]*len(alphas)
                        # max_dist = math.dist(alphas, max)
                        euc_dist_norm += euc_dist / len(x_samples_test)
                    scores.append(euc_dist_norm)
                    reg_method.append(method)
                    shift.append(s)
                total_cv_time = start - time.time()
                for _ in range(n_trials):
                    times.append(total_cv_time)      


    df = pd.DataFrame(
        {f"{shift_object}_shift": shift, "eucledean_dist_to_gt": scores, "total_cv_time":times, "method": reg_method,}
    )
    filename = f"saved_experiments/{EXPERIMENT}/localPIT_eval_clfs/df_{shift_object}_{nb_samples}.pkl"
    torch.save(df, filename)


executor = get_executor_marg(f"work_eval_localPIT_clfs")
# launch batches
with executor.batch():
    print("Submitting jobs...", end="", flush=True)
    tasks = []
    for n in [1000]:
        for shifts, s_object in zip(
            [[0],[1]],
            # [[0, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 5, 10], np.linspace(1, 20, 10)],
            ["mean", "scale"],
        ):
            kwargs = {
                "shifts": shifts,
                "shift_object": s_object,
                "x_samples": x_samples[n],
                "n_trials": 2,
            }
            tasks.append(
                executor.submit(eval_classifier_for_localPIT, **kwargs)
            )

