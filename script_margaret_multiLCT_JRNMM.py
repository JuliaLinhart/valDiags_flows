import submitit
import torch
import numpy as np


from data.feature_transforms import identity
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from diagnostics.pp_plots import multi_cde_pit_values
from diagnostics.localPIT_regression import (
    localPIT_regression_sample,
    localPIT_regression_baseline,
)
from diagnostics.multi_local_test import multi_local_pit_regression, multivariate_lct

from functools import partial

PATH_EXPERIMENT = "saved_experiments/JR-NMM/"
METHOD = "naive"
N_EXTRA = 0
SINGLE_REC = False
N_SIM = 50_000

DATASETS = torch.load(PATH_EXPERIMENT + f"datasets_{METHOD}.pkl")

POSTERIOR = torch.load(
    PATH_EXPERIMENT
    + f"posteriors_amortized/{METHOD}_posterior_nextra_{N_EXTRA}_single_rec_{SINGLE_REC}_nsim_{N_SIM}.pkl"
)

# EXP_NAME = 'max_iter_exp'
EXP_NAME = 'baseline'
# EXP_NAME = 't_stat_variance_exp'
# EXP_NAME = "reg_eval"
# EXP_NAME = 'histgrad30'
# EXP_NAME = 'histgrad70'

NB_CLASSIFIERS = 1

GAIN_LIST = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

X_OBS_PCA = torch.load(
    PATH_EXPERIMENT + "gt_observations/nextra_0/pca_experiment_new.pkl"
)[1]
X_OBS_GAIN = torch.load(
    PATH_EXPERIMENT + "gt_observations/nextra_0/gain_experiment_new.pkl"
)[1][1:10,:,:]

X_OBS_GAIN_NO_STOCH = torch.load(
    PATH_EXPERIMENT + "gt_observations/nextra_0/gain_experiment_no_stoch_new.pkl"
)

N_TRIALS = 100
N_ALPHAS = 100
ALPHA_MAX = 0.99

# MAX_ITER = 40
# MAX_ITER_LIST = np.linspace(10, 200, 20, dtype=int)
# MAX_ITER_LIST = [100]

# CLF = HistGradientBoostingClassifier(
#     monotonic_cst=[0 for i in range(33)] + [1], max_iter=MAX_ITER
# )
# CLF_LIST = [
#     HistGradientBoostingClassifier(
#         monotonic_cst=[0 for i in range(33)] + [1], max_iter=it
#     )
#     for it in MAX_ITER_LIST
# ]
# CLF = HistGradientBoostingClassifier(
#     monotonic_cst=[0 for i in range(33)] + [0], max_iter=MAX_ITER
# )
CLF = MLPClassifier(alpha=0, max_iter=25000)
CLF_LIST = [CLF for i in range(NB_CLASSIFIERS)]

# REG_METHOD = partial(localPIT_regression_sample, nb_samples=50)
REG_METHOD = partial(
    localPIT_regression_baseline, alphas=np.linspace(0, ALPHA_MAX, N_ALPHAS)
)

METHOD_NAME_LIST = [f"baseline_mlp_nalpha_{N_ALPHAS}"]
# METHOD_NAME = "sample50_histgrad90_no_monoton"
# METHOD_NAME = 'sample50_histgrad90'
# METHOD_NAME_LIST = [f"sample50_histgrad90_maxiter_{it}" for it in MAX_ITER_LIST]
# METHOD_NAME_LIST = [f"baseline_mlp_nalpha_{N_ALPHAS}_n_clf_{i}" for i in range(NB_CLASSIFIERS)]
# METHOD_NAME_LIST = [
#     f"sample50_histgrad130_{N_ALPHAS}_n_clf_{i}" for i in range(NB_CLASSIFIERS)
# ]


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


# train classifiers
def train_classifiers(
    theta_train,
    x_train,
    method_name_list,
    clf_list,
    flow=POSTERIOR,
    feature_transform=identity,
    n_trials=N_TRIALS,
    null=False,
):
    pit_values_train_flow = multi_cde_pit_values(
        theta_train, x_train, flow, feature_transform=feature_transform
    )
    for i in range(NB_CLASSIFIERS):
        if not null:

            _, trained_clfs = multi_local_pit_regression(
                dim=theta_train.shape[-1],
                pit_values_train=pit_values_train_flow,
                x_train=x_train,
                reg_method=REG_METHOD,
                classifier=clf_list[i],
            )
            torch.save(
                trained_clfs,
                PATH_EXPERIMENT
                + f"trained_classifiers/5layers/{EXP_NAME}/classifiers_{method_name_list[i]}_{METHOD}_nextra_{N_EXTRA}_nsim_{N_SIM}.pkl",
            )
        else:
            trained_clfs_null = []
            for _ in range(n_trials):
                # pit_values_train_null = [
                #     np.random.uniform(size=pit_values_train_flow[0].shape)
                # ] * theta_train.shape[-1]
                pit_values_train_null = [np.random.uniform(size=pit_values_train_flow[0].shape)]
                _, trained_clfs_null_k = multi_local_pit_regression(
                    dim=len(pit_values_train_null),
                    pit_values_train=pit_values_train_null,
                    x_train=x_train,
                    reg_method=REG_METHOD,
                    classifier=clf_list[i],
                )
                trained_clfs_null.append(trained_clfs_null_k)

            torch.save(
                trained_clfs_null,
                PATH_EXPERIMENT
                + f"trained_classifiers/{EXP_NAME}/classifiers_{method_name_list[i]}_null_ntrials_{n_trials}.pkl",
            )


# Multivariate LCT quantities
def compute_multi_lct_values(
    theta_train,
    x_train,
    x_obs,
    n_trials=N_TRIALS,
    return_pvalues=True,
    k=None,
    gain=None,
):
    for i in range(NB_CLASSIFIERS):
        trained_clfs = torch.load(
            PATH_EXPERIMENT
            + f"trained_classifiers/{EXP_NAME}/classifiers_{METHOD_NAME_LIST[i]}_{METHOD}_nextra_{N_EXTRA}_nsim_{N_SIM}.pkl"
        )
        if return_pvalues:
            trained_clfs_null = torch.load(
                PATH_EXPERIMENT
                + f"trained_classifiers/{EXP_NAME}/classifiers_{METHOD_NAME_LIST[i]}_null_ntrials_{n_trials}.pkl"
            )
        else:
            trained_clfs_null = None

        lct_dict = multivariate_lct(
            theta_train,
            x_train,
            x_obs,
            flow=POSTERIOR,
            n_trials=n_trials,
            n_alphas=N_ALPHAS,
            alpha_max=ALPHA_MAX,
            reg_method=REG_METHOD,
            classifier=CLF_LIST[i],
            trained_clfs=trained_clfs,
            trained_clfs_null=trained_clfs_null,
            return_pvalues=return_pvalues,
        )

        if gain is not None:
            filename = (
                PATH_EXPERIMENT
                + f"lct_results/naive_nextra_{N_EXTRA}_nsim_{N_SIM}/gain/baseline/lct_results_{METHOD_NAME_LIST[i]}_ntrials_{n_trials}_n_alphas_{N_ALPHAS}_gain_{gain}.pkl"
            )
        else:
            filename = (
                PATH_EXPERIMENT
                + f"lct_results/naive_nextra_{N_EXTRA}_nsim_{N_SIM}/{EXP_NAME}/lct_results_{METHOD_NAME_LIST[i]}_ntrials_{n_trials}_n_alphas_{N_ALPHAS}_pca_x{k}.pkl"
            )
        torch.save(lct_dict, filename)


def compute_expected_pit(theta_train, x_train, x_evals, clf_list, method_name_list, method_name):
    n_eval = len(x_evals)
    E_hats = {}
    for i in range(NB_CLASSIFIERS):
        E_hat_c = {}
        for n, x in enumerate(x_evals):
            trained_clfs = torch.load(
                PATH_EXPERIMENT
                + f"trained_classifiers/reg_eval/classifiers_{method_name_list[i]}_{METHOD}_nextra_{N_EXTRA}_nsim_{N_SIM}.pkl"
            )

            lct_dict = multivariate_lct(
                theta_train,
                x_train,
                x_eval=x[None, :, :],
                flow=POSTERIOR,
                n_alphas=N_ALPHAS,
                alpha_max=ALPHA_MAX,
                reg_method=REG_METHOD,
                classifier=clf_list[i],
                trained_clfs=trained_clfs,
                trained_clfs_null=None,
                return_pvalues=False,
            )
            r_alpha_n_i = lct_dict["r_alpha_learned"]
            for dim in range(1, 5):
                if n == 0:
                    e = np.array(list(r_alpha_n_i[f"dim_{dim}"].values())) / n_eval
                else:
                    e = e + np.array(list(r_alpha_n_i[f"dim_{dim}"].values())) / n_eval
                E_hat_c[f"dim_{dim}"] = e
        E_hats[i] = E_hat_c
    filename = PATH_EXPERIMENT + f"reg_eval/expected_pit_list_{method_name}.pkl"
    torch.save(E_hats, filename)


executor = get_executor_marg(f"work_localPIT")
# launch batches
with executor.batch():
    print("Submitting jobs...", end="", flush=True)
    tasks = []
    # for max_iter in MAX_ITER_LIST:
    #     clf = HistGradientBoostingClassifier(
    #         monotonic_cst=[0 for i in range(33)] + [1], max_iter=max_iter
    #     )
    #     clf_list = [clf for _ in range(NB_CLASSIFIERS)]
    #     method_name_list = [
    #         f"sample50_histgrad{max_iter}_{N_ALPHAS}_n_clf_{i}"
    #         for i in range(NB_CLASSIFIERS)
    #     ]
        # kwargs = {
        #     "theta_train": DATASETS["B_prime"]["theta"],
        #     "x_train": DATASETS["B_prime"]["x"],
        #     "null": False,
        #     "method_name_list": method_name_list,
        #     "clf_list": clf_list,
        # }
        # tasks.append(executor.submit(train_classifiers, **kwargs))

    # kwargs = {
    #         "theta_train": DATASETS["B_prime"]["theta"],
    #         "x_train": DATASETS["B_prime"]["x"],
    #         "null": True,
    #         "method_name_list": METHOD_NAME_LIST,
    #         "clf_list": CLF_LIST,
    #     }
    # tasks.append(executor.submit(train_classifiers, **kwargs))


    for g, x in zip(GAIN_LIST, X_OBS_GAIN):
        kwargs = {
            "theta_train": DATASETS["B_prime"]["theta"],
            "x_train": DATASETS["B_prime"]["x"],
            "x_obs": x[None,:,:],
            "gain":g,
            "return_pvalues":True,
        }
        tasks.append(executor.submit(compute_multi_lct_values, **kwargs))

    # k_list = [19, 18, 15, 14, 13, 10, 9, 17]
    # for k in range(20):
    #     kwargs = {
    #         "theta_train": DATASETS["B_prime"]["theta"],
    #         "x_train": DATASETS["B_prime"]["x"],
    #         "x_obs": X_OBS_PCA[k][None,:,:],
    #         "gain": None,
    #         "k":k,
    #         "return_pvalues":True
    #     }
    #     tasks.append(executor.submit(compute_multi_lct_values, **kwargs))

    # for k, x in enumerate(DATASETS['B_double_prime']['x'][:1000]):
    #     kwargs = {
    #         "theta_train": DATASETS["B_prime"]["theta"],
    #         "x_train": DATASETS["B_prime"]["x"],
    #         "x_obs": x[None,:,:],
    #         "gain": None,
    #         "k":k,
    #         "return_pvalues":False,
    #     }
    #     tasks.append(executor.submit(compute_multi_lct_values, **kwargs))

    # for max_iter in MAX_ITER_LIST:
    #     clf = HistGradientBoostingClassifier(
    #         monotonic_cst=[0 for i in range(33)] + [1], max_iter=max_iter
    #     )
    #     clf_list = [clf for _ in range(NB_CLASSIFIERS)]
    #     method_name_list = [
    #         f"sample50_histgrad{max_iter}_{N_ALPHAS}_n_clf_{i}"
    #         for i in range(NB_CLASSIFIERS)
    #     ]
    #     kwargs = {
    #         "theta_train": DATASETS["B_prime"]["theta"],
    #         "x_train": DATASETS["B_prime"]["x"],
    #         "x_evals": DATASETS["B_double_prime"]["x"][:1000],
    #         "clf_list": clf_list,
    #         "method_name_list": method_name_list,
    #         "method_name": f'hg{max_iter}',
    #     }
    #     tasks.append(executor.submit(compute_expected_pit, **kwargs))
