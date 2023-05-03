import os

import torch
import numpy as np

from tqdm import tqdm
import time

from valdiags.test_utils import eval_htest, permute_data
from valdiags.vanillaC2ST import t_stats_c2st
from valdiags.localC2ST import t_stats_lc2st, lc2st_scores
from valdiags.localHPD import t_stats_lhpd, lhpd_scores

from tasks.sbibm.data_generators import (
    generate_task_data,
    generate_npe_data_for_c2st,
    generate_npe_data_for_lc2st,
)
from tasks.sbibm.npe_utils import sample_from_npe_obs


def l_c2st_results_n_train(
    task,
    n_cal,
    n_eval,
    observation_dict,
    n_train_list,
    alpha,
    n_trials_null,
    t_stats_null_c2st_nf,
    n_trials_null_precompute,
    kwargs_c2st,
    kwargs_lc2st,
    kwargs_lhpd,
    task_path,
    t_stats_null_path,
    results_n_train_path="",
    methods=["c2st", "lc2st", "lc2st_nf"],
    test_stat_names=["accuracy", "mse", "div"],
    seed=42,
):
    # GENERATE DATA
    data_samples = generate_data_one_run(
        n_cal=n_cal,
        n_eval=n_eval,
        task=task,
        observation_dict=observation_dict,
        n_train_list=n_train_list,
        task_path=task_path,
        save_data=True,
        seed=seed,  # fixed seed for reproducibility
    )

    # precompute test statistics under null hypothesis for lc2st_nf
    # same for every estimator (no need to recompute for every n_train)
    if "lc2st_nf" in methods:
        x_cal = data_samples["joint_cal"]["x"]
        dim_theta = data_samples["joint_cal"]["theta"].shape[-1]
        t_stats_null_lc2st_nf = precompute_t_stats_null(
            n_cal=n_cal,
            n_eval=n_eval,
            dim_theta=dim_theta,
            n_trials_null=n_trials_null_precompute,
            kwargs_lc2st=kwargs_lc2st,
            x_cal=x_cal,
            observation_dict=observation_dict,
            methods=["lc2st_nf"],
            metrics=test_stat_names,
            t_stats_null_path=t_stats_null_path,
            save_results=True,
            load_results=True,
            # args only for c2st
            kwargs_c2st=None,
        )["lc2st_nf"]
    else:
        t_stats_null_lc2st_nf = None

    if "lhpd" in methods:
        x_cal = data_samples["joint_cal"]["x"]
        dim_theta = data_samples["joint_cal"]["theta"].shape[-1]
        t_stats_null_lhpd = precompute_t_stats_null(
            n_cal=n_cal,
            n_eval=n_eval,
            dim_theta=dim_theta,
            n_trials_null=n_trials_null_precompute,
            kwargs_lhpd=kwargs_lhpd,
            x_cal=x_cal,
            observation_dict=observation_dict,
            methods=["lhpd"],
            metrics=["mse"],
            t_stats_null_path=t_stats_null_path,
            save_results=True,
            load_results=True,
            # args only for c2st and lc2st
            kwargs_c2st=None,
            kwargs_lc2st=None,
        )["lhpd"]
    else:
        t_stats_null_lhpd = None

    avg_result_keys = {
        "TPR": "reject",
        "p_value_mean": "p_value",
        "p_value_std": "p_value",
        "t_stat_mean": "t_stat",
        "t_stat_std": "t_stat",
        "run_time_mean": "run_time",
        "run_time_std": "run_time",
    }
    avg_results = dict(zip(methods, [dict() for _ in methods]))
    for m in methods:
        avg_results[m] = dict(
            zip(
                avg_result_keys.keys(),
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in avg_result_keys
                ],
            )
        )
    train_runtime = dict(zip(methods, [[] for _ in methods]))

    for n_train in n_train_list:
        results_dict, train_runtime_n = compute_test_results_npe_one_run(
            alpha=alpha,
            data_samples=data_samples,
            n_train=n_train,
            observation_dict=observation_dict,
            kwargs_c2st=kwargs_c2st,
            kwargs_lc2st=kwargs_lc2st,
            kwargs_lhpd=kwargs_lhpd,
            n_trials_null=n_trials_null,
            t_stats_null_c2st_nf=t_stats_null_c2st_nf,
            t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,
            t_stats_null_lhpd=t_stats_null_lhpd,
            task_path=task_path,
            results_n_train_path=results_n_train_path,
            methods=methods,
            test_stat_names=test_stat_names,
            compute_under_null=False,
            save_results=True,
            seed=seed,
        )

        for method in methods:
            train_runtime[method].append(train_runtime_n[method])

        for method, results in results_dict.items():
            if method in methods:
                for k, v in avg_result_keys.items():
                    for t_stat_name in test_stat_names:
                        if method == "lhpd" and t_stat_name != "mse":
                            continue

                        if "std" in k:
                            if "run_time" in k:
                                avg_results[method][k][t_stat_name].append(
                                    np.std(
                                        np.array(results[v][t_stat_name])
                                        / n_trials_null
                                    )
                                )
                            else:
                                avg_results[method][k][t_stat_name].append(
                                    np.std(results[v][t_stat_name])
                                )
                        else:
                            if "t_stat" in k and t_stat_name == "mse":
                                avg_results[method][k][t_stat_name].append(
                                    np.mean(results[v][t_stat_name])
                                    + 0.5  # for comparison with other t_stats
                                )
                            elif "run_time" in k:
                                avg_results[method][k][t_stat_name].append(
                                    np.mean(
                                        np.array(results[v][t_stat_name])
                                        / n_trials_null
                                    )
                                )
                            else:
                                avg_results[method][k][t_stat_name].append(
                                    np.mean(results[v][t_stat_name])
                                )
    return avg_results, train_runtime


def compute_emp_power_l_c2st(
    n_runs,
    alpha,
    task,
    n_train,
    observation_dict,
    n_cal,
    n_eval,
    kwargs_c2st,
    kwargs_lc2st,
    kwargs_lhpd,
    n_trials_null,
    n_trials_null_precompute,
    t_stats_null_c2st_nf,
    task_path,
    methods=["c2st", "lc2st", "lc2st_nf", "lc2st_nf_perm"],
    test_stat_names=["accuracy", "mse", "div"],
    compute_emp_power=True,
    compute_type_I_error=False,
    result_path="",
):
    """Compute the empirical power of the (L)C2ST methods for a given task and npe-flow
    (corresponding to n_train). We also compute the type I error if specified.
    All methods will use the permutation method, except for (l)-c2st_nf that can use the same
    precomputed test statistics (t_stats_null_lc2st_nf, t_stats_null_c2st_nf) for every run.

    This function enables doing experiments where the results are obtained for all methods
    on the same data, by computing them all in the same run.

    Args:
        test_stat_names (List[str]): list of test statistic names to compute empirical power for.
            Must be compatible with the test_stat_estimator from `valdiags.test_utils.eval_htest`.
        n_runs (int): number of test runs to compute empirical power.
        alpha (float): significance level of the test.
        task (str): sbibm task name
        npe (sbi.DirectPosterior): neural posterior estimator (normalizing flow).
        observation_dict (dict): dict of observations over which we average the results.
            keys are observation numbers
            values are torch tensors of shape (1, dim_x)
        n_cal (int): Number of samples to use for calibration (train classifier).
        n_eval (int): Number of samples to use for evaluation (evaluate classifier).
        method (str): method to use for the test. One of ['c2st', 'lc2st', 'lc2st-nf'].
        compute_type_I_error (bool): whether to compute the type I error of the test.
        t_stats_null_lc2st_nf (dict): dict of precomputed test statistics for lc2st_nf.
            Needs to be provided for lc2st_nf (not _perm).
            Default: None.
    Returns:
        dict: dict of empirical power values for every test statistic computed in eval_htest_fn.
            The dict keys are the test statistic names.
            The dict values are lists of length len(num_observation_list) of empirical power values.
    """
    # initialize dict of p-values
    emp_power = {}
    type_I_error = {}
    p_values = {}
    p_values_h0 = {}
    for method in methods:
        print(method)
        emp_power[method] = dict(
            zip(
                test_stat_names,
                [np.zeros(len(observation_dict)) for _ in test_stat_names],
            )
        )
        type_I_error[method] = dict(
            zip(
                test_stat_names,
                [np.zeros(len(observation_dict)) for _ in test_stat_names],
            )
        )
        p_values[method] = dict(
            zip(
                test_stat_names,
                [
                    dict(
                        zip(
                            observation_dict.keys(),
                            [[] for _ in observation_dict.keys()],
                        )
                    )
                    for _ in test_stat_names
                ],
            )
        )
        p_values_h0[method] = dict(
            zip(
                test_stat_names,
                [
                    dict(
                        zip(
                            observation_dict.keys(),
                            [[] for _ in observation_dict.keys()],
                        )
                    )
                    for _ in test_stat_names
                ],
            )
        )

    for n in range(n_runs):
        print()
        print("====> RUN: ", n + 1, "/", n_runs, f", N_cal = {n_cal} <====")
        # GENERATE DATA
        data_samples = generate_data_one_run(
            n_cal=n_cal,
            n_eval=n_eval,
            task=task,
            observation_dict=observation_dict,
            n_train_list=[n_train],
            task_path=task_path,
            save_data=False,
            load_data=False,
            seed=n,  # different seed for every run (fixed for reproducibility)
        )

        # precompute test statistics under null hypothesis for lc2st_nf
        # we need to do this for every run because we use different data
        if "lc2st_nf" in methods and (compute_emp_power + compute_type_I_error != 0):
            x_cal = data_samples["joint_cal"]["x"]
            dim_theta = data_samples["joint_cal"]["theta"].shape[-1]
            t_stats_null_lc2st_nf = precompute_t_stats_null(
                n_cal=n_cal,
                n_eval=n_eval,
                dim_theta=dim_theta,
                n_trials_null=n_trials_null_precompute,
                kwargs_lc2st=kwargs_lc2st,
                x_cal=x_cal,
                observation_dict=observation_dict,
                methods=["lc2st_nf"],
                metrics=test_stat_names,
                t_stats_null_path="",
                save_results=False,
                load_results=False,
                # args only for c2st
                kwargs_c2st=None,
            )["lc2st_nf"]
        else:
            t_stats_null_lc2st_nf = None

        if "lhpd" in methods:
            x_cal = data_samples["joint_cal"]["x"]
            dim_theta = data_samples["joint_cal"]["theta"].shape[-1]
            t_stats_null_lhpd = precompute_t_stats_null(
                n_cal=n_cal,
                n_eval=n_eval,
                dim_theta=dim_theta,
                n_trials_null=n_trials_null_precompute,
                kwargs_lhpd=kwargs_lhpd,
                x_cal=x_cal,
                observation_dict=observation_dict,
                methods=["lhpd"],
                metrics=["mse"],
                t_stats_null_path="",
                save_results=False,
                load_results=False,
                # args only for c2st and lc2st
                kwargs_c2st=None,
                kwargs_lc2st=None,
            )["lhpd"]
        else:
            t_stats_null_lhpd = None

        # Empirical Power = True Positive Rate (TPR)
        # count rejection of H0 under H1 (p_value <= alpha) for every run
        # and for every observation: [reject(obs1), reject(obs2), ...]
        if compute_emp_power:
            print()
            print("Computing empirical power...")
            print()
            H1_results_dict, _ = compute_test_results_npe_one_run(
                alpha=alpha,
                data_samples=data_samples,
                n_train=n_train,
                observation_dict=observation_dict,
                kwargs_c2st=kwargs_c2st,
                kwargs_lc2st=kwargs_lc2st,
                kwargs_lhpd=kwargs_lhpd,
                n_trials_null=n_trials_null,
                t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,
                t_stats_null_c2st_nf=t_stats_null_c2st_nf,
                t_stats_null_lhpd=t_stats_null_lhpd,
                test_stat_names=test_stat_names,
                methods=methods,
                compute_under_null=False,
                task_path=task_path,
                results_n_train_path="",
                save_results=False,
                seed=n,  # different seed for every run (fixed for reproducibility)
            )
            for m in methods:
                for t_stat_name in test_stat_names:
                    if m == "lhpd" and t_stat_name != "mse":
                        continue
                    p_value_t = H1_results_dict[m]["p_value"][t_stat_name]
                    # increment list of average rejections of H0 under H1
                    emp_power[m][t_stat_name] += (
                        (np.array(p_value_t) <= alpha) * 1 / n_runs
                    )
                    # increment p_values for every observation
                    for num_obs in observation_dict.keys():
                        p_values[m][t_stat_name][num_obs].append(p_value_t[num_obs - 1])

                if n % 10 == 0:
                    torch.save(
                        emp_power[m],
                        result_path / f"emp_power_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                    )
                    torch.save(
                        p_values[m],
                        result_path
                        / f"p_values_obs_per_run_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                    )

        else:
            emp_power, p_values = None, None

        # Type I error = False Positive Rate (FPR)
        # count rejection of H0 under H0 (p_value <= alpha) for every run
        # and for every observation: [reject(obs1), reject(obs2), ...]
        if compute_type_I_error:
            print()
            print("Computing Type I error...")
            print()

            # fixed distribution for null hypothesis (base distribution)
            from scipy.stats import multivariate_normal as mvn

            # generate data for L-C2ST-NF (Q_h0 = P_h0 = N(0,1))
            base_dist_samples_null = mvn(
                mean=torch.zeros(dim_theta), cov=torch.eye(dim_theta)
            ).rvs(
                n_cal, random_state=n + 1
            )  # not same random state as for other data generation (we dont want same data for P and Q)

            # compatible with torch data
            base_dist_samples_null = torch.FloatTensor(base_dist_samples_null)

            H0_results_dict, _ = compute_test_results_npe_one_run(
                alpha=alpha,
                data_samples=data_samples,
                n_train=n_train,
                observation_dict=observation_dict,
                kwargs_c2st=kwargs_c2st,
                kwargs_lc2st=kwargs_lc2st,
                kwargs_lhpd=kwargs_lhpd,
                n_trials_null=n_trials_null,
                t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,
                t_stats_null_c2st_nf=t_stats_null_c2st_nf,
                t_stats_null_lhpd=t_stats_null_lhpd,
                test_stat_names=test_stat_names,
                methods=methods,
                compute_under_null=True,
                base_dist_samples_null=base_dist_samples_null,
                task_path=task_path,
                results_n_train_path="",
                save_results=False,
                seed=n,  # different seed for every run (fixed for reproducibility)
            )
            for m in methods:
                for t_stat_name in test_stat_names:
                    if m == "lhpd" and t_stat_name != "mse":
                        continue

                    p_value_t = H0_results_dict[m]["p_value"][t_stat_name]
                    # increment list of average rejections of H0 under H0
                    type_I_error[m][t_stat_name] += (
                        (np.array(p_value_t) <= alpha) * 1 / n_runs
                    )
                    # increment p_value for every observation
                    for num_obs in observation_dict.keys():
                        p_values_h0[m][t_stat_name][num_obs].append(
                            p_value_t[num_obs - 1]
                        )
                if n % 10 == 0:
                    torch.save(
                        type_I_error[m],
                        result_path / f"type_I_error_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                    )
                    torch.save(
                        p_values_h0[m],
                        result_path
                        / f"p_values_h0__obs_per_run_{m}_n_runs_{n}_n_cal_{n_cal}.pkl",
                    )

        else:
            type_I_error = None
            p_values_h0 = None

    return emp_power, type_I_error, p_values, p_values_h0


def generate_data_one_run(
    n_cal,
    n_eval,
    task,
    observation_dict,
    task_path,
    n_train_list,
    save_data=True,
    load_data=True,
    seed=42,  # fixed seed for reproducibility
):
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load posterior estimator (NF)
    # trained using the code from https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi
    # >>> python run.py --multirun task={task} task.num_simulations={n_train_list} algorithm=npe
    npe = {}
    for N_train in n_train_list:
        npe[N_train] = torch.load(
            task_path / f"npe_{N_train}" / f"posterior_estimator.pkl"
        ).flow
    print(f"Loaded npe posterior estimator trained on {n_train_list} samples.")
    print()

    # get base distribution (same for all npes)
    base_dist = npe[n_train_list[0]].posterior_estimator._distribution

    # generate / load calibration and evaluation datasets
    print(" ==========================================")
    print("     Generating / loading datasets")
    print(" ==========================================")
    print()
    # cal set for fixed task data
    print(f"Calibration set for fixed task data (n_cal={n_cal})")
    try:
        if not load_data:
            raise FileNotFoundError
        base_dist_samples_cal = torch.load(
            task_path / f"base_dist_samples_n_cal_{n_cal}.pkl"
        )
        reference_posterior_samples_cal = torch.load(
            task_path / f"reference_posterior_samples_n_cal_{n_cal}.pkl"
        )
        joint_samples_cal = torch.load(task_path / f"joint_samples_n_cal_{n_cal}.pkl")
        theta_cal = joint_samples_cal["theta"]
        x_cal = joint_samples_cal["x"]
    except FileNotFoundError:
        base_dist_samples_cal = base_dist.sample(n_cal).detach()
        reference_posterior_samples_cal, theta_cal, x_cal = generate_task_data(
            n_cal,
            task,
            list(observation_dict.keys()),
        )
        joint_samples_cal = {"theta": theta_cal, "x": x_cal}
        if save_data:
            torch.save(
                base_dist_samples_cal,
                task_path / f"base_dist_samples_n_cal_{n_cal}.pkl",
            )
            torch.save(
                reference_posterior_samples_cal,
                task_path / f"reference_posterior_samples_n_cal_{n_cal}.pkl",
            )
            torch.save(
                joint_samples_cal,
                task_path / f"joint_samples_n_cal_{n_cal}.pkl",
            )

    # eval set for fixed task data
    print()
    print(f"Evaluation set for fixed task data (n_eval={n_eval})")
    try:
        if not load_data:
            raise FileNotFoundError
        reference_posterior_samples_eval = torch.load(
            task_path / f"reference_posterior_samples_n_eval_{n_eval}.pkl"
        )
        base_dist_samples_eval = torch.load(
            task_path / f"base_dist_samples_n_eval_{n_eval}.pkl"
        )
    except FileNotFoundError:
        reference_posterior_samples_eval, _, _ = generate_task_data(
            n_eval,
            task,
            list(observation_dict.keys()),
            sample_from_joint=False,
        )
        base_dist_samples_eval = base_dist.sample(n_eval).detach()
        if save_data:
            torch.save(
                reference_posterior_samples_eval,
                task_path / f"reference_posterior_samples_n_eval_{n_eval}.pkl",
            )
            torch.save(
                base_dist_samples_eval,
                task_path / f"base_dist_samples_n_eval_{n_eval}.pkl",
            )

    # cal and eval set for every estimator
    print()
    print(
        f"Calibration and evaluation sets for every estimator (n_cal={n_cal}, n_eval={n_eval})"
    )
    npe_samples_obs = {"cal": {}, "eval": {}}
    reference_inv_transform_samples_cal = {}
    reference_inv_transform_samples_eval = {}
    npe_samples_x_cal = {}
    inv_transform_samples_theta_cal = {}

    for N_train in n_train_list:
        print()
        print(f"Data for npe with N_train = {N_train}:")
        npe_samples_obs["cal"][N_train] = {}
        reference_inv_transform_samples_cal[N_train] = {}

        npe_path = task_path / f"npe_{N_train}"
        # ==== C2ST calibration dataset ==== #
        print("     1. C2ST: at fixed observation x_0")
        try:
            if not load_data:
                raise FileNotFoundError
            npe_samples_obs["cal"][N_train] = torch.load(
                npe_path / f"npe_samples_obs_n_cal_{n_cal}.pkl"
            )
            npe_samples_obs["eval"][N_train] = torch.load(
                npe_path / f"npe_samples_obs_n_eval_{n_eval}.pkl"
            )
            reference_inv_transform_samples_cal[N_train] = torch.load(
                npe_path / f"reference_inv_transform_samples_n_cal_{n_cal}.pkl"
            )
            reference_inv_transform_samples_eval[N_train] = torch.load(
                npe_path / f"reference_inv_transform_samples_n_eval_{n_eval}.pkl"
            )
        except FileNotFoundError:
            (
                npe_samples_obs["cal"][N_train],
                reference_inv_transform_samples_cal[N_train],
            ) = generate_npe_data_for_c2st(
                npe[N_train],
                base_dist_samples_cal,
                reference_posterior_samples_cal,
                list(observation_dict.values()),
            )
            (
                npe_samples_obs["eval"][N_train],
                reference_inv_transform_samples_eval[N_train],
            ) = generate_npe_data_for_c2st(
                npe[N_train],
                base_dist_samples_eval,
                reference_posterior_samples_eval,
                list(observation_dict.values()),
            )
            if save_data:
                torch.save(
                    npe_samples_obs["cal"][N_train],
                    npe_path / f"npe_samples_obs_n_cal_{n_cal}.pkl",
                )
                torch.save(
                    reference_inv_transform_samples_cal[N_train],
                    npe_path / f"reference_inv_transform_samples_n_cal_{n_cal}.pkl",
                )
                torch.save(
                    npe_samples_obs["eval"][N_train],
                    npe_path / f"npe_samples_obs_n_eval_{n_eval}.pkl",
                )
                torch.save(
                    reference_inv_transform_samples_eval[N_train],
                    npe_path / f"reference_inv_transform_samples_n_eval_{n_eval}.pkl",
                )

        # ==== L-C2ST calibration dataset ==== #
        print("     2. L-C2ST: for every x in x_cal")
        try:
            if not load_data:
                raise FileNotFoundError
            npe_samples_x_cal[N_train] = torch.load(
                npe_path / f"npe_samples_x_cal_{n_cal}.pkl"
            )
            inv_transform_samples_theta_cal[N_train] = torch.load(
                npe_path / f"inv_transform_samples_theta_cal_{n_cal}.pkl"
            )
        except FileNotFoundError:
            (
                npe_samples_x_cal[N_train],
                inv_transform_samples_theta_cal[N_train],
            ) = generate_npe_data_for_lc2st(
                npe[N_train], base_dist_samples_cal, joint_samples_cal
            )
            if save_data:
                torch.save(
                    npe_samples_x_cal[N_train],
                    npe_path / f"npe_samples_x_cal_{n_cal}.pkl",
                )
                torch.save(
                    inv_transform_samples_theta_cal[N_train],
                    npe_path / f"inv_transform_samples_theta_cal_{n_cal}.pkl",
                )

    base_dist_samples = {"cal": base_dist_samples_cal, "eval": base_dist_samples_eval}
    reference_posterior_samples = {
        "cal": reference_posterior_samples_cal,
        "eval": reference_posterior_samples_eval,
    }
    npe_samples_obs = {"cal": npe_samples_obs["cal"], "eval": npe_samples_obs["eval"]}
    reference_inv_transform_samples = {
        "cal": reference_inv_transform_samples_cal,
        "eval": reference_inv_transform_samples_eval,
    }

    data_dict = {
        "base_dist": base_dist_samples,
        "ref_posterior": reference_posterior_samples,
        "npe_obs": npe_samples_obs,
        "ref_inv_transform": reference_inv_transform_samples,
        "joint_cal": joint_samples_cal,
        "npe_x_cal": npe_samples_x_cal,
        "inv_transform_theta_cal": inv_transform_samples_theta_cal,
    }

    return data_dict


def compute_test_results_npe_one_run(
    data_samples,
    n_train,
    observation_dict,
    kwargs_c2st,
    kwargs_lc2st,
    kwargs_lhpd,
    n_trials_null,
    t_stats_null_c2st_nf,
    t_stats_null_lc2st_nf,
    t_stats_null_lhpd,
    task_path,
    results_n_train_path,
    test_stat_names=["accuracy", "mse", "div"],
    methods=["c2st", "lc2st", "c2st_nf", "lc2st_nf", "lc2st_nf_perm", "lhpd"],
    alpha=0.05,
    compute_under_null=False,
    base_dist_samples_null=None,
    save_results=True,
    seed=42,  # fix seed for reproducibility
):
    # extract data samples independent from estimator
    base_dist_samples = data_samples["base_dist"]
    reference_posterior_samples = data_samples["ref_posterior"]
    theta_cal, x_cal = data_samples["joint_cal"].values()
    # extract data samples for the considered "n_train"-estimator (npe)
    npe_samples_obs = {k: v[n_train] for k, v in data_samples["npe_obs"].items()}
    reference_inv_transform_samples = {
        k: v[n_train] for k, v in data_samples["ref_inv_transform"].items()
    }
    npe_samples_x_cal = data_samples["npe_x_cal"][n_train]
    inv_transform_samples_theta_cal = data_samples["inv_transform_theta_cal"][n_train]

    n_cal = len(base_dist_samples["cal"])
    n_eval = len(base_dist_samples["eval"])

    print()
    print(" ==========================================")
    print("     COMPUTING TEST RESULTS")
    print(" ==========================================")
    print()
    print(f"N_train = {n_train}")

    result_path = task_path / f"npe_{n_train}" / results_n_train_path
    if compute_under_null:
        result_path = result_path / "null"
    if save_results and not os.path.exists(result_path):
        os.makedirs(result_path)

    train_runtime = dict(zip(methods, [0 for _ in methods]))
    results_dict = dict(zip(methods, [{} for _ in methods]))

    result_keys = ["reject", "p_value", "t_stat", "t_stats_null", "run_time"]
    trained_clfs_lc2st_nf = None
    runtime_lc2st_nf = None
    for m in methods:
        results_dict[m] = dict(
            zip(
                result_keys,
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in result_keys
                ],
            )
        )
        try:
            results_dict[m] = torch.load(
                result_path / f"{m}_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
            )
            if "l" in m:
                train_runtime[m] = torch.load(
                    result_path / f"runtime_{m}_n_cal_{n_cal}.pkl"
                )
        except FileNotFoundError:
            if m == "c2st" or m == "c2st_nf":
                print()
                print("     C2ST: train for every observation x_0")
                print()
                # loop over observations x_0
                for n_obs in tqdm(
                    observation_dict.keys(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    if m == "c2st":
                        # class 0: T ~ p_est(theta | x_0) vs. p_ref(theta | x_0)
                        P, Q = (
                            npe_samples_obs["cal"][n_obs],
                            reference_posterior_samples["cal"][n_obs],
                        )
                        P_eval, Q_eval = (
                            npe_samples_obs["eval"][n_obs],
                            reference_posterior_samples["eval"][n_obs],
                        )
                        # permutation method
                        if compute_under_null:
                            P, Q = permute_data(P, Q, seed=seed)
                            P_eval, Q_eval = permute_data(P_eval, Q_eval, seed=seed)
                        t_stats_null = None

                    elif m == "c2st_nf":
                        # class 0: Z ~ N(0,I) vs. Z ~ T^{-1}(p_ref(theta | x_0))
                        P, Q = (
                            base_dist_samples["cal"],
                            reference_inv_transform_samples["cal"][n_obs],
                        )
                        P_eval, Q_eval = (
                            base_dist_samples["eval"],
                            reference_inv_transform_samples["eval"][n_obs],
                        )
                        # no permuation method and precomputed test stats under null
                        if compute_under_null:
                            Q = base_dist_samples_null
                        t_stats_null = t_stats_null_c2st_nf

                    t0 = time.time()
                    c2st_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_c2st,
                        metrics=test_stat_names,
                        t_stats_null=t_stats_null,
                        # kwargs for t_stats_c2st
                        P=P,
                        Q=Q,
                        P_eval=P_eval,
                        Q_eval=Q_eval,
                        use_permutation=True,
                        n_trials_null=n_trials_null,
                        # kwargs for c2st_scores
                        **kwargs_c2st,
                    )
                    runtime = time.time() - t0

                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                results_dict[m][result_name][t_stat_name].append(
                                    runtime
                                )
                            else:
                                results_dict[m][result_name][t_stat_name].append(
                                    c2st_results_obs[i][t_stat_name]
                                )

            elif "lc2st" in m:
                print()
                print("     L-C2ST: amortized")
                print()

                x_P, x_Q = x_cal, x_cal

                if m == "lc2st":
                    P, Q = npe_samples_x_cal, theta_cal
                    P_eval_obs = npe_samples_obs["eval"]

                if m == "lc2st_nf" or m == "lc2st_nf_perm":
                    P, Q = base_dist_samples["cal"], inv_transform_samples_theta_cal
                    P_eval_obs = {
                        n_obs: base_dist_samples["eval"]
                        for n_obs in observation_dict.keys()
                    }
                if m == "lc2st" or m == "lc2st_nf_perm":
                    # permutation method and no precomputed test stats under null
                    t_stats_null = {n_obs: None for n_obs in observation_dict.keys()}
                    if compute_under_null:
                        joint_P_x = torch.cat([P, x_P], dim=1)
                        joint_Q_x = torch.cat([Q, x_Q], dim=1)
                        joint_P_x, joint_Q_x = permute_data(
                            joint_P_x, joint_Q_x, seed=seed
                        )
                        P, x_P = (
                            joint_P_x[:, : P.shape[-1]],
                            joint_P_x[:, P.shape[-1] :],
                        )
                        Q, x_Q = (
                            joint_Q_x[:, : Q.shape[-1]],
                            joint_Q_x[:, Q.shape[-1] :],
                        )
                else:
                    # no permutation method and precomputed test stats under null
                    t_stats_null = t_stats_null_lc2st_nf
                    if compute_under_null:
                        Q = base_dist_samples_null

                # train classifier on the joint
                print(f"{m}: TRAINING CLASSIFIER on the joint ...")
                print()
                print("... for the observed data")
                if m == "lc2st" or compute_under_null or trained_clfs_lc2st_nf is None:
                    t0 = time.time()
                    _, _, trained_clfs_lc2st = lc2st_scores(
                        P=P,
                        Q=Q,
                        x_P=x_P,
                        x_Q=x_Q,
                        x_eval=None,
                        eval=False,
                        **kwargs_lc2st,
                    )
                    runtime = time.time() - t0
                    train_runtime[m] = runtime
                    if "lc2st_nf" in m and not compute_under_null:
                        trained_clfs_lc2st_nf = trained_clfs_lc2st
                        runtime_lc2st_nf = runtime
                    train_runtime[m] = runtime
                else:
                    print("     Using classifier trained for lc2st_nf method")
                    trained_clfs_lc2st = trained_clfs_lc2st_nf
                    runtime = runtime_lc2st_nf
                    train_runtime[m] = runtime
                if save_results:
                    torch.save(runtime, result_path / f"runtime_{m}_n_cal_{n_cal}.pkl")

                if t_stats_null[list(observation_dict.keys())[0]] is None:
                    print("... under the null hypothesis")
                    # train classifier on the joint under null
                    _, _, trained_clfs_null_lc2st = t_stats_lc2st(
                        null_hypothesis=True,
                        n_trials_null=n_trials_null,
                        use_permutation=True,
                        P=P,
                        Q=Q,
                        x_P=x_P,
                        x_Q=x_Q,
                        x_eval=None,
                        P_eval=None,
                        return_clfs_null=True,
                        # kwargs for lc2st_sores
                        eval=False,
                        **kwargs_lc2st,
                    )
                else:
                    trained_clfs_null_lc2st = None

                for num_observation, observation in tqdm(
                    observation_dict.items(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    t0 = time.time()
                    lc2st_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_lc2st,
                        metrics=test_stat_names,
                        t_stats_null=t_stats_null[num_observation],
                        # kwargs for t_stats_estimator
                        x_eval=observation,
                        P_eval=P_eval_obs[num_observation],
                        Q_eval=None,
                        use_permutation=True,
                        n_trials_null=n_trials_null,
                        return_probas=False,
                        # unnessary args as we have pretrained clfs
                        P=P,
                        Q=Q,
                        x_P=x_P,
                        x_Q=x_Q,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lc2st,
                        trained_clfs_null=trained_clfs_null_lc2st,
                        # kwargs for lc2st_scores
                        **kwargs_lc2st,
                    )
                    runtime = (time.time() - t0) / n_trials_null

                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                results_dict[m][result_name][t_stat_name].append(
                                    runtime
                                )
                            else:
                                results_dict[m][result_name][t_stat_name].append(
                                    lc2st_results_obs[i][t_stat_name]
                                )

            elif m == "lhpd":
                print()
                print("     Local HPD: amortized")
                print()

                npe = torch.load(
                    task_path / f"npe_{n_train}" / "posterior_estimator.pkl"
                ).flow

                def npe_sample_fn(n_samples, x):
                    return sample_from_npe_obs(npe, x, n_samples=n_samples)

                print(f"{m}: TRAINING CLASSIFIER on the joint ...")
                print()
                print("... for the observed data")
                t0 = time.time()
                if compute_under_null:
                    _, _, trained_clfs_lhpd = t_stats_lhpd(
                        null_hypothesis=True,
                        n_trials_null=1,
                        Y=theta_cal,
                        X=x_cal,
                        x_eval=None,
                        est_log_prob_fn=None,
                        est_sample_fn=None,
                        eval=False,
                        return_clfs_null=True,
                        **kwargs_lhpd,
                    )
                    trained_clfs_lhpd = trained_clfs_lhpd[0]
                else:
                    _, _, trained_clfs_lhpd = lhpd_scores(
                        Y=theta_cal,
                        X=x_cal,
                        est_log_prob_fn=npe.log_prob,
                        est_sample_fn=npe_sample_fn,
                        return_clfs=True,
                        x_eval=None,
                        eval=False,
                        **kwargs_lhpd,
                    )
                runtime = time.time() - t0
                train_runtime[m] = runtime

                for num_observation, observation in tqdm(
                    observation_dict.items(),
                    desc=f"{m}: Computing T for every observation x_0",
                ):
                    t0 = time.time()
                    lhpd_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_lhpd,
                        metrics=["mse"],
                        t_stats_null=t_stats_null_lhpd[num_observation],
                        # kwargs for t_stats_estimator
                        x_eval=observation,
                        Y=theta_cal,
                        X=x_cal,
                        n_trials_null=n_trials_null,
                        return_r_alphas=False,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lhpd,
                        # kwargs for lhpd_scores
                        est_log_prob_fn=None,
                        est_sample_fn=None,
                        **kwargs_lhpd,
                    )
                    runtime = (time.time() - t0) / n_trials_null

                    for i, result_name in enumerate(result_keys):
                        if result_name == "run_time":
                            results_dict[m][result_name]["mse"].append(runtime)
                        else:
                            results_dict[m][result_name]["mse"].append(
                                lhpd_results_obs[i]["mse"]
                            )

            if save_results:
                torch.save(
                    results_dict[m],
                    result_path / f"{m}_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )
                torch.save(
                    train_runtime[m], result_path / f"runtime_{m}_n_cal_{n_cal}.pkl"
                )

    return results_dict, train_runtime


def precompute_t_stats_null(
    metrics,
    n_cal,
    n_eval,
    dim_theta,
    n_trials_null,
    observation_dict,
    t_stats_null_path,
    kwargs_c2st,
    kwargs_lc2st,
    x_cal,
    kwargs_lhpd={},
    methods=["c2st_nf", "lc2st_nf", "lhpd"],
    save_results=True,
    load_results=True,
):
    # fixed distribution for null hypothesis (base distribution)
    from scipy.stats import multivariate_normal as mvn

    if save_results and not os.path.exists(t_stats_null_path):
        os.makedirs(t_stats_null_path)

    P_dist_null = mvn(mean=torch.zeros(dim_theta), cov=torch.eye(dim_theta))
    list_P_null = [
        P_dist_null.rvs(n_cal, random_state=t) for t in range(2 * n_trials_null)
    ]
    list_P_eval_null = [
        P_dist_null.rvs(n_eval, random_state=t) for t in range(2 * n_trials_null)
    ]

    t_stats_null_dict = dict(zip(methods, [{} for _ in methods]))

    for m in methods:
        try:
            if not load_results:
                raise FileNotFoundError
            t_stats_null = torch.load(
                t_stats_null_path
                / f"{m}_stats_null_nt_{n_trials_null}_n_cal_{n_cal}.pkl"
            )
            print()
            print(f"Loaded pre-computed test statistics for {m}-H_0")
        except FileNotFoundError:
            print()
            print(
                f"Pre-compute test statistics for {m}-H_0 (N_cal={n_cal}, n_trials={n_trials_null})"
            )
            if m == "c2st_nf":
                print()
                print("C2ST: TRAIN / EVAL CLASSIFIERS ...")
                print()
                t_stats_null = t_stats_c2st(
                    null_hypothesis=True,
                    metrics=metrics,
                    list_P_null=list_P_null,
                    list_P_eval_null=list_P_eval_null,
                    use_permutation=False,
                    n_trials_null=n_trials_null,
                    # required kwargs for t_stats_c2st
                    P=None,
                    Q=None,
                    # kwargs for c2st_scores
                    **kwargs_c2st,
                )
            elif m == "lc2st_nf":
                # train clfs on joint samples
                print()
                print("L-C2ST: TRAINING CLASSIFIERS on the joint ...")
                print()
                _, _, trained_clfs_null = t_stats_lc2st(
                    null_hypothesis=True,
                    metrics=metrics,
                    list_P_null=list_P_null,
                    list_x_P_null=[x_cal] * len(list_P_null),
                    use_permutation=False,
                    n_trials_null=n_trials_null,
                    return_clfs_null=True,
                    # required kwargs for t_stats_lc2st
                    P=None,
                    Q=None,
                    x_P=None,
                    x_Q=None,
                    P_eval=None,
                    list_P_eval_null=list_P_eval_null,
                    x_eval=None,
                    # kwargs for lc2st_scores
                    eval=False,
                    **kwargs_lc2st,
                )
                print()
                print("L-C2ST: Evaluate for every observation ...")
                t_stats_null = {}
                for num_obs, observation in observation_dict.items():
                    t_stats_null[num_obs] = t_stats_lc2st(
                        null_hypothesis=True,
                        metrics=metrics,
                        list_P_null=list_P_null,
                        list_P_eval_null=list_P_eval_null,
                        # ==== added for LC2ST ====
                        list_x_P_null=[x_cal] * len(list_P_null),
                        x_eval=observation,
                        return_probas=False,
                        # =========================
                        use_permutation=False,
                        n_trials_null=n_trials_null,
                        trained_clfs_null=trained_clfs_null,
                        # required kwargs for t_stats_lc2st
                        P=None,
                        Q=None,
                        x_P=None,
                        x_Q=None,
                        P_eval=None,
                        # kwargs for lc2st_scores
                        **kwargs_lc2st,
                    )
            elif m == "lhpd":
                # train clfs on joint samples
                print()
                print("L-HPD: TRAINING CLASSIFIERS on the joint ...")
                print()
                _, _, trained_clfs_null = t_stats_lhpd(
                    metrics=["mse"],
                    Y=list_P_null[0],  # for dim inside lhpd_scores
                    X=x_cal,
                    null_hypothesis=True,
                    n_trials_null=n_trials_null,
                    return_clfs_null=True,
                    # required kwargs for t_stats_lhpd
                    x_eval=None,
                    # kwargs for lhpd_scores
                    eval=False,
                    est_log_prob_fn=None,
                    est_sample_fn=None,
                    **kwargs_lhpd,
                )
                print()
                print("L-HPD: Evaluate for every observation ...")
                t_stats_null = {}
                for num_obs, observation in observation_dict.items():
                    t_stats_null[num_obs] = t_stats_lhpd(
                        metrics=["mse"],
                        Y=list_P_null[0],  # for dim inside lhpd_scores
                        X=x_cal,
                        null_hypothesis=True,
                        n_trials_null=n_trials_null,
                        trained_clfs_null=trained_clfs_null,
                        return_clfs_null=False,
                        return_r_alphas=False,
                        # required kwargs for t_stats_lhpd
                        x_eval=observation,
                        # kwargs for lhpd_scores
                        est_log_prob_fn=None,
                        est_sample_fn=None,
                        **kwargs_lhpd,
                    )

            if save_results:
                torch.save(
                    t_stats_null,
                    t_stats_null_path
                    / f"{m}_stats_null_nt_{n_trials_null}_n_cal_{n_cal}.pkl",
                )
        t_stats_null_dict[m] = t_stats_null

    return t_stats_null_dict


if __name__ == "__main__":
    import torch
    import sbibm
    from valdiags.localHPD import hpd_ranks, t_stats_lhpd
    from tasks.sbibm.npe_utils import sample_from_npe_obs

    import matplotlib.pyplot as plt

    task = sbibm.get_task("two_moons")
    npe = torch.load(
        "saved_experiments/neurips_2023/exp_2/two_moons/npe_1000/posterior_estimator.pkl"
    ).flow
    joint_samples = torch.load(
        "saved_experiments/neurips_2023/exp_2/two_moons/joint_samples_n_cal_10000.pkl"
    )
    x, theta = joint_samples["x"], joint_samples["theta"]
    observation = task.get_observation(1)

    def sample_fn(n_samples, x):
        return sample_from_npe_obs(npe, x, n_samples)

    alphas = np.linspace(0.1, 0.9, 20)
    t_stat, r_alphas = t_stats_lhpd(
        Y=theta[:100],
        X=x[:100],
        alphas=alphas,
        x_eval=observation,
        est_log_prob_fn=npe.log_prob,
        est_sample_fn=sample_fn,
        return_r_alphas=True,
    )
    # t_stats_null, r_alphas_null = t_stats_lhpd(
    #     null_hypothesis=True,
    #     Y=theta[:100],
    #     X=x[:100],
    #     x_eval=observation,
    #     alphas=alphas,
    #     est_log_prob_fn=npe.log_prob,
    #     est_sample_fn=sample_fn,
    #     return_r_alphas=True,
    # )

    t_stats_null = precompute_t_stats_null(
        methods=["lhpd"],
        x_cal=x[:100],
        metrics=None,
        n_cal=100,
        n_eval=None,
        dim_theta=theta.shape[-1],
        n_trials_null=10,
        observation_dict={1: observation},
        t_stats_null_path="",
        save_results=False,
        load_results=False,
        kwargs_c2st={},
        kwargs_lc2st={},
        kwargs_lhpd={},
        alphas=alphas,
    )["lhpd"][1]

    from valdiags.test_utils import compute_pvalue

    pvalue = compute_pvalue(t_stat, t_stats_null)
    print(t_stat)
    print(pvalue)

    alphas = np.concatenate([np.array([0]), alphas, np.array([1])])
    r_alphas = {**{0.0: 0.0}, **r_alphas, **{1.0: 1.0}}
    plt.plot(alphas, r_alphas.values())

    # import pandas as pd

    # r_alphas_null = {**{0.0: [0.0] * 100}, **r_alphas_null, **{1.0: [1.0] * 100}}
    # lower_band = pd.DataFrame(r_alphas_null).quantile(q=0.05 / 2, axis=0)
    # upper_band = pd.DataFrame(r_alphas_null).quantile(q=1 - 0.05 / 2, axis=0)

    # plt.fill_between(alphas, lower_band, upper_band, color="grey", alpha=0.2)
    plt.show()
