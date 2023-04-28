import os

import torch
import numpy as np

from tqdm import tqdm
import time

from sklearn.neural_network import MLPClassifier

from valdiags.test_utils import eval_htest, permute_data
from valdiags.vanillaC2ST import t_stats_c2st
from valdiags.localC2ST import t_stats_lc2st, train_lc2st

from tasks.sbibm.data_generators import (
    generate_task_data,
    generate_npe_data_for_c2st,
    generate_npe_data_for_lc2st,
)


def l_c2st_results_n_train(
    task,
    n_cal,
    n_eval,
    observation_dict,
    n_train_list,
    alpha,
    c2st_stats_fn,
    lc2st_stats_fn,
    c2st_stats_null_nf,
    lc2st_stats_null_nf,
    task_path,
    results_n_train_path,
    methods=["c2st", "lc2st", "lc2st_nf"],
    test_stat_names=["accuracy", "mse", "div"],
):
    # load posterior estimator (NF)
    # trained using the code from https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi
    # >>> python run.py --multirun task={task} task.num_simulations={n_train_list} algorithm=npe
    npe = {}
    for N_train in n_train_list:
        npe[N_train] = torch.load(
            task_path / f"npe_{N_train}" / f"posterior_estimator.pkl"
        ).flow
    print(f"Loaded npe posterior estimators trained on {n_train_list} samples.")
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
        torch.save(
            base_dist_samples_cal, task_path / f"base_dist_samples_n_cal_{n_cal}.pkl"
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
    print(f"Evaluation set for fixed task data (n_eval={n_eval})")
    try:
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
        torch.save(
            reference_posterior_samples_eval,
            task_path / f"reference_posterior_samples_n_eval_{n_eval}.pkl",
        )
        torch.save(
            base_dist_samples_eval, task_path / f"base_dist_samples_n_eval_{n_eval}.pkl"
        )

    # cal and eval set for every estimator
    print(
        f"Calibration and evaluation sets for every estimator (n_cal={n_cal}, n_eval={n_eval})"
    )
    npe_samples_obs_cal = {}
    npe_samples_obs_eval = {}
    reference_inv_transform_samples_cal = {}
    reference_inv_transform_samples_eval = {}
    npe_samples_x_cal = {}
    inv_transform_samples_theta_cal = {}

    for N_train in n_train_list:
        print()
        print(f"Data for npe with N_train = {N_train}:")
        npe_samples_obs_cal[N_train] = {}
        reference_inv_transform_samples_cal[N_train] = {}
        # ==== C2ST calibration dataset ==== #
        print("     1. C2ST: at fixed observation x_0")
        try:
            npe_samples_obs_cal[N_train] = torch.load(
                task_path / f"npe_{N_train}" / f"npe_samples_obs_n_cal_{n_cal}.pkl"
            )
            npe_samples_obs_eval[N_train] = torch.load(
                task_path / f"npe_{N_train}" / f"npe_samples_obs_n_eval_{n_eval}.pkl"
            )
            reference_inv_transform_samples_cal[N_train] = torch.load(
                task_path
                / f"npe_{N_train}"
                / f"reference_inv_transform_samples_n_cal_{n_cal}.pkl"
            )
            reference_inv_transform_samples_eval[N_train] = torch.load(
                task_path
                / f"npe_{N_train}"
                / f"reference_inv_transform_samples_n_eval_{n_eval}.pkl"
            )
        except FileNotFoundError:
            (
                npe_samples_obs_cal[N_train],
                reference_inv_transform_samples_cal[N_train],
            ) = generate_npe_data_for_c2st(
                npe[N_train],
                base_dist_samples_cal,
                reference_posterior_samples_cal,
                list(observation_dict.values()),
            )
            (
                npe_samples_obs_eval[N_train],
                reference_inv_transform_samples_eval[N_train],
            ) = generate_npe_data_for_c2st(
                npe[N_train],
                base_dist_samples_eval,
                reference_posterior_samples_eval,
                list(observation_dict.values()),
            )
            torch.save(
                npe_samples_obs_cal[N_train],
                task_path / f"npe_{N_train}" / f"npe_samples_obs_n_cal_{n_cal}.pkl",
            )
            torch.save(
                reference_inv_transform_samples_cal[N_train],
                task_path
                / f"npe_{N_train}"
                / f"reference_inv_transform_samples_n_cal_{n_cal}.pkl",
            )
            torch.save(
                npe_samples_obs_eval[N_train],
                task_path / f"npe_{N_train}" / f"npe_samples_obs_n_eval_{n_eval}.pkl",
            )
            torch.save(
                reference_inv_transform_samples_eval[N_train],
                task_path
                / f"npe_{N_train}"
                / f"reference_inv_transform_samples_n_eval_{n_eval}.pkl",
            )

        # ==== L-C2ST calibration dataset ==== #
        print("     2. L-C2ST: for every x in x_cal")
        try:
            npe_samples_x_cal[N_train] = torch.load(
                task_path / f"npe_{N_train}" / f"npe_samples_x_cal_{n_cal}.pkl"
            )
            inv_transform_samples_theta_cal[N_train] = torch.load(
                task_path
                / f"npe_{N_train}"
                / f"inv_transform_samples_theta_cal_{n_cal}.pkl"
            )
        except FileNotFoundError:
            (
                npe_samples_x_cal[N_train],
                inv_transform_samples_theta_cal[N_train],
            ) = generate_npe_data_for_lc2st(
                npe[N_train], base_dist_samples_cal, joint_samples_cal
            )
            torch.save(
                npe_samples_x_cal[N_train],
                task_path / f"npe_{N_train}" / f"npe_samples_x_cal_{n_cal}.pkl",
            )
            torch.save(
                inv_transform_samples_theta_cal[N_train],
                task_path
                / f"npe_{N_train}"
                / f"inv_transform_samples_theta_cal_{n_cal}.pkl",
            )

    # Testing with all methods for all N_train
    print()
    print(" ==========================================")
    print("     COMPUTING TEST RESULTS")
    print(" ==========================================")

    avg_result_keys = {
        "TPR": "reject",
        "p_value_mean": "p_value",
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
    train_runtime = {
        "lc2st": [],
        "lc2st_nf": [],
        "c2st": [0] * len(n_train_list),
        "c2st_nf": [0] * len(n_train_list),
    }

    for N_train in n_train_list:
        print()
        print(f"N_train = {N_train}")
        print("     1. C2ST: for every x_0 in x_test")

        result_path = task_path / f"npe_{N_train}" / results_n_train_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        try:
            if "c2st" in methods:
                c2st_results = torch.load(
                    result_path / f"c2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
                )
            if "c2st_nf" in methods:
                c2st_nf_results = torch.load(
                    result_path / f"c2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
                )
        except FileNotFoundError:
            result_keys = ["reject", "p_value", "t_stat", "t_stats_null", "run_time"]
            c2st_results = dict(
                zip(
                    result_keys,
                    [
                        dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                        for _ in result_keys
                    ],
                )
            )
            c2st_nf_results = dict(
                zip(
                    result_keys,
                    [
                        dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                        for _ in result_keys
                    ],
                )
            )

            # C2ST:
            # class 0: T ~ p_est(theta | x_0) vs. p_ref(theta | x_0)
            if "c2st" in methods:
                # loop over observations x_0
                for n_obs in tqdm(
                    observation_dict.keys(),
                    desc=f"Computing t_stats for every observation x_0",
                ):
                    t0 = time.time()
                    c2st_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=c2st_stats_fn,
                        metrics=test_stat_names,  # vanilla C2ST and Reg-C2ST
                        # kwargs for t_stats_estimator
                        use_permutation=True,  # it takes to long to sample new data from the reference / npe
                        P=npe_samples_obs_cal[N_train][n_obs],
                        Q=reference_posterior_samples_cal[n_obs],
                        P_eval=npe_samples_obs_eval[N_train][n_obs],
                        Q_eval=reference_posterior_samples_eval[n_obs],
                    )
                    runtime = time.time() - t0
                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                c2st_results[result_name][t_stat_name].append(runtime)
                            else:
                                c2st_results[result_name][t_stat_name].append(
                                    c2st_results_obs[i][t_stat_name]
                                )
                torch.save(
                    c2st_results,
                    result_path / f"c2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

            # C2ST-NF:
            # class 0: Z ~ N(0,I) vs. Z ~ T^{-1}(p_ref(theta | x_0))
            if "c2st_nf" in methods:
                # loop over observations x_0
                for n_obs in tqdm(
                    observation_dict.keys(),
                    desc=f"Computing t_stats for every observation x_0",
                ):
                    t0 = time.time()
                    c2st_nf_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=c2st_stats_fn,
                        metrics=test_stat_names,  # vanilla C2ST and Reg-C2ST
                        t_stats_null=c2st_stats_null_nf,  # use precomputed test statistics
                        # kwargs for t_stats_estimator
                        P=base_dist_samples_cal,
                        Q=reference_inv_transform_samples_cal[N_train][n_obs],
                        P_eval=base_dist_samples_eval,
                        Q_eval=reference_inv_transform_samples_eval[N_train][n_obs],
                    )
                    runtime = time.time() - t0
                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                c2st_nf_results[result_name][t_stat_name].append(
                                    runtime
                                )
                            else:
                                c2st_nf_results[result_name][t_stat_name].append(
                                    c2st_nf_results_obs[i][t_stat_name]
                                )
                torch.save(
                    c2st_nf_results,
                    result_path / f"c2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

        print("     2. L-C2ST: amortized")
        try:
            if "lc2st" in methods:
                lc2st_results = torch.load(
                    result_path / f"lc2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
                )
            if "lc2st_nf" in methods:
                lc2st_nf_results = torch.load(
                    result_path / f"lc2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
                )
        except FileNotFoundError:
            result_keys = ["reject", "p_value", "t_stat", "t_stats_null"]
            lc2st_results = dict(
                zip(
                    result_keys,
                    [
                        dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                        for _ in result_keys
                    ],
                )
            )
            lc2st_nf_results = dict(
                zip(
                    result_keys,
                    [
                        dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                        for _ in result_keys
                    ],
                )
            )

            print("TRAINING CLASSIFIER on the joint ...")
            # L-C2ST:
            if "lc2st" in methods:
                # train classifier on the joint
                t0 = time.time()
                trained_clf_lc2st = train_lc2st(
                    P=npe_samples_x_cal[N_train],
                    Q=theta_cal,
                    x_P=x_cal,
                    x_Q=x_cal,
                )
                runtime = time.time() - t0
                train_runtime["lc2st"].append(runtime)

                for num_observation, observation in tqdm(
                    observation_dict.items(),
                    desc=f"Computing t_stats for every observation x_0",
                ):
                    t0 = time.time()
                    lc2st_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=lc2st_stats_fn,
                        metrics=test_stat_names,  # vanilla C2ST and Reg-C2ST
                        # kwargs for t_stats_estimator
                        use_permutation=True,  # we have no new data of the estimator on the joint (no new simualtions x)
                        P=npe_samples_x_cal[N_train],
                        Q=theta_cal,
                        x_P=x_cal,
                        x_Q=x_cal,
                        x_eval=observation,
                        P_eval=npe_samples_obs_eval[N_train][num_observation],
                        Q_eval=None,
                        trained_clfs=[trained_clf_lc2st],
                    )
                    runtime = time.time() - t0
                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                lc2st_results[result_name][t_stat_name].append(runtime)
                            else:
                                lc2st_results[result_name][t_stat_name].append(
                                    lc2st_results_obs[i][t_stat_name]
                                )
                torch.save(
                    lc2st_results,
                    result_path / f"lc2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

            # L-C2ST-NF:
            if "lc2st_nf" in methods:
                # train classifier on the joint
                t0 = time.time()
                trained_clf_lc2st_nf = train_lc2st(
                    P=base_dist_samples_cal,
                    Q=inv_transform_samples_theta_cal[N_train],
                    x_P=x_cal,
                    x_Q=x_cal,
                )
                runtime = time.time() - t0
                train_runtime["lc2st_nf"].append(runtime)

                for num_observation, observation in tqdm(
                    observation_dict.items(),
                    desc=f"Computing t_stats for every observation x_0",
                ):
                    t0 = time.time()
                    lc2st_nf_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=lc2st_stats_fn,
                        metrics=test_stat_names,
                        t_stats_null=lc2st_stats_null_nf,  # use precomputed test statistics
                        # kwargs for t_stats_estimator
                        P=base_dist_samples_cal,
                        Q=inv_transform_samples_theta_cal[N_train],
                        x_P=x_cal,
                        x_Q=x_cal,
                        x_eval=observation,
                        P_eval=base_dist_samples_eval,
                        Q_eval=None,
                        trained_clfs=[trained_clf_lc2st_nf],
                    )
                    runtime = time.time() - t0

                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                lc2st_nf_results[result_name][t_stat_name].append(
                                    runtime
                                )
                            else:
                                lc2st_nf_results[result_name][t_stat_name].append(
                                    lc2st_nf_results_obs[i][t_stat_name]
                                )
                torch.save(
                    lc2st_nf_results,
                    result_path / f"lc2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

        for method, results in zip(
            ["c2st", "c2st_nf", "lc2st", "lc2st_nf"],
            [c2st_results, c2st_nf_results, lc2st_results, lc2st_nf_results],
        ):
            if method in methods:
                for k, v in avg_result_keys.items():
                    for t_stat_name in test_stat_names:
                        if "std" in k:
                            avg_results[method][k][t_stat_name].append(
                                np.std(results[v][t_stat_name])
                            )
                        else:
                            if "t_stat" in k and t_stat_name == "mse":
                                avg_results[method][k][t_stat_name].append(
                                    np.mean(results[v][t_stat_name])
                                    + 0.5  # for comparison with other t_stats
                                )
                            else:
                                avg_results[method][k][t_stat_name].append(
                                    np.mean(results[v][t_stat_name])
                                )
    return avg_results


def compute_emp_power_l_c2st(
    n_runs,
    alpha,
    task,
    npe,
    observation_dict,
    n_cal,
    n_eval,
    methods=["c2st", "lc2st", "lc2st_nf", "lc2st_nf_perm"],
    test_stat_names=["accuracy", "mse", "div"],
    clf_lc2st=MLPClassifier(alpha=0, max_iter=25000),
    compute_type_I_error=False,
    t_stats_null_lc2st_nf=None,
    **kwargs_eval_htest,
):
    """Compute the empirical power of the (L)C2ST methods for a given task and npe-flow.
    We also compute the type I error if specified. For now all methods will use
    the permutation method, except fpr lc2st_nf, if specified.
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
        kwargs_eval_htest (dict): kwargs for `valdiags.test_utils.eval_htest`.
    Returns:
        dict: dict of empirical power values for every test statistic computed in eval_htest_fn.
            The dict keys are the test statistic names.
            The dict values are lists of length len(num_observation_list) of empirical power values.
    """
    # initialize dict of p-values
    p_values = {}
    p_values_h0 = {}
    for method in methods:
        p_values[method] = dict(
            zip(
                observation_dict.keys(),
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in observation_dict
                ],
            )
        )
        p_values_h0[method] = dict(
            zip(
                observation_dict.keys(),
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in observation_dict
                ],
            )
        )

    for _ in tqdm(
        range(n_runs),
        desc=f"Computing empirical power over all observations",
    ):
        # Generate data from base distribution
        base_dist_samples = npe.posterior_estimator._distribution.sample(
            n_cal + n_eval
        ).detach()

        # Generate data from estimator for every observation
        # (used in all methods except nf)
        if sum(["nf" in method for method in methods]) != len(methods):
            npe_samples_obs_eval, _ = generate_npe_data_for_c2st(
                npe=npe,
                base_dist_samples=base_dist_samples[n_cal:],
                reference_posterior_samples=None,
                observation_list=list(observation_dict.values()),
            )

        # Generate data from the joint
        # (used in all lc2st methods)
        if sum(["lc2st" in method for method in methods]) != 0:
            _, theta_cal, x_cal = generate_task_data(
                n_samples=n_cal,
                task=task,
                num_observation_list=None,
            )

        if "c2st" in methods:
            # Generate data from reference posterior
            reference_posterior_samples, _, _ = generate_task_data(
                n_samples=n_cal + n_eval,
                task=task,
                num_observation_list=list(observation_dict.keys()),
                sample_from_joint=False,
            )
            # Generate data from npe
            (
                npe_samples_obs_cal,
                _,
            ) = generate_npe_data_for_c2st(
                npe=npe,
                base_dist_samples=base_dist_samples[:n_cal],
                reference_posterior_samples=None,
                observation_list=list(observation_dict.values()),
            )

            # evaluate test
            for num_observation in tqdm(
                observation_dict.keys(),
                desc=f"Testing for observation {num_observation}",
            ):
                _, p_value, _, _ = eval_htest(
                    t_stats_estimator=t_stats_c2st,
                    # args for t_stats_estimator
                    P=npe_samples_obs_cal[num_observation],
                    Q=reference_posterior_samples[num_observation][:n_cal],
                    P_eval=npe_samples_obs_eval[num_observation],
                    Q_eval=reference_posterior_samples[num_observation][n_cal:],
                    **kwargs_eval_htest,
                )
                for test_stat_name in test_stat_names:
                    p_values["c2st"][num_observation][test_stat_name].append(
                        p_value[test_stat_name]
                    )

                if compute_type_I_error:
                    P_perm, Q_perm = permute_data(
                        npe_samples_obs_cal[num_observation],
                        reference_posterior_samples[num_observation][:n_cal],
                    )
                    _, p_value_h0, _, _ = eval_htest(
                        t_stats_estimator=t_stats_c2st,
                        # args for t_stats_estimator
                        P=P_perm,
                        Q=Q_perm,
                        P_eval=npe_samples_obs_eval[num_observation],
                        Q_eval=reference_posterior_samples[num_observation][n_cal:],
                        use_permutation=True,
                        **kwargs_eval_htest,
                    )
                    for test_stat_name in test_stat_names:
                        p_values_h0["c2st"][num_observation][test_stat_name].append(
                            p_value_h0[test_stat_name]
                        )

        if "lc2st" in methods:
            # Generate data from npe over joint
            npe_samples_joint, _ = generate_npe_data_for_lc2st(
                npe=npe,
                base_dist_samples=base_dist_samples[:n_cal],
                joint_samples={"theta": theta_cal, "x": x_cal},
                nf_case=False,
            )
            # evaluate test
            # train classifier
            trained_clf = train_lc2st(
                P=npe_samples_joint,
                Q=theta_cal,
                x_P=x_cal,
                x_Q=x_cal,
                clf=clf_lc2st,
            )

            if compute_type_I_error:
                P_joint = torch.cat([npe_samples_joint, x_cal], dim=1)
                Q_joint = torch.cat([theta_cal, x_cal], dim=1)
                P_joint_perm, Q_joint_perm = permute_data(P_joint, Q_joint)
                P_perm = P_joint_perm[:, : theta_cal.shape[-1]]
                Q_perm = Q_joint_perm[:, : theta_cal.shape[-1]]
                x_P_perm = P_joint_perm[:, theta_cal.shape[-1] :]
                x_Q_perm = Q_joint_perm[:, theta_cal.shape[-1] :]

                # train clf under H0
                trained_clf_h0 = train_lc2st(
                    P=P_perm,
                    Q=Q_perm,
                    x_P=x_P_perm,
                    x_Q=x_Q_perm,
                    clf=clf_lc2st,
                )
            for num_observation, observation in tqdm(
                observation_dict.items(),
                desc=f"Testing for observation {num_observation}",
            ):
                _, p_value, _, _ = eval_htest(
                    t_stats_estimator=t_stats_lc2st,
                    # args for t_stats_estimator
                    P=npe_samples_joint,
                    Q=theta_cal,
                    x_P=x_cal,
                    x_Q=x_cal,
                    x_eval=observation,
                    P_eval=npe_samples_obs_eval[num_observation],
                    Q_eval=None,
                    single_class_eval=True,
                    trained_clf=[trained_clf],
                    **kwargs_eval_htest,
                )
                for test_stat_name in test_stat_names:
                    p_values[num_observation]["lc2st"][test_stat_name].append(
                        p_value[test_stat_name]
                    )

                if compute_type_I_error:
                    _, p_value_h0, _, _ = eval_htest(
                        t_stats_estimator=t_stats_lc2st,
                        # args for t_stats_estimator
                        P=P_perm,
                        Q=Q_perm,
                        x_P=x_P_perm,
                        x_Q=x_Q_perm,
                        x_eval=observation,
                        P_eval=npe_samples_obs_eval[num_observation],
                        Q_eval=None,
                        single_class_eval=True,
                        trained_clf=[trained_clf_h0],
                        **kwargs_eval_htest,
                    )
                    for test_stat_name in test_stat_names:
                        p_values_h0[num_observation]["lc2st"][test_stat_name].append(
                            p_value_h0[test_stat_name]
                        )

            if "lc2st_nf" or "lc2st_nf_perm" in methods:
                # Compute inverse transformation on joint
                _, inv_transform_samples_joint = generate_npe_data_for_lc2st(
                    npe=npe,
                    base_dist_samples=base_dist_samples[:n_cal],
                    joint_samples={"theta": theta_cal, "x": x_cal},
                    nf_case=True,
                )

                # evaluate test
                # train classifier
                trained_clf = train_lc2st(
                    P=base_dist_samples[:n_cal],
                    Q=inv_transform_samples_joint,
                    x_P=x_cal,
                    x_Q=x_cal,
                    clf=clf_lc2st,
                )
                if "lc2st_nf" in methods and compute_type_I_error:
                    # generate new base dist samples
                    base_dist_samples_2 = npe.posterior_estimator._distribution.sample(
                        n_cal
                    ).detach()
                    # null is independent of x, it is the same for each observation
                    _, p_value_h0, _, _ = eval_htest(
                        t_stats_estimator=t_stats_c2st,
                        # args for t_stats_estimator
                        P=base_dist_samples[:n_cal],
                        Q=base_dist_samples_2,
                        P_eval=base_dist_samples[n_cal:],
                        Q_eval=None,
                        single_class_eval=True,
                        t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,  # not None here
                        **kwargs_eval_htest,
                    )
                if "lc2st_nf_perm" in methods and compute_type_I_error:
                    # permute data
                    P_joint = torch.cat([base_dist_samples[:n_cal], x_cal], dim=1)
                    Q_joint = torch.cat([inv_transform_samples_joint, x_cal], dim=1)
                    P_joint_perm, Q_joint_perm = permute_data(P_joint, Q_joint)
                    P_perm = P_joint_perm[:, : theta_cal.shape[-1]]
                    Q_perm = Q_joint_perm[:, : theta_cal.shape[-1]]
                    x_P_perm = P_joint_perm[:, theta_cal.shape[-1] :]
                    x_Q_perm = Q_joint_perm[:, theta_cal.shape[-1] :]

                    # train clf under H0
                    trained_clf_h0 = train_lc2st(
                        P=P_perm,
                        Q=Q_perm,
                        x_P=x_P_perm,
                        x_Q=x_Q_perm,
                        clf=clf_lc2st,
                    )

                for num_observation, observation in tqdm(
                    observation_dict.items(),
                    desc=f"Testing for observation {num_observation}",
                ):
                    _, p_value, _, _ = eval_htest(
                        t_stats_estimator=t_stats_lc2st,
                        # args for t_stats_estimator
                        P=base_dist_samples[:n_cal],
                        Q=inv_transform_samples_joint,
                        x_P=x_cal,
                        x_Q=x_cal,
                        x_eval=observation,
                        P_eval=base_dist_samples[n_cal:],
                        Q_eval=None,
                        single_class_eval=True,
                        trained_clf=[trained_clf],
                        t_stats_null=t_stats_null_lc2st_nf,  # not None here
                        **kwargs_eval_htest,
                    )
                    for test_stat_name in test_stat_names:
                        p_values["lc2st_nf"][num_observation][test_stat_name].append(
                            p_value[test_stat_name]
                        )
                    if compute_type_I_error:
                        p_values_h0["lc2st_nf"][num_observation][test_stat_name].append(
                            p_value_h0[test_stat_name]
                        )

                    if "lc2st_nf_perm" in methods:
                        _, p_value_perm, _, _ = eval_htest(
                            t_stats_estimator=t_stats_lc2st,
                            # args for t_stats_estimator
                            P=base_dist_samples[:n_cal],
                            Q=inv_transform_samples_joint,
                            x_P=x_cal,
                            x_Q=x_cal,
                            x_eval=observation,
                            P_eval=base_dist_samples[n_cal:],
                            Q_eval=None,
                            single_class_eval=True,
                            trained_clf=[trained_clf],
                            t_stats_null=None,  # permutation method
                            **kwargs_eval_htest,
                        )
                        for test_stat_name in test_stat_names:
                            p_values["lc2st_nf_perm"][num_observation][
                                test_stat_name
                            ].append(p_value_perm[test_stat_name])
                        if compute_type_I_error:
                            _, p_value_h0_perm, _, _ = eval_htest(
                                t_stats_estimator=t_stats_lc2st,
                                # args for t_stats_estimator
                                P=P_perm,
                                Q=Q_perm,
                                x_P=x_P_perm,
                                x_Q=x_Q_perm,
                                x_eval=observation,
                                P_eval=base_dist_samples[n_cal:],
                                Q_eval=None,
                                single_class_eval=True,
                                trained_clf=[trained_clf_h0],
                                t_stats_null=None,  # permutation method
                                **kwargs_eval_htest,
                            )

                            for test_stat_name in test_stat_names:
                                p_values_h0["lc2st_nf_perm"][num_observation][
                                    test_stat_name
                                ].append(p_value_h0_perm[test_stat_name])

    # compute power and type I error
    emp_power = dict(zip(methods, [{} for _ in methods]))  # TPR
    type_I_error = dict(zip(methods, [{} for _ in methods]))  # FPR
    for m in methods:
        emp_power[m] = dict(zip(test_stat_names, [[] for _ in test_stat_names]))  # TPR
        type_I_error[m] = dict(
            zip(test_stat_names, [[] for _ in test_stat_names])
        )  # FPR

        for num_observation, observation in observation_dict.items():
            # append TPR/TPF at alpha for each test statistic
            for t in test_stat_names:
                if alpha != 0:
                    emp_power[m][t].append(
                        np.mean(np.array(p_values[m][num_observation][t]) <= alpha)
                    )
                else:
                    emp_power[m][t].append(0)
                if compute_type_I_error and alpha != 0:
                    type_I_error[m][t].append(
                        np.mean(np.array(p_values_h0[m][num_observation][t]) <= alpha)
                    )
                else:
                    type_I_error[m][t].append(0)

    return emp_power, type_I_error, p_values, p_values_h0
