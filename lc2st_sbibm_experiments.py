import os

import torch
import numpy as np

from tqdm import tqdm
import time

from sklearn.neural_network import MLPClassifier

from valdiags.test_utils import eval_htest, permute_data
from valdiags.vanillaC2ST import t_stats_c2st, c2st_scores
from valdiags.localC2ST import t_stats_lc2st, lc2st_scores

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
    n_trials_null,
    t_stats_null_c2st_nf,
    t_stats_null_lc2st_nf,
    kwargs_c2st,
    kwargs_lc2st,
    task_path,
    results_n_train_path="",
    methods=["c2st", "lc2st", "lc2st_nf"],
    test_stat_names=["accuracy", "mse", "div"],
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
    )

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
            n_trials_null=n_trials_null,
            t_stats_null_c2st_nf=t_stats_null_c2st_nf,
            t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,
            task_path=task_path,
            results_n_train_path=results_n_train_path,
            methods=methods,
            test_stat_names=test_stat_names,
            compute_under_null=False,
            save_results=True,
        )

        for method in methods:
            train_runtime[method].append(train_runtime_n[method])

        for method, results in results_dict.items():
            if method in methods:
                for k, v in avg_result_keys.items():
                    for t_stat_name in test_stat_names:
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
    n_trials_null,
    t_stats_null_lc2st_nf,
    t_stats_null_c2st_nf,
    task_path,
    methods=["c2st", "lc2st", "lc2st_nf", "lc2st_nf_perm"],
    test_stat_names=["accuracy", "mse", "div"],
    compute_emp_power=True,
    compute_type_I_error=False,
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
                [np.zeros(len(observation_dict)) for _ in test_stat_names],
            )
        )
        p_values_h0[method] = dict(
            zip(
                test_stat_names,
                [np.zeros(len(observation_dict)) for _ in test_stat_names],
            )
        )

    for n in range(n_runs):
        print("====> RUN: ", n + 1, "/", n_runs, " <====")
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
        )

        # Empirical Power = True Positive Rate (TPR)
        # count rejection of H0 under H1 (p_value <= alpha) for every run
        # and for every observation: [reject(obs1), reject(obs2), ...]
        if compute_emp_power:
            print("Computing empirical power...")
            H1_results_dict, _ = compute_test_results_npe_one_run(
                alpha=alpha,
                data_samples=data_samples,
                n_train=n_train,
                observation_dict=observation_dict,
                kwargs_c2st=kwargs_c2st,
                kwargs_lc2st=kwargs_lc2st,
                n_trials_null=n_trials_null,
                t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,
                t_stats_null_c2st_nf=t_stats_null_c2st_nf,
                test_stat_names=test_stat_names,
                methods=methods,
                compute_under_null=False,
                task_path=task_path,
                results_n_train_path="",
                save_results=False,
            )
            for m in methods:
                for t_stat_name in test_stat_names:
                    # increment list of average rejections of H0 under H1
                    emp_power[m][t_stat_name] += (
                        (np.array(H1_results_dict[m]["p_value"][t_stat_name]) <= alpha)
                        * 1
                        / n_runs
                    )
                    # increment list of average p_values for every observation
                    for num_obs in observation_dict.keys():
                        p_values[m][t_stat_name] += H1_results_dict[m]["p_value"][
                            t_stat_name
                        ]
            else:
                emp_power, p_values = None, None

        # Type I error = False Positive Rate (FPR)
        # count rejection of H0 under H0 (p_value <= alpha) for every run
        # and for every observation: [reject(obs1), reject(obs2), ...]
        if compute_type_I_error:
            print("Computing Type I error...")
            H0_results_dict, _ = compute_test_results_npe_one_run(
                alpha=alpha,
                data_samples=data_samples,
                n_train=n_train,
                observation_dict=observation_dict,
                kwargs_c2st=kwargs_c2st,
                kwargs_lc2st=kwargs_lc2st,
                n_trials_null=n_trials_null,
                t_stats_null_lc2st_nf=t_stats_null_lc2st_nf,
                t_stats_null_c2st_nf=t_stats_null_c2st_nf,
                test_stat_names=test_stat_names,
                methods=methods,
                compute_under_null=True,
                task_path=task_path,
                results_n_train_path="",
                save_results=False,
            )
            for m in methods:
                for t_stat_name in test_stat_names:
                    # increment list of average rejections of H0 under H0
                    type_I_error[m][t_stat_name] += (
                        (np.array(H0_results_dict[m]["p_value"][t_stat_name]) <= alpha)
                        * 1
                        / n_runs
                    )
                    # append p_value of this run for every observation
                    for num_obs in observation_dict.keys():
                        p_values_h0[m][t_stat_name] += H0_results_dict[m]["p_value"][
                            t_stat_name
                        ][num_obs - 1]
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
):
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
        # ==== C2ST calibration dataset ==== #
        print("     1. C2ST: at fixed observation x_0")
        try:
            if not load_data:
                raise FileNotFoundError
            npe_samples_obs["cal"][N_train] = torch.load(
                task_path / f"npe_{N_train}" / f"npe_samples_obs_n_cal_{n_cal}.pkl"
            )
            npe_samples_obs["eval"][N_train] = torch.load(
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
                    task_path / f"npe_{N_train}" / f"npe_samples_obs_n_cal_{n_cal}.pkl",
                )
                torch.save(
                    reference_inv_transform_samples_cal[N_train],
                    task_path
                    / f"npe_{N_train}"
                    / f"reference_inv_transform_samples_n_cal_{n_cal}.pkl",
                )
                torch.save(
                    npe_samples_obs["eval"][N_train],
                    task_path
                    / f"npe_{N_train}"
                    / f"npe_samples_obs_n_eval_{n_eval}.pkl",
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
            if not load_data:
                raise FileNotFoundError
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
            if save_data:
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
    n_trials_null,
    t_stats_null_c2st_nf,
    t_stats_null_lc2st_nf,
    task_path,
    results_n_train_path,
    test_stat_names=["accuracy", "mse", "div"],
    methods=["c2st", "lc2st", "c2st_nf", "lc2st_nf", "lc2st_nf_perm"],
    alpha=0.05,
    compute_under_null=False,
    base_dist_samples_null=None,
    save_results=True,
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
    print()

    result_path = task_path / f"npe_{n_train}" / results_n_train_path
    if compute_under_null:
        result_path = result_path / "null"
    if save_results and not os.path.exists(result_path):
        os.makedirs(result_path)

    train_runtime = dict(zip(methods, [0 for _ in methods]))
    results_dict = dict(zip(methods, [{} for _ in methods]))

    print("     1. C2ST: for every x_0 in x_test")
    try:
        if "c2st" in methods:
            results_dict["c2st"] = torch.load(
                result_path / f"c2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
            )
        if "c2st_nf" in methods:
            results_dict["c2st_nf"] = torch.load(
                result_path / f"c2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
            )
    except FileNotFoundError:
        result_keys = ["reject", "p_value", "t_stat", "t_stats_null", "run_time"]
        results_dict["c2st"] = dict(
            zip(
                result_keys,
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in result_keys
                ],
            )
        )
        results_dict["c2st_nf"] = dict(
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
                desc=f"C2ST: Computing T for every observation x_0",
            ):
                P, Q = (
                    npe_samples_obs["cal"][n_obs],
                    reference_posterior_samples["cal"][n_obs],
                )
                P_eval, Q_eval = (
                    npe_samples_obs["eval"][n_obs],
                    reference_posterior_samples["eval"][n_obs],
                )
                if compute_under_null:
                    P, Q = permute_data(P, Q)
                    P_eval, Q_eval = permute_data(P_eval, Q_eval)

                t0 = time.time()
                c2st_results_obs = eval_htest(
                    conf_alpha=alpha,
                    t_stats_estimator=t_stats_c2st,
                    metrics=test_stat_names,
                    # kwargs for t_stats_c2st
                    P=P,
                    Q=Q,
                    P_eval=P_eval,
                    Q_eval=Q_eval,
                    use_permutation=True,  # it takes to long to sample new data from the reference / npe
                    n_trials_null=n_trials_null,
                    # kwargs for c2st_scores
                    **kwargs_c2st,
                )
                runtime = time.time() - t0
                for i, result_name in enumerate(result_keys):
                    for t_stat_name in test_stat_names:
                        if result_name == "run_time":
                            results_dict["c2st"][result_name][t_stat_name].append(
                                runtime
                            )
                        else:
                            results_dict["c2st"][result_name][t_stat_name].append(
                                c2st_results_obs[i][t_stat_name]
                            )
            if save_results:
                torch.save(
                    results_dict["c2st"],
                    result_path / f"c2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

        # C2ST-NF:
        # class 0: Z ~ N(0,I) vs. Z ~ T^{-1}(p_ref(theta | x_0))
        if "c2st_nf" in methods:
            # loop over observations x_0
            for n_obs in tqdm(
                observation_dict.keys(),
                desc=f"C2ST-NF: Computing T for every observation x_0",
            ):
                P = base_dist_samples["cal"]
                Q = reference_inv_transform_samples["cal"][n_obs]
                P_eval = base_dist_samples["eval"]
                Q_eval = reference_inv_transform_samples["eval"][n_obs]

                if compute_under_null:
                    P, Q = permute_data(P, Q)
                    P_eval, Q_eval = permute_data(P_eval, Q_eval)

                t0 = time.time()
                c2st_nf_results_obs = eval_htest(
                    conf_alpha=alpha,
                    t_stats_estimator=t_stats_c2st,
                    metrics=test_stat_names,
                    t_stats_null=t_stats_null_c2st_nf,  # use precomputed test statistics
                    # kwargs for t_stats_estimator
                    P=P,
                    Q=Q,
                    P_eval=P_eval,
                    Q_eval=Q_eval,
                    # kwargs for c2st_scores
                    **kwargs_c2st,
                )
                runtime = time.time() - t0
                for i, result_name in enumerate(result_keys):
                    for t_stat_name in test_stat_names:
                        if result_name == "run_time":
                            results_dict["c2st_nf"][result_name][t_stat_name].append(
                                runtime
                            )
                        else:
                            results_dict["c2st_nf"][result_name][t_stat_name].append(
                                c2st_nf_results_obs[i][t_stat_name]
                            )
            if save_results:
                torch.save(
                    results_dict["c2st_nf"],
                    result_path / f"c2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

    print("     2. L-C2ST: amortized")
    try:
        if "lc2st" in methods:
            results_dict["lc2st"] = torch.load(
                result_path / f"lc2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
            )
            train_runtime["lc2st"] = torch.load(
                result_path / f"runtime_lc2st_n_cal_{n_cal}.pkl"
            )

        if "lc2st_nf" in methods:
            results_dict["lc2st_nf"] = torch.load(
                result_path / f"lc2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
            )
            train_runtime["lc2st_nf"] = torch.load(
                result_path / f"runtime_lc2st_nf_n_cal_{n_cal}.pkl"
            )

        if "lc2st_nf_perm" in methods:
            results_dict["lc2st_nf_perm"] = torch.load(
                result_path / f"lc2st_nf_perm_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl"
            )
            train_runtime["lc2st_nf_perm"] = torch.load(
                result_path / f"runtime_lc2st_nf_perm_n_cal_{n_cal}.pkl"
            )

    except FileNotFoundError:
        result_keys = ["reject", "p_value", "t_stat", "t_stats_null", "run_time"]
        results_dict["lc2st"] = dict(
            zip(
                result_keys,
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in result_keys
                ],
            )
        )
        results_dict["lc2st_nf"] = dict(
            zip(
                result_keys,
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in result_keys
                ],
            )
        )
        results_dict["lc2st_nf_perm"] = dict(
            zip(
                result_keys,
                [
                    dict(zip(test_stat_names, [[] for _ in test_stat_names]))
                    for _ in result_keys
                ],
            )
        )

        # L-C2ST:
        if "lc2st" in methods:
            # train classifier on the joint
            print("L-C2ST: TRAINING CLASSIFIER on the joint ...")

            P = npe_samples_x_cal
            Q = theta_cal
            x_P = x_cal
            x_Q = x_cal
            if compute_under_null:
                joint_P_x = torch.cat([P, x_P], dim=1)
                joint_Q_x = torch.cat([Q, x_Q], dim=1)
                joint_P_x, joint_Q_x = permute_data(joint_P_x, joint_Q_x)
                P, x_P = joint_P_x[:, : P.shape[-1]], joint_P_x[:, P.shape[-1] :]
                Q, x_Q = joint_Q_x[:, : Q.shape[-1]], joint_Q_x[:, Q.shape[-1] :]

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
            torch.save(runtime, result_path / f"runtime_lc2st_n_cal_{n_cal}.pkl")
            train_runtime["lc2st"] = runtime

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

            for num_observation, observation in tqdm(
                observation_dict.items(),
                desc=f"L-C2ST: Computing T for every observation x_0",
            ):
                t0 = time.time()
                lc2st_results_obs = eval_htest(
                    conf_alpha=alpha,
                    t_stats_estimator=t_stats_lc2st,
                    metrics=test_stat_names,
                    # kwargs for t_stats_estimator
                    x_eval=observation,
                    P_eval=npe_samples_obs["eval"][num_observation],
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
                            results_dict["lc2st"][result_name][t_stat_name].append(
                                runtime
                            )
                        else:
                            results_dict["lc2st"][result_name][t_stat_name].append(
                                lc2st_results_obs[i][t_stat_name]
                            )
            if save_results:
                torch.save(
                    results_dict["lc2st"],
                    result_path / f"lc2st_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )

        # L-C2ST-NF:
        if "lc2st_nf" or "lc2st_nf_perm" in methods:
            # train classifier on the joint
            print("L-C2ST-NF: TRAINING CLASSIFIER on the joint ...")

            P = base_dist_samples["cal"]
            Q = inv_transform_samples_theta_cal
            x_P = x_cal
            x_Q = x_cal
            if compute_under_null:
                for m in ["lc2st_nf", "lc2st_nf_perm"]:
                    if m in methods:
                        if "perm" in m:
                            joint_P_x = torch.cat([P, x_P], dim=1)
                            joint_Q_x = torch.cat([Q, x_Q], dim=1)
                            joint_P_x, joint_Q_x = permute_data(joint_P_x, joint_Q_x)
                            P, x_P = (
                                joint_P_x[:, : P.shape[-1]],
                                joint_P_x[:, P.shape[-1] :],
                            )
                            Q, x_Q = (
                                joint_Q_x[:, : Q.shape[-1]],
                                joint_Q_x[:, Q.shape[-1] :],
                            )
                        else:
                            Q = base_dist_samples_null
                        t0 = time.time()
                        _, _, trained_clfs_lc2st_nf = lc2st_scores(
                            P=P,
                            Q=Q,
                            x_P=x_P,
                            x_Q=x_Q,
                            x_eval=None,
                            eval=False,
                            **kwargs_lc2st,
                        )
                        runtime = time.time() - t0
                        torch.save(
                            runtime, result_path / f"runtime_{m}_n_cal_{n_cal}.pkl"
                        )
                        train_runtime[m] = runtime
            else:
                t0 = time.time()
                _, _, trained_clfs_lc2st_nf = lc2st_scores(
                    P=base_dist_samples["cal"],
                    Q=inv_transform_samples_theta_cal,
                    x_P=x_cal,
                    x_Q=x_cal,
                    x_eval=None,
                    eval=False,
                    **kwargs_lc2st,
                )
                runtime = time.time() - t0
                torch.save(runtime, result_path / f"runtime_lc2st_nf_n_cal_{n_cal}.pkl")
                torch.save(
                    runtime, result_path / f"runtime_lc2st_nf_perm_n_cal_{n_cal}.pkl"
                )
                train_runtime["lc2st_nf"] = runtime
                train_runtime["lc2st_nf_perm"] = runtime

            _, _, trained_clfs_null_perm = t_stats_lc2st(
                null_hypothesis=True,
                n_trials_null=n_trials_null,
                use_permutation=True,
                P=base_dist_samples["cal"],
                Q=inv_transform_samples_theta_cal,
                x_P=x_cal,
                x_Q=x_cal,
                x_eval=None,
                P_eval=None,
                return_clfs_null=True,
                # kwargs for lc2st_scores
                eval=False,
                **kwargs_lc2st,
            )

            # if not permuation method, there is no need to train lc2st_nf
            # on the joint under the null hypothesis
            # we use precomuted test-statistics...

            for num_observation, observation in tqdm(
                observation_dict.items(),
                desc=f"L-C2ST-NF: Computing T for every observation x_0",
            ):
                if "lc2st_nf" in methods:
                    t0 = time.time()
                    lc2st_nf_results_obs = eval_htest(
                        conf_alpha=alpha,
                        t_stats_estimator=t_stats_lc2st,
                        metrics=test_stat_names,
                        t_stats_null=t_stats_null_lc2st_nf,  # use precomputed test statistics under null
                        # kwargs for t_stats_estimator
                        x_eval=observation,
                        P_eval=base_dist_samples["eval"],
                        Q_eval=None,
                        return_probas=False,
                        # unnecessary args as we have pretrained clfs and precomputed test statistics
                        P=base_dist_samples["cal"],
                        Q=inv_transform_samples_theta_cal,
                        x_P=x_cal,
                        x_Q=x_cal,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lc2st_nf,
                        # kwargs for lc2st_scores
                        **kwargs_lc2st,
                    )
                    runtime = (time.time() - t0) / n_trials_null

                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                results_dict["lc2st_nf"][result_name][
                                    t_stat_name
                                ].append(runtime)
                            else:
                                results_dict["lc2st_nf"][result_name][
                                    t_stat_name
                                ].append(lc2st_nf_results_obs[i][t_stat_name])

                if "lc2st_nf_perm" in methods:
                    t0 = time.time()
                    lc2st_nf_perm_results_obs = eval_htest(
                        t_stats_estimator=t_stats_lc2st,
                        metrics=test_stat_names,
                        # args for t_stats_estimator
                        x_eval=observation,
                        P_eval=base_dist_samples["cal"],
                        Q_eval=None,
                        use_permutation=True,
                        n_trials_null=n_trials_null,
                        return_probas=False,
                        # unnessary args as we pretrained the clfs
                        P=base_dist_samples["cal"],
                        Q=inv_transform_samples_theta_cal,
                        x_P=x_cal,
                        x_Q=x_cal,
                        # use same clf for all observations (amortized)
                        trained_clfs=trained_clfs_lc2st,
                        trained_clfs_null=trained_clfs_null_perm,
                        # kwargs for lc2st_scores
                        **kwargs_lc2st,
                    )
                    runtime = (time.time() - t0) / n_trials_null
                    for i, result_name in enumerate(result_keys):
                        for t_stat_name in test_stat_names:
                            if result_name == "run_time":
                                results_dict["lc2st_nf_perm"][result_name][
                                    t_stat_name
                                ].append(runtime)
                            else:
                                results_dict["lc2st_nf_perm"][result_name][
                                    t_stat_name
                                ].append(lc2st_nf_perm_results_obs[i][t_stat_name])

            if save_results:
                torch.save(
                    results_dict["lc2st_nf"],
                    result_path / f"lc2st_nf_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                )
                if "lc2st_nf_perm" in methods:
                    torch.save(
                        results_dict["lc2st_nf_perm"],
                        result_path
                        / f"lc2st_nf_perm_results_n_eval_{n_eval}_n_cal_{n_cal}.pkl",
                    )

    return results_dict, train_runtime


def precompute_t_stats_null(
    metrics,
    list_P_null,
    list_P_eval_null,
    t_stats_null_path,
    methods=["c2st_nf", "lc2st_nf"],
    save_results=True,
    **kwargs_c2st,
):
    # pre-compute / load test statistics for the null hypothesis
    if save_results and not os.path.exists(t_stats_null_path):
        os.makedirs(t_stats_null_path)

    n_trials_null = len(list_P_null) // 2
    n_cal = list_P_null[0].shape[0]
    n_eval = list_P_eval_null[0].shape[0]

    t_stats_null_dict = dict(zip(methods, [{} for _ in methods]))

    for m in methods:
        try:
            t_stats_null = torch.load(
                t_stats_null_path
                / f"{m}_stats_null_nt_{n_trials_null}_n_cal_{n_cal}.pkl"
            )
            print(
                f"Loaded pre-computed test statistics for (NF)-H_0 (N_cal={n_cal}, n_trials={n_trials_null})"
            )
        except FileNotFoundError:
            print(
                f"Pre-compute test statistics for (NF)-H_0 (N_cal={n_cal}, n_trials={n_trials_null})"
            )
            if m == "lc2st_nf":
                single_class_eval = True
            else:
                single_class_eval = False

            t_stats_null = t_stats_c2st(
                null_hypothesis=True,
                metrics=metrics,
                list_P_null=list_P_null,
                list_P_eval_null=list_P_eval_null,
                use_permutation=False,
                n_trials_null=n_trials_null,
                # required kwargs for t_stats_c2st
                P=list_P_null[0],
                Q=list_P_null[1],
                # kwargs for c2st_scores
                single_class_eval=single_class_eval,
                **kwargs_c2st,
            )
            torch.save(
                t_stats_null,
                t_stats_null_path
                / f"{m}_stats_null_nt_{n_trials_null}_n_cal_{n_cal}.pkl",
            )
        t_stats_null_dict[m] = t_stats_null

    return t_stats_null_dict
