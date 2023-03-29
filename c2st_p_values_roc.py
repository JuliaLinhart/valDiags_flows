# This is a general script to perform evaluations on the Classifier Two Sample Test (C2ST).
# =======================
# We define a utility function that computes the p-values, TPR and FPR over several runs of the C2ST
# and depending on different parameters of the experiment (e.g. the data distributions P and Q,
# the number of samples, the test statistics, whether to use only one class during evalaution, etc.).
# =======================
# In the main function, we define the parameters of the experiment and call the function to compute the p-values, TPR and FPR.
# The implemented experiments that can be run with this script are:
#   - plot the p-values/TPR/FPR/ROC curves for each metric over a grid of alpha values (significance levels).
#   - evaluate the type 1 error rate (FPR) for a given significance level alpha as a function of the sample size.
# Other experiments can be added.

import numpy as np
from tqdm import tqdm


def c2st_p_values_tfpr(
    eval_c2st_fn,
    n_runs,
    alpha_list,
    P_dist,
    Q_dist,
    n_samples,
    metrics,
    metrics_cv=None,
    n_folds=2,
    compute_FPR=True,
    compute_TPR=True,
    scores_null=None,
    use_permutation=True,
):
    """Computes the p-values, TPR and FPR over several runs of the Classifier Two Sample Test (C2ST)
    between two distributions P and Q:

                                         (H0): P = Q   (H1): P != Q 
    for different metrics (test statistics).

    The p-value of a test-run is defined as the probability of falsely rejecting the null hypothesis (H0).
    For a given significance level alpha, we reject the null hypothesis (H0) if p-value < alpha.
    - TPR is the average number of times we correctly reject the null hypothesis (H0): power of the test.
    - FPR is the average number of times we incorrectly reject the null hypothesis (H0).
    We compute them for a range of significance levels alpha in (0,1), so that we can plot the ROC curve.
    
    Args:
        eval_c2st_fn (function): function that evaluates the C2ST test
        n_runs (int): number of test runs to compute FPR and TPR. Each time with new samples from P and Q.
        alpha_list (list): list of significance levels alpha in (0,1) to compute FPR and TPR at
        P_dist (scipy.stats.rv_continuous): distribution of P
        Q_dist (scipy.stats.rv_continuous): distribution of Q
        n_samples (int): number of samples from P and Q (same for train and evaluation).
        metrics (list): list of metrics to be used for the test (test statistics)
        metrics_cv (list): list of metrics to be used for the cross-validation. 
            Defauts to None.
        compute_FPR (bool): whether to compute FPR or not. 
            Defaults to True.
        compute_TPR (bool): whether to compute TPR or not.
            Defaults to True.
        scores_null (dict): dict of test statistics under the null.
            keys: True (cross-val) and False (no cross-val).
            values: second output of t_stats_c2st function.
            If None, use_permuation should be True.
            Defaults to None.
        use_permutation (bool): whether to use permutation to compute the test statistics under the null.
            Defaults to True.
    
    Returns:
        p_values_H1 (dict): dict of p-values for each metric under (H1)
        p_values_H0 (dict): dict of p-values for each metric under (H0)
        TPR (dict): dict of TPR for each metric at each alpha in alpha_list
        FPR (dict): dict of FPR for each metric at each alpha in alpha_list
    """
    all_metrics = metrics
    if metrics_cv is not None:
        # combine metrics and metrics_cv
        all_metrics = metrics + metrics_cv

    if scores_null is None:
        t_stats_null = None
        t_stats_null_cv = None
    else:
        t_stats_null = scores_null[False]
        t_stats_null_cv = scores_null[True]

    # initialize dict with empty lists
    p_values_H1 = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    p_values_H0 = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

    # loop over test runs
    for _ in tqdm(range(n_runs), desc="Test runs"):
        # generate samples from P and Q
        P = P_dist.rvs(n_samples)
        Q = Q_dist.rvs(n_samples)
        Q_H0 = P_dist.rvs(n_samples)

        P_eval = P_dist.rvs(n_samples)
        Q_eval = Q_dist.rvs(n_samples)
        Q_H0_eval = P_dist.rvs(n_samples)

        if compute_TPR:
            # evaluate test under (H1)
            _, p_value = eval_c2st_fn(
                metrics=metrics,
                # args for t_stats_c2st
                P=P,
                Q=Q,
                P_eval=P_eval,
                Q_eval=Q_eval,
                cross_val=False,
                t_stats_null=t_stats_null,
                use_permutation=use_permutation,
            )
            # update the empirical power at alpha for each metric
            for m in metrics:
                p_values_H1[m].append(p_value[m])

        if compute_FPR:
            # evaluate test under (H0)
            _, p_value = eval_c2st_fn(
                metrics=metrics,
                P=P,
                Q=Q_H0,
                P_eval=P_eval,
                Q_eval=Q_H0_eval,
                cross_val=False,
                t_stats_null=t_stats_null,
                use_permutation=use_permutation,
            )
            # update the FPR at alpha for each metric
            for m in metrics:
                p_values_H0[m].append(p_value[m])

        if metrics_cv is not None:
            if compute_TPR:
                # evaluate test under (H1) over several cross-val folds
                _, p_value_cv = eval_c2st_fn(
                    metrics=metrics_cv,
                    P=P,
                    Q=Q,
                    cross_val=True,
                    n_folds=n_folds,
                    t_stats_null=t_stats_null_cv,
                    use_permutation=use_permutation,
                )
                # update the empirical power at alpha for each cv-metric
                for m in metrics_cv:
                    p_values_H1[m].append(p_value_cv[m])

            if compute_FPR:
                # evaluate test under (H0) over several cross-val folds
                _, p_value_cv = eval_c2st_fn(
                    metrics=metrics_cv,
                    P=P,
                    Q=Q_H0,
                    cross_val=True,
                    n_folds=n_folds,
                    t_stats_null=t_stats_null_cv,
                    use_permutation=use_permutation,
                )
                # update the FPR at alpha for each cv-metric
                for m in metrics_cv:
                    p_values_H0[m].append(p_value_cv[m])

    # compute TPR and TPF at every alpha
    TPR = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    FPR = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))
    for alpha in alpha_list:
        # append TPR/TPF at alpha for each metric
        for m in all_metrics:
            if alpha == 0:
                TPR[m].append(0)
                FPR[m].append(0)
            else:
                TPR[m].append(np.mean(np.array(p_values_H1[m]) <= alpha))
                FPR[m].append(np.mean(np.array(p_values_H0[m]) <= alpha))

    return TPR, FPR, p_values_H1, p_values_H0


if __name__ == "__main__":
    import argparse
    import os

    from functools import partial
    import matplotlib.pyplot as plt

    from valdiags.test_utils import eval_htest
    from valdiags.vanillaC2ST import t_stats_c2st

    from scipy.stats import multivariate_normal as mvn

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier

    # experiment parameters that need to be defined and cannot be passed in the argparser

    PATH_EXPERIMENT = "saved_experiments/c2st_evaluation/"

    # distributions P and Q
    dim = 5  # data dimension
    P_dist = mvn(mean=np.zeros(dim), cov=np.eye(dim))
    mu = np.sqrt(0.05)  # mean shift between P and Q
    Q_dist = mvn(mean=np.array([mu] * dim), cov=np.eye(dim))

    # metrics / test statistics
    metrics = ["accuracy", "div", "mse"]
    metrics_cv = ["accuracy_cv", "div_cv", "mse_cv"]
    cross_val_folds = 2

    # parse arguments
    # default values according to [Lee et al. 2018](https://arxiv.org/abs/1812.08927)
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument(
        "--n_samples",  # make as list
        "-ns",
        type=int,
        default=100,
        help="Number of samples for P and Q to train and evaluate the classifier.",
    )

    # test parameters
    parser.add_argument(
        "--n_runs", "-nr", type=int, default=300, help="Number of test runs.",
    )
    parser.add_argument(
        "-alphas",
        "-a",
        nargs="+",
        type=float,
        default=np.linspace(0, 1, 20),
        help="List of significance levels to evaluate the test at.",
    )

    # null distribution parameters
    parser.add_argument(
        "--n_trials_null",
        "-nt",
        type=int,
        default=100,
        help="Number of trials to estimate the distribution of the test statistic under the null.",
    )
    parser.add_argument(
        "--use_permutation",
        "-p",
        action="store_true",
        help="Use permutations to estimate the null distribution. \
            If False, approximate the true null distribution with samples from P.",
    )

    # classifier parameters
    parser.add_argument(
        "--clf_name",  # make list
        "-c",
        type=str,
        default="LDA",
        choices=["LDA", "MLP"],
        help="Classifier to use.",
    )
    parser.add_argument(
        "--single_class_eval",
        "-1c",
        action="store_true",
        help="Evaluate the classifier on one class only.",
    )
    parser.add_argument(
        "--in_sample",
        "-in",
        action="store_true",
        help="In-sample evaluation of the classifier (on training data).",
    )

    # experiment parameters
    parser.add_argument(
        "--roc",
        action="store_true",
        help="Compute and Plot ROC curve for the test. In this case `alphas` should be a grid in (0,1).",
    )
    parser.add_argument(
        "--type1_err",
        "-t1",
        action="store_true",
        help="Compute and Plot Type 1 error for the test over multiple sample sizes.",
    )
    args = parser.parse_args()

    # Initialize classifier
    if args.clf_name == "LDA":
        clf_class = LinearDiscriminantAnalysis
        clf_kwargs = {"solver": "eigen", "priors": [0.5, 0.5]}
    elif args.clf_name == "MLP":
        clf_class = MLPClassifier
        clf_kwargs = {"alpha": 0, "max_iter": 25000}
    else:
        raise NotImplementedError

    # important parameters
    N_SAMPLES = args.n_samples
    N_RUNS = args.n_runs
    N_TRIALS_NULL = args.n_trials_null

    # test statistic function
    t_stats_c2st_custom = partial(
        t_stats_c2st,
        n_trials_null=N_TRIALS_NULL,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        in_sample=args.in_sample,
        single_class_eval=args.single_class_eval,
    )

    # test statistics under the null distribution
    if not args.use_permutation:
        # not using the permutation method to simulate the null distribution
        # use data from P to compute the scores/test statistics under the true null distribution
        print()
        print(
            "Pre-computing or loading the test statistics under the null distribution."
            + "\n They will be reused at every test-run. The permutation method is not needed."
        )
        print()
        scores_null = dict(zip([True, False], [None, None]))
        for cross_val, metric_list in zip([True, False], [metrics_cv, metrics]):
            filename = f"nt_{N_TRIALS_NULL}_dim_{dim}_{args.clf_name}_single_class_{args.single_class_eval}_in_sample_{args.in_sample}"
            if cross_val:
                filename += f"_cross_val_nfolds_{cross_val_folds}.npy"
            else:
                filename += ".npy"
            if os.path.exists(PATH_EXPERIMENT + "t_stats_null/" + filename):
                # load null scores if they exist
                t_stats_null = np.load(
                    PATH_EXPERIMENT + "t_stats_null/" + filename, allow_pickle=True,
                ).item()
            else:
                # otherwise, compute them
                # generate data from P
                list_P_null = [P_dist.rvs(N_SAMPLES) for _ in range(2 * N_TRIALS_NULL)]
                list_P_eval_null = [
                    P_dist.rvs(N_SAMPLES) for _ in range(2 * N_TRIALS_NULL)
                ]
                _, t_stats_null = t_stats_c2st_custom(
                    use_permutation=False,
                    metrics=metric_list,
                    cross_val=cross_val,
                    n_folds=cross_val_folds,
                    list_P_null=list_P_null,
                    list_P_eval_null=list_P_eval_null,
                    # unnecessary, but needed inside `t_stats_c2st`
                    P=list_P_null[0],
                    Q=list_P_eval_null[0],
                    P_eval=list_P_null[1],
                    Q_eval=list_P_eval_null[1],
                )
                # save null scores
                np.save(
                    PATH_EXPERIMENT + "t_stats_null/" + filename, t_stats_null,
                )
            scores_null[cross_val] = t_stats_null
    else:
        print()
        print(
            f"Not pre-computing the test-statistics under the null."
            + "\n Using the permutation method to estimate them at each test run."
        )
        print()
        scores_null = None

    # define function to evaluate the test
    eval_c2st = partial(
        eval_htest,
        t_stats_estimator=t_stats_c2st_custom,
        use_permutation=args.use_permutation,
        verbose=False,
    )

    if args.roc:
        # compute p_value at alpha=0.05, for each metric with `eval_c2st_lda`
        TPR, FPR, p_values_H1, p_values_H0 = c2st_p_values_tfpr(
            eval_c2st_fn=eval_c2st,
            n_runs=N_RUNS,
            n_samples=N_SAMPLES,
            alpha_list=args.alphas,
            P_dist=P_dist,
            Q_dist=Q_dist,
            metrics=metrics,
            metrics_cv=metrics_cv,
            n_folds=cross_val_folds,
            scores_null=scores_null,
            use_permutation=args.use_permutation,
        )

        # plot p-values for each metric

        for m in metrics + metrics_cv:
            p_values = np.concatenate(
                [p_values_H1[m], p_values_H0[m]]
            )  # concatenate H1 and H0 p-values
            index = np.concatenate(
                [np.ones(N_RUNS), np.zeros(N_RUNS)]
            )  # 1 for H1, 0 for H0
            sorter = np.argsort(p_values)  # sort p-values
            sorted_index = index[sorter]  # sort index
            idx_0 = np.where(sorted_index == 0)[0]  # find index of H0 p-values
            idx_1 = np.where(sorted_index == 1)[0]  # find index of H1 p-values

            plt.plot(np.sort(p_values), color="blue", label="p-values")

            plt.scatter(
                np.arange(2 * N_RUNS)[idx_1],
                np.sort(p_values)[idx_1],
                c="g",
                label=f"H1 (mu={np.round(mu,2)})",
                alpha=0.3,
            )
            plt.scatter(
                np.arange(2 * N_RUNS)[idx_0],
                np.sort(p_values)[idx_0],
                c="r",
                label="H0",
                alpha=0.3,
            )
            plt.legend()
            plt.title(f"C2ST-{m}, (N={N_SAMPLES}, dim={dim})")
            plt.savefig(
                PATH_EXPERIMENT
                + f"p_values_{m}_{args.clf_name}_mu_{np.round(mu,2)}_dim_{dim}_nruns_{N_RUNS}_single_class_{args.single_class_eval}_permutation_{args.use_permutation}_insample_{args.in_sample}.pdf"
            )
            plt.show()

        # plot TPR for each metric
        for m in metrics + metrics_cv:
            plt.plot(args.alphas, TPR[m], label=m)
        plt.legend()
        plt.title(
            f"TPR for C2ST, (H1): mu={np.round(mu,2)}, (N={N_SAMPLES}, dim={dim})"
        )
        plt.savefig(
            PATH_EXPERIMENT
            + f"tpr_{args.clf_name}_mu_{np.round(mu,2)}_dim_{dim}_nruns_{N_RUNS}_single_class_{args.single_class_eval}_permutation_{args.use_permutation}_insample_{args.in_sample}.pdf"
        )
        plt.show()

        # plot FPR for each metric
        for m in metrics + metrics_cv:
            plt.plot(args.alphas, FPR[m], label=m)
        plt.legend()
        plt.title(
            f"FPR for C2ST, (H1): mu={np.round(mu,2)}, (N={N_SAMPLES}, dim={dim})"
        )
        plt.savefig(
            PATH_EXPERIMENT
            + f"fpr_{args.clf_name}_mu_{np.round(mu,2)}_dim_{dim}_nruns_{N_RUNS}_single_class_{args.single_class_eval}_permutation_{args.use_permutation}_insample_{args.in_sample}.pdf"
        )
        plt.show()

        # roc curve
        for m in metrics + metrics_cv:
            plt.plot(FPR[m], TPR[m], label=m)
        plt.legend()
        plt.title(
            f"ROC for C2ST, (H1): mu={np.round(mu,2)}, (N={N_SAMPLES}, dim={dim})"
        )
        plt.savefig(
            PATH_EXPERIMENT
            + f"roc_{args.clf_name}_mu_{np.round(mu,2)}_dim_{dim}_nruns_{N_RUNS}_single_class_{args.single_class_eval}_permutation_{args.use_permutation}_insample_{args.in_sample}.pdf"
        )
        plt.show()

    if args.type1_err:

        # type I error as a function of n_samples at alpha
        # (as in [Lopez-Paz et al. 2016](https://arxiv.org/abs/1610.06545))
        n_samples_list = [25, 50, 100, 200, 500, 1000, 1500, 2000]
        FPR_n = dict(zip(metrics + metrics_cv, [[] for _ in metrics + metrics_cv]))
        for n in n_samples_list:
            print(f"n={n}:")
            _, FPR, _, p_values_H0 = c2st_p_values_tfpr(
                n_samples=n,
                eval_c2st_fn=eval_c2st,
                n_runs=N_RUNS,
                alpha_list=args.alphas,
                P_dist=P_dist,
                Q_dist=Q_dist,
                metrics=metrics,
                metrics_cv=metrics_cv,
                n_folds=cross_val_folds,
                scores_null=scores_null,
                use_permutation=args.use_permutation,
                compute_TPR=False,
            )

            for m in metrics + metrics_cv:
                FPR_n[m].append(FPR[m][0])
                print(f"{m}: {FPR[m]}")

        for m in metrics + metrics_cv:
            plt.plot(n_samples_list, FPR_n[m], label=m)
        plt.legend()
        plt.title(
            f"C2ST Type I error as a function of n_samples (dim={dim})"
            + f"\n alpha = {args.alphas} "
        )
        plt.savefig(
            PATH_EXPERIMENT
            + f"type_I_error_alpha_{args.alphas}_{args.clf_name}_n_dim_{dim}_nruns_{N_RUNS}_single_class_{args.single_class_eval}_permutation_{args.use_permutation}_insample_{args.in_sample}.pdf"
        )
        plt.show()

