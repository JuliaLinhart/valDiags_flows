import numpy as np
import pandas as pd
import torch

from .localPIT import (
    localPIT_regression_baseline,
    infer_r_alphas_baseline,
    infer_r_alphas_amortized,
)
from .graphical_diagnostics import multi_cde_pit_values

from tasks.toy_examples.embeddings import identity

from sklearn.neural_network import MLPClassifier

from scipy.stats import hmean
from statsmodels.stats.multitest import multipletests

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


def compute_test_statistic(r_alpha_learned):
    """ Compute local test statistics T(x).

    inputs:
    - r_alpha_learned: dict, keys: alpha-values
        Regressor(s) evaluated in x and over a grid of alpha-values
        Output from "infer_r_alphas...".

    output:
    - T_value: float
        Test statistic T(x) = mean_{alphas}[(r_alpha_learned - alpha)^2]
    """
    alphas = np.array(list(r_alpha_learned.keys()))
    r_alpha_values = pd.Series(r_alpha_learned)
    T_value = ((r_alpha_values - alphas) ** 2).sum() / len(alphas)
    return T_value


def compute_pvalue(r_alpha_learned, r_alpha_null_list):
    """ Compute local p-values at x.

    inputs:
    - r_alpha_learned: dict, keys: alpha-values
        Regressor(s) evaluated in x and over a grid of alpha-values
        Output from "infer_r_alphas...".
    - r_alpha_null_list: list of dicts of length n_trials
        Regressor(s) trained on uniform pit-values over n_trials trials.
        The dicts are the regressors evaluated in x and over a grid of alpha-values (keys):
        output from "infer_r_alphas...".

    output:
    - pvalue: float
        P-value p(x) = mean_{k=1:n_trials}[T_est(x) < T_null_k(x)]
        where T_est(x) (resp. T_null_k(x)) is the output from "compute_test_statistic"
        for the regressed local pit values of the estimator
        (resp. the regressed uniform pit values of the k^th trial, i.e. under the null hypothesis),
        evaluated at x.
    """
    T_est = compute_test_statistic(r_alpha_learned)
    all_unif_T_values = {}
    for k, r_alpha_null_k in enumerate(r_alpha_null_list):
        all_unif_T_values[k] = compute_test_statistic(r_alpha_null_k)

    pvalue = sum(1 * (T_est < pd.Series(all_unif_T_values))) / len(all_unif_T_values)

    return pvalue


def multi_local_pit_regression(
    dim,
    pit_values_train,
    x_train,
    reg_method,
    classifier,
    alphas_eval=None,
    x_eval=None,
    trained_clfs=None,
):
    """ Train regressors and/or infer estimated local pit values
        for multivariate target data.

    inputs:
    - dim: int
        dimension of the multivariate target data
    - pit_values_train: list of length dim with numpy arrays of size (N, )
        List of pit-values for each dimension.
        The pit-values are computed on N samples (\Theta, X) from the joint.
    - x_train: torch.Tensor, size: (N, nb_features, 1)
    - reg_method: function
        One of "localPIT_regression_baseline", "localPIT_regression_grid",
        "localPIT_regression_sample" defined in localPIT_regression.py.
    - classifier: object
        Regression model trained to estimate the point-wise c.d.f.
    - alphas_eval: numpy array, size (K,)
        Grid of alpha values in (0,1) to evaluate the regressor in.
        Default is None: we just train, not evaluate.
    - x_eval: torch.Tensor, size: (1, nb_features, 1)
        Observation to evaluate the trained regressors in.
        Default is None: we just train, not evaluate.
    - trained_clfs: object or list of objects
        Trained regression model(s), that need to be evaluated in x_eval.
        Default is None: the regressor(s) need to be trained.

    outputs:
    - r_alpha_learned: dict of dicts.
        keys level 1: dimension, keys level 2: alpha-values
        One output from "infer_r-alpha..." for every dimension.
    - trained_clfs: object or list of objects
        Trained regression model(s), as output from "localPIT_regression...".
    """
    r_alpha_learned = {}
    trained_clfs_new = {}
    for i in range(dim):
        if trained_clfs is None:
            # Estimated Local PIT-values
            trained_clfs_new[f"dim_{i+1}"] = reg_method(
                pit_values_train=pit_values_train[
                    i
                ].ravel(),  # pit-values used to compute the targets
                x_train=x_train[:, :, 0],
                classifier=classifier,
            )
        else:
            if len(trained_clfs) > 1:
                trained_clfs_new = trained_clfs
            else:
                trained_clfs_new[f"dim_{i+1}"] = trained_clfs["dim_1"]

        if x_eval is not None and alphas_eval is not None:
            if "baseline" in str(reg_method):
                r_alpha_learned[f"dim_{i+1}"] = infer_r_alphas_baseline(
                    x_eval[:, :, 0].numpy(), trained_clfs_new[f"dim_{i+1}"]
                )
            else:
                r_alpha_learned[f"dim_{i+1}"] = infer_r_alphas_amortized(
                    x_eval[:, :, 0].numpy(), alphas_eval, trained_clfs_new[f"dim_{i+1}"]
                )

    return r_alpha_learned, trained_clfs_new


def multivariate_lct(
    theta_train,
    x_train,
    x_eval,
    flow,
    feature_transform=identity,
    n_trials=1000,
    n_alphas=21,
    alpha_max=0.99,
    reg_method=localPIT_regression_baseline,
    classifier=DEFAULT_CLF,
    trained_clfs=None,
    trained_clfs_null=None,
    return_pvalues=False,
):
    """ Compute LCT quantities for multivariate target data.

    inputs:
    - theta_train: torch.Tensor, size: (N, dim)
        dimension of the multivariate target data
    - x_train: torch.Tensor, size: (N, nb_features, 1)
    - x_eval: torch.Tensor, size: (1, nb_features, 1)
        Observation to evaluate the trained regressors in.
    - flow: class based on pyknos.nflows.distributions.base.Distribution
        Pytorch neural network defining our Normalizing Flow,
        hence conditional (posterior) density estimator.
    - feature_transform: function
        Default is "identity": no feature transform on x.
    - n_trials: int
        Number of trials for the null-hypothesis regression.
        One trial equals one regression problem under the null-hypothesis.
    - n_alphas: int
        Size of the grid of alpha-values between (0,1)
        Default is K=21.
    - alpha_max: float
        Maximum alpha-value.
        Default is 0.99 (for the baseline regression method, otherwise it should be 1).
    - reg_method: function
        One of the functions defined in localPIT_regression.py.
        Default is "localPIT_regression_baseline".
    - classifier: object
        Regression model trained to estimate the point-wise c.d.f.
        Default is sklearn.MLPClassifier(alpha=0, max_iter=25000).
    - trained_clfs: object or list of objects
        Trained regression model(s) on the estimated pit-values,
        that need to be evaluated in x_eval.
        Default is None: the regressor(s) need to be trained.
    - trained_clfs_null: object or list of objects
        Trained regression model(s) on uniform pit-values (i.e. under the null hypothesis),
        that need to be evaluated in x_eval.
        Default is None: the regressor(s) need to be trained.
    - return_pvalues: bool
        Wheather to compute the pvalues or not.
        Default is False: no need to train the regressors under the null.

    outputs:
    - lct_dict: dict of dicts.
        keys level 1: different test quantities
        (r_alpha_learned, r_alpha_null, test stats, pvalues, etc)
        keys level 2: dimensions of the target data.
    """
    lct_dict = {}

    alphas = np.linspace(0, alpha_max, n_alphas)

    # Estimated Local PIT-values
    pit_values_train_flow = multi_cde_pit_values(
        theta_train, x_train, flow, feature_transform=feature_transform
    )
    r_alpha_learned, _ = multi_local_pit_regression(
        dim=theta_train.shape[-1],
        pit_values_train=pit_values_train_flow,
        x_train=x_train,
        x_eval=x_eval,
        alphas_eval=alphas,
        reg_method=reg_method,
        classifier=classifier,
        trained_clfs=trained_clfs,
    )

    lct_dict["r_alpha_learned"] = r_alpha_learned

    # test statistic
    T_values = {}
    for i, r_alpha_i in enumerate(r_alpha_learned.values()):
        Ti_value = compute_test_statistic(r_alpha_i)
        T_values[f"dim_{i+1}"] = Ti_value

    lct_dict["test_stats"] = T_values

    if return_pvalues:
        # Local PIT-values under the null hypothesis
        r_alpha_null_list = []
        for k in range(n_trials):
            pit_values_train_null = [
                np.random.uniform(size=pit_values_train_flow[0].shape)
            ] * theta_train.shape[-1]

            if trained_clfs_null is not None:
                trained_clfs_null_k = trained_clfs_null[k]
            else:
                trained_clfs_null_k = None

            r_alpha_null_k, _ = multi_local_pit_regression(
                dim=theta_train.shape[-1],
                pit_values_train=pit_values_train_null,
                x_train=x_train,
                x_eval=x_eval,
                alphas_eval=alphas,
                reg_method=reg_method,
                classifier=classifier,
                trained_clfs=trained_clfs_null_k,
            )
            r_alpha_null_list.append(r_alpha_null_k)

        lct_dict["r_alpha_null_list"] = r_alpha_null_list

        # p-values
        pvalues = {}
        for i, r_alpha_i in enumerate(r_alpha_learned.values()):
            r_alpha_null_list_i = [
                r_alpha_null_list[k][f"dim_{i+1}"] for k in range(n_trials)
            ]
            pvalues[f"dim_{i+1}"] = compute_pvalue(r_alpha_i, r_alpha_null_list_i)

        lct_dict["pvalues"] = pvalues

    return lct_dict


def get_lct_results(lct_paths, alpha_level=0.05, n_dims=4, pvalues=True):
    """ Generate DataFrame with LCT results.

    inputs:
    - lct_paths: list of strings
        Paths to files with the output from "multivariate_lct".
        One element of the list corresponds to the lct results
        of one observations x_eval.
    - alpha_level: float
        Defines the (1-alpha_level) confidence level for the test.
        Default is 0.05.
    - n_dims: int
        Dimension of the multivariate target data.
        Default is 4.
    - pvalues: bool
        Whether to output the pvalues or not.
        Default is True.

    outputs:
    - df: pandas DataFrame
        Dataframe with multivaraite LCT results.
    """
    test_stats = {}
    if not pvalues:
        for i in range(1, n_dims + 1):
            test_stats[f"dim_{i}"] = []

        for lct_path in lct_paths:
            lct = torch.load(lct_path)
            for i in range(1, n_dims + 1):
                test_stats[f"dim_{i}"].append(lct["test_stats"][f"dim_{i}"])
        df = pd.DataFrame(test_stats)

    else:
        pvalues = {}
        test_results = {}
        for i in range(1, n_dims + 1):
            pvalues[f"dim_{i}"] = []
            test_stats[f"dim_{i}"] = []
            test_results[f"dim_{i}"] = []

        pvalues["hmean"] = []
        test_results["hmean"] = []
        test_results["combined"] = []
        for lct_path in lct_paths:
            lct = torch.load(lct_path)
            pvalues_g = lct["pvalues"]
            pvalues["hmean"].append(hmean(list(pvalues_g.values())))
            test_result_hmean = pvalues["hmean"][-1] <= alpha_level  # rejected if True
            multi_test_result = multipletests(list(pvalues_g.values()), method="b")[0]
            test_result = multi_test_result.sum() > 0
            test_results["combined"].append(test_result)
            test_results["hmean"].append(test_result_hmean)

            for i in range(1, n_dims + 1):
                pvalues[f"dim_{i}"].append(pvalues_g[f"dim_{i}"])
                test_stats[f"dim_{i}"].append(lct["test_stats"][f"dim_{i}"])
                test_results[f"dim_{i}"].append(multi_test_result[i - 1])
        df_test_stats = pd.DataFrame(test_stats)
        df_pvalues = pd.DataFrame(pvalues)
        df_results = pd.DataFrame(test_results)
        df = (
            df_test_stats.add_prefix("test_stats__")
            .join(df_pvalues.add_prefix("p_values__"))
            .join(df_results.add_prefix("lct_results__"))
        )
    return df

