import numpy as np
import pandas as pd

from diagnostics.localPIT_regression import (
    localPIT_regression_baseline,
    infer_r_alphas_baseline,
    infer_r_alphas_amortized,
)
from diagnostics.pp_plots import multi_cde_pit_values

from data.feature_transforms import identity

from sklearn.neural_network import MLPClassifier

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


def compute_test_statistic(r_alpha_learned):
    alphas = np.array(list(r_alpha_learned.keys()))
    r_alpha_pit_values = pd.Series(r_alpha_learned)
    T_value = ((r_alpha_pit_values - alphas) ** 2).sum() / len(alphas)
    return T_value


def compute_pvalue(r_alpha_learned, r_alpha_null_list):
    T_value = compute_test_statistic(r_alpha_learned)
    all_unif_T_values = {}
    for k, r_alpha_null_k in enumerate(r_alpha_null_list):
        all_unif_T_values[k] = compute_test_statistic(r_alpha_null_k)

    pvalue = sum(1 * (T_value < pd.Series(all_unif_T_values))) / len(all_unif_T_values)

    return pvalue

def multi_local_pit_regression(dim, pit_values_train, x_train, reg_method, classifier, alphas_eval = None, x_eval = None, trained_clfs=None):
    local_pit_values = {}
    trained_clfs_new = {}
    for i in range(dim):
        if trained_clfs is None:
            # Estimated Local PIT-values
            trained_clfs_new[f'dim_{i+1}'] = reg_method(
                pit_values_train=pit_values_train[
                    i
                ].ravel(),  # pit-values used to compute the targets
                x_train=x_train[:, :, 0],
                classifier=classifier,
            )
        else:
            trained_clfs_new = trained_clfs
        if x_eval is not None and alphas_eval is not None:
            if 'baseline' in str(reg_method):
                local_pit_values[f"dim_{i+1}"] = infer_r_alphas_baseline(x_eval[:,:,0].numpy(), trained_clfs_new[f'dim_{i+1}'])
            else:
                local_pit_values[f"dim_{i+1}"] = infer_r_alphas_amortized(x_eval[:,:,0].numpy(), alphas_eval, trained_clfs_new[f'dim_{i+1}'])
        
    return local_pit_values, trained_clfs_new


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
    return_pvalues = False,
):
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

    lct_dict['r_alpha_learned'] = r_alpha_learned

    # test statistic
    T_values = {}
    for i, r_alpha_i in enumerate(r_alpha_learned.values()):
        Ti_value = compute_test_statistic(r_alpha_i)
        T_values[f'dim_{i+1}'] = Ti_value

    lct_dict['test_stats'] = T_values
    

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
        
        lct_dict['r_alpha_null_list'] = r_alpha_null_list

        # p-values
        pvalues = {}
        for i, r_alpha_i in enumerate(r_alpha_learned.values()):
            r_alpha_null_list_i = [
                r_alpha_null_list[k][f"dim_{i+1}"] for k in range(n_trials)
            ]
            pvalues[f"dim_{i+1}"] = compute_pvalue(r_alpha_i, r_alpha_null_list_i)

        lct_dict['pvalues'] = pvalues
    
    return lct_dict


