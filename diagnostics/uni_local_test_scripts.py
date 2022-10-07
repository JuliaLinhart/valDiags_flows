import numpy as np
import pandas as pd
import torch

from scipy.stats import norm
import time

from nde.flows import cdf_flow
from diagnostics.localPIT_regression import (
    infer_r_alphas_baseline,
    infer_r_alphas_amortized,
)

#### 1d scripts for method comparison purposes ###


def run_localPIT_regression(
    methods,
    method_names,
    x_evals,
    nb_train_samples,
    joint_data_generator,
    flow,
    feature_transform,
    samples_train=None,
    null_hyp=False,
    alpha_points=100,
    n_trials=1000,
):
    # Training set for regression task
    if samples_train is not None:
        x_train_PIT_new, theta_train_PIT_new = samples_train
    else:
        # Generate samples from the joint !
        x_train_PIT_new, theta_train_PIT_new = joint_data_generator(n=nb_train_samples)
        x_train_PIT_new, theta_train_PIT_new = (
            torch.FloatTensor(x_train_PIT_new),
            torch.FloatTensor(theta_train_PIT_new),
        )
    samples_train_new = (x_train_PIT_new, theta_train_PIT_new)

    # Compute the PIT-values [PIT(Theta_i, X_i, flow)]
    pit_values_train = np.array(
        [
            cdf_flow(theta_train_PIT_new[i][None], context=x, flow=flow)
            .detach()
            .numpy()
            for i, x in enumerate(feature_transform(x_train_PIT_new))
        ]
    ).ravel()

    if not null_hyp:
        n_trials = 1
    print("NULL HYP: ", null_hyp)

    r_alpha_learned = {}
    true_pit_values = {}
    for i, x_eval in enumerate(x_evals):
        print(f"x_eval {i}: ", x_eval)
        if n_trials > 1:
            r_alpha_learned[i] = {}

        # samples from the true distribution
        samples_theta_x = torch.FloatTensor(
            norm(loc=x_eval[:, 0] + x_eval[:, 1], scale=1).rvs(nb_train_samples)
        ).reshape(-1, 1)
        # true PIT-values
        true_pit_values[i] = (
            cdf_flow(samples_theta_x, context=feature_transform(x_eval), flow=flow)
            .detach()
            .numpy()
        )

        r_alpha_x_eval = []
        labels = []
        times = []
        j = 0
        for method, method_name in zip(methods, method_names):
            print(method_name)
            r_alpha_k = []
            for k in range(n_trials):
                # print(k)
                if null_hyp:
                    pit_values_train = np.random.uniform(size=nb_train_samples)

                method_kwargs = {
                    "pit_values_train": pit_values_train,
                    "x_train": x_train_PIT_new,
                }

                start = time.time()
                if "baseline" in str(method):
                    alphas = np.linspace(0, 0.99, alpha_points)
                    method_kwargs["alphas"] = alphas
                    clfs = method(**method_kwargs)
                    r_alpha_test = infer_r_alphas_baseline(x_eval, clfs)

                else:
                    alphas = np.linspace(0, 0.999, alpha_points)
                    clf= method(**method_kwargs)
                    r_alpha_test = infer_r_alphas_amortized(x_eval, alphas, clf)
                r_alpha_k.append(r_alpha_test)
                times.append(start - time.time())

            if n_trials > 1:
                r_alpha_x_eval.append(r_alpha_k)
            else:
                r_alpha_x_eval.append(r_alpha_test)

        r_alpha_learned[i] = r_alpha_x_eval

    return r_alpha_learned, method_names, true_pit_values, samples_train_new, times


def compute_pvalues(
    methods,
    method_names,
    x_evals,
    nb_train_samples,
    joint_data_generator,
    flow,
    feature_transform,
    samples_train,
    n_trials=10,
):
    r_alpha_pit_dict, _, _, _, _ = run_localPIT_regression(
        methods,
        method_names,
        x_evals,
        nb_train_samples,
        joint_data_generator,
        flow,
        feature_transform,
        samples_train,
        null_hyp=False,
        alpha_points=11,
    )

    r_alpha_null_dict, _, _, _, _ = run_localPIT_regression(
        methods,
        method_names,
        x_evals,
        nb_train_samples,
        joint_data_generator,
        flow,
        feature_transform,
        samples_train,
        null_hyp=True,
        alpha_points=11,
        n_trials=n_trials,
    )

    pvalues = {}

    for i in range(len(x_evals)):
        pvalues[str(x_evals[i].numpy())] = {}
        for method_name, r_alpha_pit, r_alpha_null in zip(
            method_names, r_alpha_pit_dict[i], r_alpha_null_dict[i]
        ):
            alphas = np.array(list(r_alpha_pit.keys()))
            r_alpha_pit_values = pd.Series(r_alpha_pit)
            Ti_value = ((r_alpha_pit_values - alphas) ** 2).sum() / len(alphas)
            all_unif_Ti_values = {}
            for k in range(len(r_alpha_null)):
                r_alpha_k = pd.Series(r_alpha_null[k])
                all_unif_Ti_values[k] = ((r_alpha_k - alphas) ** 2).sum() / len(alphas)

            pvalues[str(x_evals[i].numpy())][method_name] = sum(
                1 * (Ti_value < pd.Series(all_unif_Ti_values))
            ) / len(all_unif_Ti_values)

    return r_alpha_pit_dict, r_alpha_null_dict, pvalues