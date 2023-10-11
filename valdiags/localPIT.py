import numpy as np
import pandas as pd
import torch
import time

from scipy.stats import norm

import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor

from itertools import combinations

from nde.flows import cdf_flow

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)

DEFAULT_REG = MLPRegressor(alpha=0, max_iter=25000)

# BASELINE
def localPIT_regression_baseline(
    alphas, pit_values_train, x_train, classifier=DEFAULT_CLF,
):
    """ Estimate the 1D local PIT-distribution:

    Algorithm from [Zhao et. al, UAI 2021]: https://arxiv.org/abs/2102.10473:
    FOR EVERY ALPHA, the point-wise c.d.f
        r_{\alpha}(X) = P(PIT <= alpha | X) = E[1_{PIT <= alpha} | X]
    is learned as a function of X, by regressing 1_{PIT <= alpha} on X.

    inputs:
    - alphas: numpy array, size: (K,)
        Grid of alpha values. One alpha-value equals one regression problem.
    - pit_values_train: numpy array, size: (N,)
        pit values computed on N samples (\Theta, X) from the joint.
        Used to compute the regression targets W.
    - x_train: torch.Tensor, size: (N, nb_features)
        regression features: data X of each pair (\Theta, X) from the same dataset as
        used to compute the pit values.
    - classifier: object
        Regression model trained to estimate the point-wise c.d.f.
        Default is sklearn.MLPClassifier(alpha=0, max_iter=25000).

    output:
    - clfs: dict
        Trained regression models for each alpha-value.
    """
    clfs = {}
    for alpha in alphas:
        # compute the binary regression targets
        W_a_train = (pit_values_train <= alpha).astype(int)  # size: (N,)
        # define classifier
        clf = sklearn.base.clone(classifier)
        # train regression model
        clf.fit(X=x_train, y=W_a_train)
        clfs[alpha] = clf

    return clfs


def infer_r_alphas_baseline(x_eval, clfs):
    """ Infer the point-wise CDF for a given observation x_eval.

    inputs:
    - x_eval: numpy array, size: (1, nb_features)
        Observation to evaluate the trained regressors in
    - clfs: dict, keys: alpha-values
        Trained regression models for each alpha-value.
        Ouput from the function "localPIT_regression_baseline".

    output:
    - r_alphas: dict, keys: alpha-values
        Estimated c.d.f values at x_eval: regressors evaluated in x_eval.
        There is one for every alpha value.
    """
    alphas = np.array(list(clfs.keys()))
    r_alphas = {}
    for alpha in alphas:
        # evaluate in x_eval
        prob = clfs[alpha].predict_proba(x_eval)
        if prob.shape[1] < 2:  # Dummy Classifier
            r_alphas[alpha] = prob[:, 0][0]
        else:  # MLPClassifier or other
            r_alphas[alpha] = prob[:, 1][0]
    return r_alphas


# AMORTIZED IN ALPHA
def localPIT_regression_grid(
    pit_values_train, x_train, classifier=DEFAULT_CLF, alphas=np.linspace(0, 1, 100),
):
    """ Estimate the 1D local PIT-distribution:

    Extension - Amortized on alpha - GRID:
    Learn the point-wise c.d.f
        r_{\alpha}(X) = P(PIT <= alpha | X) = E[1_{PIT <= alpha} | X]
    as a function of X and alpha, by regressing W = 1_{PIT <= alpha} on X and alpha.
    The dataset is augmented: for every X, we compute W on a grid of alpha values in (0,1).

    inputs:
    - pit_values_train: numpy array, size: (N,)
        pit values computed on N samples (\Theta, X) from the joint.
        Used to compute the regression targets W.
    - x_train: torch.Tensor, size: (N, nb_features)
        regression features: data X of each pair (\Theta, X) from the same dataset as
        used to compute the pit values.
    - classifier: object
        Regression model trained to estimate the point-wise c.d.f.
        Default is sklearn.MLPClassifier(alpha=0, max_iter=25000).
    - alphas: numpy array, size: (K,)
        Grid of alpha values. Used to augment the dataset.
        Default is np.linspace(0,1,100).

    output:
    - clf: object
        Trained regression model.
    """
    K = len(alphas)
    train_features = []
    W_a_train = []
    for x, pit in zip(x_train.numpy(), pit_values_train):
        # regression features
        x_rep = x[None].repeat(K, axis=0)  # size: (K, nb_features)
        alphas_train = alphas.reshape(-1, 1)  # size: (K, 1)
        train_features += [
            np.concatenate([x_rep, alphas_train], axis=1)
        ]  # size: (K, nb_features + 1)
        # regression targets W_{\alpha}(pit)
        W_a_train += [1 * (pit <= alpha) for alpha in alphas_train]  # size: (1,)

    train_features = np.row_stack(train_features)  # size: (K x N, nb_features + 1)
    W_a_train = np.row_stack(W_a_train)  # size: (K x N, 1)

    # define classifier
    clf = sklearn.base.clone(classifier)
    # train classifier
    clf.fit(X=train_features, y=W_a_train.ravel())  # train classifier

    return clf


def localPIT_regression_sample(
    pit_values_train, x_train, nb_samples=1, classifier=DEFAULT_CLF,
):
    """Estimate the 1D local PIT-distribution:

    Extension - Amortized on alpha - SAMPLE:
    Learn the point-wise c.d.f
        r_{\alpha}(X) = P(PIT <= alpha | X) = E[1_{PIT <= alpha} | X]
    as a function of X and alpha, by regressing W = 1_{PIT <= alpha} on X and alpha.
    The dataset is augmented: for every X, we sample alpha uniformly over (0,1)
    and compute W.

    inputs:
    - pit_values_train: numpy array, size: (N,)
        pit values computed on N samples (\Theta, X) from the joint.
        Used to compute the regression targets W.
    - x_train: torch.Tensor, size: (N, nb_features)
        Regression features: data X of each pair (\Theta, X) from the same dataset as
        used to compute the pit values.
    - nb_samples: int K
        Number of alpha samples used to augment the dataset.
        Default is 1.
    - classifier: object
        Regression model trained to estimate the point-wise c.d.f.
        Default is sklearn.MLPClassifier(alpha=0, max_iter=25000).

    output:
    - clf: object
        Trained regression model.
    """
    train_features = []
    W_a_train = []
    for x, pit in zip(x_train.numpy(), pit_values_train):
        # regression features
        x_rep = x[None].repeat(nb_samples, axis=0)  # size: (K, nb_features)
        alphas_sample = np.random.rand(nb_samples).reshape(-1, 1)  # size: (K, 1)
        train_features += [
            np.concatenate([x_rep, alphas_sample], axis=1)
        ]  # size: (K, nb_features + 1)
        # regression targets W_alpha(pit)
        W_a_train += [1 * (pit <= alpha) for alpha in alphas_sample]  # size: (1,)

    train_features = np.row_stack(train_features)  # size: (K x N, nb_features + 1)
    W_a_train = np.row_stack(W_a_train)  # size: (K x N, 1)

    # define classifier
    clf = sklearn.base.clone(classifier)
    # train classifier
    clf.fit(X=train_features, y=W_a_train.ravel())  # train classifier

    return clf


def infer_r_alphas_amortized(x_eval, alphas, clfs):
    """ Infer the point-wise CDF for a given observation x_eval and
    one or more alpha values.

    inputs:
    - x_eval: numpy array, size: (1, nb_features)
        Observation to evaluate the trained regressors in.
    - alphas: numpy array, size: (K,)
        alpha-values we want to evaluate the regressor in.
    - clf:
        Trained regression model aortized in alpha.
        Ouput from the function "localPIT_regression_grid" or "localPIT_regression_sample".

    output:
    - r_alphas: dict
        Estimated c.d.f values at x_eval:
        same regressor evaluated in x_eval and for every given alpha value.
    """
    r_alphas = {}
    for alpha in alphas:
        test_features = np.concatenate(
            [x_eval, np.array(alpha).reshape(-1, 1)], axis=1
        )  # size: (1, nb_features + 1)
        r_alphas[alpha] = clfs.predict_proba(test_features)[:, 1][0]

    return r_alphas


def local_correlation_regression(
    df_flow_transform, x_train, x_eval=None, regressor=DEFAULT_REG, null=False
):
    Z_labels = list(df_flow_transform.keys())
    # compute train targets
    train_targets = []
    for comb in combinations(Z_labels, 2):
        train_targets.append(df_flow_transform[comb[0]] * df_flow_transform[comb[1]])
    labels = ["12", "13", "14", "23", "24", "34"]
    if null:
        train_targets = [df_flow_transform[0], df_flow_transform[1]]
        labels = ["12"]
    results = {}
    regs = {}
    for target, label in zip(train_targets, labels):
        reg = sklearn.base.clone(regressor)
        reg.fit(X=x_train, y=target)
        regs[label] = reg
        if x_eval is not None:
            results[label] = reg.predict(x_eval)
    return regs, results


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
                    clf = method(**method_kwargs)
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
