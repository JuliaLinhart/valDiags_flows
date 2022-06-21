import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.distributions as D

from sklearn.neural_network import MLPClassifier
import sklearn

from scipy.stats import norm

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)

## BASELINE
def localPIT_regression_baseline(
    alphas, pit_values_train, x_train, x_eval, classifier=DEFAULT_CLF
):
    """Method 1: Algorithm from [Zhao et. al, UAI 2021]: https://arxiv.org/abs/2102.10473"""

    r_alpha_test = {}  # evaluated classifiers for each alpha
    accuracies = {}  # accuracies for each classifier

    # Estimate the local PIT-distribution quantiles
    for alpha in alphas:
        W_a_train = (pit_values_train <= alpha).astype(int)  # compute the targets
        clf = sklearn.base.clone(classifier)
        clf.fit(X=x_train, y=W_a_train)  # train classifier
        accuracies[alpha] = clf.score(x_train, W_a_train) * 100
        # evaluate in x_0
        prob = clf.predict_proba(x_eval)
        if prob.shape[1] < 2:  # Dummy
            r_alpha_test[alpha] = prob[:, 0][0]
        else:  # MLPClassifier
            r_alpha_test[alpha] = prob[:, 1][0]

    return r_alpha_test, accuracies


# AMORTIZED IN ALPHA
def localPIT_regression_grid(
    alphas, pit_values_train, x_train, x_eval, classifier=DEFAULT_CLF, train=True
):
    """Method 2: Train the Classifier amortized on x and all alpha"""

    # train features: all alpha and x
    T = len(alphas)
    train_features = []
    W_a_train = []
    for x, z in zip(x_train.numpy(), pit_values_train):
        x_rep = x[None].repeat(T, axis=0)
        alphas_train = alphas.reshape(-1, 1)
        train_features += [np.concatenate([x_rep, alphas_train], axis=1)]
        # train labels W_alpha(z)
        W_a_train += [1 * (z <= alpha) for alpha in alphas_train]

    train_features = np.row_stack(train_features)
    W_a_train = np.row_stack(W_a_train)

    if train:
        # define classifier
        clf = sklearn.base.clone(classifier)
        # train classifier
        # from sklearn.ensemble import HistGradientBoostingClassifier
        # clf = HistGradientBoostingClassifier(monotonic_cst=[0, 0, 1], max_iter=10)
        clf.fit(X=train_features, y=W_a_train.ravel())  # train classifier
    else:
        clf = classifier
    train_accuracy = clf.score(train_features, W_a_train.ravel()) * 100

    # Evaluate in x_0 and for all alphas in [0,1]
    r_alpha_test = {}
    for alpha in alphas:
        test_features = np.concatenate([x_eval, np.array(alpha).reshape(-1, 1)], axis=1)
        r_alpha_test[alpha] = clf.predict_proba(test_features)[:, 1][0]

    return r_alpha_test, train_accuracy, clf.loss_curve_, clf


def localPIT_regression_sample(
    alphas,
    pit_values_train,
    x_train,
    x_eval,
    nb_samples=1,
    classifier=DEFAULT_CLF,
    train=True,
):
    """METHOD 3: Train the Classifier amortized on x and sampled alpha"""
    train_features = []
    W_a_train = []
    for x, z in zip(x_train.numpy(), pit_values_train):
        x_rep = x[None].repeat(nb_samples, axis=0)
        alphas_sample = np.random.rand(nb_samples).reshape(-1, 1)
        train_features += [np.concatenate([x_rep, alphas_sample], axis=1)]
        # train labels W_alpha(z)
        W_a_train += [1 * (z <= alpha) for alpha in alphas_sample]

    train_features = np.row_stack(train_features)
    W_a_train = np.row_stack(W_a_train)

    if train:
        # define classifier
        clf = sklearn.base.clone(classifier)
        # train classifier
        clf.fit(X=train_features, y=W_a_train.ravel())  # train classifier
    else:
        clf = classifier
    train_accuracy = clf.score(train_features, W_a_train.ravel()) * 100

    r_alpha_test = {}
    for alpha in alphas:
        test_features = np.concatenate([x_eval, np.array(alpha).reshape(-1, 1)], axis=1)
        r_alpha_test[alpha] = clf.predict_proba(test_features)[:, 1][0]

    return r_alpha_test, train_accuracy, clf.loss_curve_, clf


# CDF function of a (conditional) flow evaluated in x: F_{Q|context}(x)
cdf_flow = lambda x, context, flow: D.Normal(0, 1).cdf(
    flow._transform(x, context=context)[0]
)


def run_localPIT_regression(
    methods,
    x_evals,
    nb_train_samples,
    alpha_samples,
    joint_data_generator,
    flow,
    feature_transform,
    samples_train=None,
):
    # Training set for regression task
    if samples_train is not None:
        x_train_PIT, theta_train_PIT = samples_train
    else:
        # Generate samples from the joint !
        x_train_PIT, theta_train_PIT = joint_data_generator(n=nb_train_samples)
        x_train_PIT, theta_train_PIT = (
            torch.FloatTensor(x_train_PIT),
            torch.FloatTensor(theta_train_PIT),
        )
    samples_train_new = (x_train_PIT, theta_train_PIT)

    # Compute the PIT-values [PIT(Theta_i, X_i, flow)]
    pit_values_train = np.array(
        [
            cdf_flow(theta_train_PIT[i][None], context=x, flow=flow).detach().numpy()
            for i, x in enumerate(feature_transform(x_train_PIT))
        ]
    ).ravel()

    r_alpha_learned = {}
    true_pit_values = {}
    for i, x_eval in enumerate(x_evals):
        method_kwargs = {
            "pit_values_train": pit_values_train,
            "x_train": x_train_PIT,
            "x_eval": x_eval,
        }

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
        for method in methods:
            if "baseline" in str(method):
                alphas = np.linspace(0, 0.99, 100)
                method_kwargs["alphas"] = alphas
                r_alpha_test, _ = method(**method_kwargs)
                r_alpha_x_eval.append(r_alpha_test)
                labels.append("baseline")
            else:
                alphas = np.linspace(0, 0.999, 100)
                method_kwargs["alphas"] = alphas
                if "sample" in str(method):
                    for ns in alpha_samples:
                        method_kwargs["nb_samples"] = ns
                        r_alpha_test, _, _, _ = method(**method_kwargs)
                        r_alpha_x_eval.append(r_alpha_test)
                        labels.append(f"sample (T={ns})")
                else:
                    r_alpha_test, _, _, _ = method(**method_kwargs)
                    r_alpha_x_eval.append(r_alpha_test)
                    labels.append(f"grid (T=100)")
        r_alpha_learned[i] = r_alpha_x_eval

    return r_alpha_learned, labels, true_pit_values, samples_train_new
