import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn

from scipy.stats import norm

import time

from nde.flows import cdf_flow
from data.feature_transforms import identity
from diagnostics.pp_plots import multi_cde_pit_values

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)
# DEFAULT_CLF = HistGradientBoostingClassifier(monotonic_cst=[0, 0, 1], max_iter=60)

## BASELINE
def localPIT_regression_baseline(
    alphas, pit_values_train, x_train, x_eval, classifier=DEFAULT_CLF,
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

    return r_alpha_test, accuracies, clf


# AMORTIZED IN ALPHA
def localPIT_regression_grid(
    alphas, pit_values_train, x_train, x_eval, classifier=DEFAULT_CLF, train=True,
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

    return r_alpha_test, train_accuracy, clf


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

    return r_alpha_test, train_accuracy, clf


# # CDF function of a (conditional) flow evaluated in x: F_{Q|context}(x)
# cdf_flow = lambda x, context, flow: D.Normal(0, 1).cdf(
#     flow._transform(x, context=context)[0]
# )

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
    n_trials = 1000,
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
    print('NULL HYP: ', null_hyp)

    r_alpha_learned = {}
    true_pit_values = {}
    for i, x_eval in enumerate(x_evals):
        print(f'x_eval {i}: ', x_eval)
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
                    "x_eval": x_eval,
                }

                start = time.time()
                if "baseline" in str(method):
                    alphas = np.linspace(0, 0.99, alpha_points)
                    method_kwargs["alphas"] = alphas
                    r_alpha_test, _ = method(**method_kwargs)
                    
                else:
                    alphas = np.linspace(0, 0.999, alpha_points)
                    method_kwargs["alphas"] = alphas
                    r_alpha_test, _, _ = method(**method_kwargs)
                r_alpha_k.append(r_alpha_test)
                times.append(start-time.time())

        
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
    n_trials = 10,
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
        n_trials=n_trials
    )

    pvalues = {}

    for i in range(len(x_evals)):
        pvalues[str(x_evals[i].numpy())] = {}
        for method_name, r_alpha_pit, r_alpha_null in zip(method_names, r_alpha_pit_dict[i], r_alpha_null_dict[i]):
            alphas = np.array(list(r_alpha_pit.keys()))
            r_alpha_pit_values = pd.Series(r_alpha_pit)
            Ti_value = ((r_alpha_pit_values - alphas) ** 2).sum() / len(alphas)
            all_unif_Ti_values = {}
            for k in range(len(r_alpha_null)):
                r_alpha_k = pd.Series(r_alpha_null[k])
                all_unif_Ti_values[k] = ((r_alpha_k - alphas) ** 2).sum() / len(alphas)

            pvalues[str(x_evals[i].numpy())][method_name] = sum(1 * (Ti_value < pd.Series(all_unif_Ti_values))) / len(all_unif_Ti_values)
    
    return r_alpha_pit_dict, r_alpha_null_dict, pvalues



##### scripts for multivariate case #####

def learn_multi_local_pit(theta, x, x_obs, flow, feature_transform=identity, null_hyp = False, n_trials = 1, alphas = np.linspace(0,0.99,100), clf=DEFAULT_CLF, reg_method=localPIT_regression_baseline):

    pit_values_train = multi_cde_pit_values(
        theta, x, flow, feature_transform=feature_transform
    )
    r_alpha_k = []
    for k in range(n_trials):

        if null_hyp:
            pit_values_train = [np.random.uniform(size=pit_values_train[0].shape)]*len(pit_values_train)

        r_alpha_learned = {}
        for i in range(len(pit_values_train)):
            r_alpha_learned[f"dim_{i+1}"], _, _ = reg_method(
                alphas=alphas,
                pit_values_train=pit_values_train[
                    i
                ].ravel(),  # pit-values used to compute the targets
                x_train=x[:, :, 0],
                x_eval=x_obs[:, :, 0].numpy(),  # evaluation sample x_0
                classifier=clf,
            )
        r_alpha_k.append(r_alpha_learned)
    if n_trials > 1:
        return r_alpha_k
    else:
        return r_alpha_learned

def compute_multi_pvalues(theta, x, x_obs, flow, n_trials, n_alphas=11, alpha_max=0.99, reg_method=localPIT_regression_baseline, clf=DEFAULT_CLF):
    r_alpha_learned = learn_multi_local_pit(theta, x, x_obs, flow, alphas=np.linspace(0,alpha_max,n_alphas), reg_method=reg_method, clf=clf)

    r_alpha_null_list = learn_multi_local_pit(theta, x, x_obs, flow, null_hyp=True, n_trials=n_trials, alphas=np.linspace(0,alpha_max,n_alphas),reg_method=reg_method, clf=clf)
    
    pvalues = {}
    for i, r_alpha_i in enumerate(r_alpha_learned.values()):
        alphas = np.array(list(r_alpha_i.keys()))
        r_alpha_pit_values = pd.Series(r_alpha_i)
        Ti_value = ((r_alpha_pit_values - alphas) ** 2).sum() / len(alphas)
        all_unif_Ti_values = {}
        for k, r_alpha_null_k in enumerate(r_alpha_null_list):
            r_alpha_k_i = pd.Series(r_alpha_null_k[f"dim_{i+1}"])
            all_unif_Ti_values[k] = ((r_alpha_k_i - alphas) ** 2).sum() / len(alphas)

        pvalues[f'dim {i+1}'] = sum(1 * (Ti_value < pd.Series(all_unif_Ti_values))) / len(all_unif_Ti_values)
    
    return pvalues, r_alpha_learned, r_alpha_null_list

def multi_LCT(pvalues, alpha=0.05):
    accepted = True
    for p in pvalues:
        if p<alpha:
            accepted = False
    return accepted


