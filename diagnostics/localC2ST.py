import numpy as np

from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.utils import shuffle
import sklearn

from scipy.stats import wasserstein_distance
from diagnostics.pp_plots import PP_vals

import pandas as pd

DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)

def c2st_clf(ndim):
    """ same setup as in :
    https://github.com/mackelab/sbi/blob/3e3522f177d4f56f3a617b2f15a5b2e25360a90f/sbi/utils/metrics.py
    """
    return MLPClassifier(
        **{
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        }
    )

def local_flow_c2st(flow_samples_train, x_train, classifier="mlp"):

    N = len(x_train)
    dim = flow_samples_train.shape[-1]
    reference = mvn(mean=np.zeros(dim), cov=np.eye(dim))  # base distribution

    # flow_samples_train = flow._transform(theta_train, context=x_train)[0].detach().numpy()
    ref_samples_train = reference.rvs(N)
    if dim == 1:
        ref_samples_train = ref_samples_train[:, None]

    features_flow_train = np.concatenate([flow_samples_train, x_train], axis=1)
    features_ref_train = np.concatenate([ref_samples_train, x_train], axis=1)
    features_train = np.concatenate([features_ref_train, features_flow_train], axis=0)
    labels_train = np.concatenate([np.array([0] * N), np.array([1] * N)]).ravel()
    features_train, labels_train = shuffle(
        features_train, labels_train, random_state=13
    )

    if classifier == "mlp":
        clf = c2st_clf(features_train.shape[-1])
    else:
        clf = DEFAULT_CLF

    clf.fit(X=features_train, y=labels_train)

    return clf


def eval_local_flow_c2st(clf, x_eval, dim, size=1000, z_values=None):
    if z_values is not None:
        z_values = z_values
    else: 
        # sample from normal dist (class 0)
        z_values = mvn(mean=np.zeros(dim), cov=np.eye(dim)).rvs(size)
    
    if dim == 1 and z_values.ndim == 1:
        z_values = z_values.reshape(-1,1)

    assert (z_values.shape[0] == size) and (z_values.shape[-1] == dim)

    features_eval = np.concatenate([z_values, x_eval.repeat(size,1)], axis=1)
    proba = clf.predict_proba(features_eval)[:,0]

    return proba, z_values

def score_lc2st(P, Q, x_train, x_eval, n_folds=10, classifier='mlp'):

    return 

def score_lc2st_flow(flow, theta_train, x_train, x_eval, metrics=['mean'], n_folds=10, val_frac=0.1, clf=MLPClassifier, clf_kwargs={'alpha':0, 'max_iter':25000}):
    inv_flow_samples = flow(x_train).transform(theta_train).detach().numpy()
    base_dist_samples = flow(x_train).base.sample()
    joint_inv_flow = np.concatenate([inv_flow_samples, x_train], axis=1)
    joint_base_dist = np.concatenate([base_dist_samples, x_train], axis=1)

    features = np.concatenate([joint_base_dist, joint_inv_flow], axis=0)
    labels = np.concatenate([np.array([0] * len(x_train)), np.array([1] * len(x_train))]).ravel()

    features, labels = shuffle(features, labels, random_state=13)
    
    classifier = clf(**clf_kwargs)

    probas = []
    scores = {}
    for m in metrics:
        scores[m] = []
    for _ in range(n_folds):
        # train n^th classifier 
        clf_n = sklearn.base.clone(classifier)
        clf_n.fit(X=features, y=labels)

        # eval n^th classifier
        val_size = int(len(x_train)*val_frac)
        base_dist_samples_eval = flow(x_train[:val_size]).base.sample()
        proba, _ = eval_local_flow_c2st(clf_n, x_eval, dim=theta_train.shape[-1], size=val_size, z_values=base_dist_samples_eval)
        probas.append(proba)
        for m in metrics:
            if m == 'mean':
                scores[m].append(np.mean(proba))
            elif m == 'w_dist': # wasserstein distance to dirac
                scores[m].append(wasserstein_distance([0.5]*val_size, proba)) 
            elif m == 'TV': # total variation: distance between cdfs of dirac and probas
                alphas = np.linspace(0,1,100)
                pp_vals_dirac = pd.Series(PP_vals([0.5]*val_size, alphas))
                pp_vals = PP_vals(proba, alphas)
                scores[m].append(((pp_vals - pp_vals_dirac) ** 2).sum() / len(alphas))
            else:
                print(f'metric "{m}" not implemented')
    
    return scores, probas