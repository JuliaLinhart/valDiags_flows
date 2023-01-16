import numpy as np

from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.utils import shuffle

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


def eval_local_flow_c2st(clf, x_eval, dim, size=1000, grid=None):
    if grid is not None:
        z_values = grid
    else: 
        # sample from normal dist (class 0)
        z_values = mvn(mean=np.zeros(dim), cov=np.eye(dim)).rvs(size)
    
    if dim == 1 and z_values.ndim == 1:
        z_values = z_values.reshape(-1,1)

    assert (z_values.shape[0] == size) and (z_values.shape[-1] == dim)

    features_eval = np.concatenate([z_values, x_eval.repeat(size,1)], axis=1)
    proba = clf.predict_proba(features_eval)[:,0]

    return proba