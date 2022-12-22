import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor
import sklearn

from itertools import combinations

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


def eval_local_flow_c2st(clf, x_eval, dim, n_rounds=1000):
    if dim == 1:
        reference = norm()
    else:
        reference = mvn(mean=np.zeros(dim), cov=np.eye(dim))  # base distribution

    proba = []
    for i in range(n_rounds):
        features_eval = np.concatenate([reference.rvs(1), x_eval])[None, :]
        proba.append(clf.predict_proba(features_eval)[0][0])

    return proba
