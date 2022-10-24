import numpy as np

from sklearn.neural_network import MLPClassifier
import sklearn


DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)

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
        W_a_train += [1 * (pit <= alpha) for alpha in alphas_train] # size: (1,)

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
        x_rep = x[None].repeat(nb_samples, axis=0) # size: (K, nb_features)
        alphas_sample = np.random.rand(nb_samples).reshape(-1, 1) # size: (K, 1)
        train_features += [np.concatenate([x_rep, alphas_sample], axis=1)] # size: (K, nb_features + 1)
        # regression targets W_alpha(pit)
        W_a_train += [1 * (pit <= alpha) for alpha in alphas_sample] # size: (1,)

    train_features = np.row_stack(train_features) # size: (K x N, nb_features + 1)
    W_a_train = np.row_stack(W_a_train) # size: (K x N, 1)

    # define classifier
    clf = sklearn.base.clone(classifier)
    # train classifier
    clf.fit(X=train_features, y=W_a_train.ravel())  # train classifier

    return clf


def infer_r_alphas_amortized(x_eval, alphas, clf):
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
        test_features = np.concatenate([x_eval, np.array(alpha).reshape(-1, 1)], axis=1) # size: (1, nb_features + 1)
        r_alphas[alpha] = clf.predict_proba(test_features)[:, 1][0]

    return r_alphas


def multiPIT_regression_baseline(
    alphas, pit_values_train, x_train=None, classifier=DEFAULT_CLF,
):
    """ Estimate the 1D local PIT-distribution:
    
    Algorithm from [Zhao et. al, UAI 2021]: https://arxiv.org/abs/2102.10473
    --> adapted to multivariate data (conditionals).
    FOR EVERY ALPHA, the point-wise c.d.f 
        r_{\alpha}(X) = P(PIT_i <= alpha | X, PIT_{1:i-1}) = E[1_{PIT_i <= alpha} | X, PIT_{1:i-1}]
    is learned as a function of X, by regressing 1_{PIT_i <= alpha} on PIT_{1:i-1}.

    inputs:
    - alphas: numpy array, size: (K,)
        Grid of alpha values. One alpha-value equals one regression problem.
    - pit_values_train: list of numpy arrays of size (N,) for each dim 
        pit values computed on N samples (\Theta, X) from the joint.
        Used to compute the regression targets W and the context data.
    - x_train: torch.Tensor, size: (N, nb_features)
        Additional regression features for local PIT (i.e. when considering a conditional 
        target distribution p(\Theta | X)): data X of each pair (\Theta, X) 
        from the same dataset as used to compute the pit values.
        Default is None (for non-conditional data-distributions p(\Theta))
    - classifier: object
        Regression model trained to estimate the point-wise c.d.f. 
        Default is sklearn.MLPClassifier(alpha=0, max_iter=25000).
    
    output:
    - clfs: dict
        Trained regression models for each alpha-value.
    """
    clfs = {}
    for i,pit_i in enumerate(pit_values_train):
        pit_i = pit_i.ravel()
        clfs[i] = {}
        if x_train is not None:
            # context for conditional regression
            context_features_train = x_train
            if i!=0:
                context_features_train = np.concatenate([x_train]+pit_values_train[:i], axis=1)
        elif x_train is None and i!=0:
            context_features_train = np.concatenate(pit_values_train[:i], axis=1)
        else:
            continue

        for alpha in alphas:
            # compute the binary regression targets
            W_a_train = (pit_i <= alpha).astype(int)  # size: (N,)
            # define classifier
            clf = sklearn.base.clone(classifier)
            # train regression model
            clf.fit(X=context_features_train, y=W_a_train)
            clfs[i][alpha] = clf
    return clfs

def infer_multiPIT_r_alphas_baseline(pit_marginals_eval, clfs, x_eval=None):
    """ Infer the point-wise conditional CDF of the PIT for a given dimension i:
    r_{\alpha} = P(PIT_i <= alpha | PIT_{1:i-1}) = E[1_{PIT_i <= alpha} | PIT_{1:i-1}]

    inputs:
    - pit_marginals_eval: numpy array of size (n_alphas,i-1), where n_alphas = len(clfs)
        Marginal point-wise distribution for every PIT_{1:i-1}:
        empirical over test test (mean(PIT_i <= alpha)) or output from localPIT_regression_baseline.
        Used as context data / regression features.
    - clfs: dict, keys: alpha-values
        Trained regression models for each alpha-value. 
        Ouput from the function "localPIT_regression_baseline". 
    - x_eval: numpy array, size: (1, nb_features)
        Observation to evaluate the trained regressors in.

    output:
    - r_alphas: dict, keys: alpha-values 
        Estimated c.d.f values at x_eval: regressors evaluated in x_eval and PIT_{1:i-1}(x_eval).
        There is one for every alpha value.
    """
    alphas = np.array(list(clfs.keys()))
    r_alphas = {}
    for k,alpha in enumerate(alphas):
        features_eval = pit_marginals_eval[k][None,:]
        if x_eval is not None:
            features_eval = np.concatenate([x_eval, pit_marginals_eval[k][None,:]], axis =1)
        # evaluate 
        prob = clfs[alpha].predict_proba(features_eval)
        if prob.shape[1] < 2:  # Dummy Classifier
            r_alphas[alpha] = prob[:, 0][0]
        else:  # MLPClassifier or other
            r_alphas[alpha] = prob[:, 1][0]
    return r_alphas