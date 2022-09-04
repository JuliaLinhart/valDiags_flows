import numpy as np

from sklearn.neural_network import MLPClassifier
import sklearn


DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)

## BASELINE
def localPIT_regression_baseline(
    alphas, pit_values_train, x_train, classifier=DEFAULT_CLF,
):
    """Method 1: Algorithm from [Zhao et. al, UAI 2021]: https://arxiv.org/abs/2102.10473"""
    clfs = {}
    # Estimate the local PIT-distribution quantiles
    for alpha in alphas:
        W_a_train = (pit_values_train <= alpha).astype(int)  # compute the targets
        clf = sklearn.base.clone(classifier)
        clf.fit(X=x_train, y=W_a_train)  # train classifier
        clfs[alpha] = clf

    return clfs

def infer_r_alphas_baseline(x_eval, clfs):
    alphas = np.array(list(clfs.keys()))
    r_alphas = {}
    for alpha in alphas:
        # evaluate in x_0
        prob = clfs[alpha].predict_proba(x_eval)
        if prob.shape[1] < 2:  # Dummy Classifier
            r_alphas[alpha] = prob[:, 0][0]
        else:  # MLPClassifier
            r_alphas[alpha] = prob[:, 1][0]
    return r_alphas


# AMORTIZED IN ALPHA
def localPIT_regression_grid(
    pit_values_train, x_train, classifier=DEFAULT_CLF,
):
    """Method 2: Train the Classifier amortized on x and all alpha"""
    alphas = np.linspace(0,1,100)
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

    # define classifier
    clf = sklearn.base.clone(classifier)
    # train classifier
    clf.fit(X=train_features, y=W_a_train.ravel())  # train classifier

    return clf

def localPIT_regression_sample(
    pit_values_train,
    x_train,
    nb_samples=1,
    classifier=DEFAULT_CLF,
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

    # define classifier
    clf = sklearn.base.clone(classifier)
    # train classifier
    clf.fit(X=train_features, y=W_a_train.ravel())  # train classifier

    return clf

def infer_r_alphas_amortized(x_eval, alphas, clf):
    # Evaluate in x_0 and for all alphas in [0,1]
    r_alphas = {}
    for alpha in alphas:
        test_features = np.concatenate([x_eval, np.array(alpha).reshape(-1, 1)], axis=1)
        r_alphas[alpha] = clf.predict_proba(test_features)[:, 1][0]

    return r_alphas

