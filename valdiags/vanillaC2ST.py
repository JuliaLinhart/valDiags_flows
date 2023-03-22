# Implementation of vanilla C2ST (Classifier Two Sample Test)
# - [Lopez et al. (2017)](https://arxiv.org/abs/1610.06545)
# - [Lee et al. (2018)](https://arxiv.org/abs/1812.08927)

import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import sklearn

from .c2st_utils import compute_metric

from tqdm import tqdm

# define default classifier
DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


def train_c2st(P, Q, clf=DEFAULT_CLF):
    """ Trains a classifier to distinguish between data from P and Q.

    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        clf (sklearn model, optional): the initialized classifier to use.
            needs to have a method `.fit(X,y)`. 
            Defaults to DEFAULT_CLF.

    Returns:
        (sklearn model): trained classifier (cloned from clf).
    """

    # define features and labels
    features = np.concatenate([P, Q], axis=0)  # (2*n_samples, dim)
    labels = np.concatenate(
        [np.array([0] * len(P)), np.array([1] * len(Q))]
    ).ravel()  # (2*n_samples,)

    # shuffle features and labels
    features, labels = shuffle(features, labels)

    # train the classifier
    clf = sklearn.base.clone(clf)
    clf.fit(X=features, y=labels)

    return clf


def eval_c2st(P, Q, clf, single_class_eval=False):
    """Evaluates a classifier on data from P and Q.
    
    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        clf (sklearn model): the trained classifier on both classes.
            needs to have a methods `.score(X,y)` and `.predict_proba(X)`.
        single_class_eval (bool, optional): if True, only evaluate on P.
            Defaults to False.
    
    Returns:
        (float, numpy.array): accuracy and proba of class 0.
    """

    n_samples = len(P)
    # define features and labels
    if single_class_eval:
        X_val = P  # only evaluate on P
        y_val = np.array([0] * (n_samples))  # labels are all 0
    else:
        X_val = np.concatenate([P, Q], axis=0)  # evaluate on both P and Q
        y_val = np.array(
            [0] * n_samples + [1] * n_samples
        )  # labels are 0 for P and 1 for Q

    # evaluate the classifier
    accuracy = clf.score(X_val, y_val)  # accuracy
    proba = clf.predict_proba(X_val)[:, 0]  # proba of class 0

    return accuracy, proba


def c2st_scores(
    P,
    Q,
    metrics=["accuracy"],
    n_folds=10,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    single_class_eval=False,
    cross_val=True,
):
    """Computes scores for a classifier trained on P and Q. 
    They represent the test statistics of the C2ST test estimated on P and Q.
    
    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        metrics (list of str, optional): list of names of metrics to compute.
            Defaults to ["accuracy"].
        n_folds (int, optional): number of folds for cross-validation.
            Defaults to 10.
        clf_class (sklearn model class, optional): the class of classifier to use.
            needs to have a methods `.fit(X,y)`, score(X,y)` and `.predict_proba(X)`.
            Defaults to MLPClassifier.
        clf_kwargs (dict, optional): keyword arguments for clf_class.
            Defaults to {"alpha": 0, "max_iter": 25000}.
        single_class_eval (bool, optional): if True, only evaluate on P.
            Defaults to False.
        cross_val (bool, optional): if True, perform cross-validation.
            Defaults to True.
    
    Returns:
        (dict): dictionary of computed scores, i.e. estimated test statistics on P and Q.
    """
    # initialize classifier
    classifier = clf_class(**clf_kwargs)

    if not cross_val:
        # train classifier
        clf = train_c2st(P, Q, clf=classifier)
        # eval classifier
        accuracy, proba = eval_c2st(P, Q, clf=clf, single_class_eval=single_class_eval)
        # compute metrics
        scores = {}
        for m in metrics:
            if "accuracy" in m:
                scores[m] = accuracy
            else:
                scores[m] = compute_metric(
                    proba, metrics=[m], single_class_eval=single_class_eval
                )[m]

    else:
        # initialize scores as dict of empty lists
        scores = dict(zip(metrics, [[] for _ in range(len(metrics))]))

        # cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(P):
            # split data into train and val sets for n^th cv-fold
            P_train = P[train_index]
            P_val = P[val_index]
            Q_train = Q[train_index]
            Q_val = Q[val_index]

            # train n^th classifier
            clf_n = train_c2st(P_train, Q_train, clf=classifier)
            # eval n^th classifier
            accuracy, proba = eval_c2st(
                P_val, Q_val, clf=clf_n, single_class_eval=single_class_eval
            )
            # compute metrics
            for m in metrics:
                if "accuracy" in m:
                    scores[m].append(accuracy)
                else:
                    scores[m].append(
                        compute_metric(
                            proba, metrics=[m], single_class_eval=single_class_eval
                        )[m]
                    )

    return scores

