# Implementation of the local C2ST method simpler version based on the vanilla C2ST method
# Author: Julia Linhart
# Institution: Inria Paris-Saclay (MIND team)

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

from .test_utils import compute_pvalue
from .c2st_utils import compute_metric

from .vanillaC2ST import train_c2st, eval_c2st

# define default classifier
DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


def train_lc2st(P, Q, x_P, x_Q, clf=DEFAULT_CLF):
    """ Trains a classifier to distinguish between data from the joint distributions
    
        - P,x = P * x|P (where x|P is denoted as x_P)
        - Q,x = Q * x|Q (where x|Q is denoted as x_Q)
    
    This function is built on the `train_c2st`, adapting it to joint distributions.
    
    Example for SBI:
    ----------------
        - P is the prior and x_P is generated via the simulator from the parameters P.
        - Q is the approximate posterior amortized in x. x_Q is a shuffled version of x_P, 
        used to generate independant samples from Q | x.

    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        x_P (numpy.array): data drawn from p(x), such that [P ,x_P] ~ p(P,x)
            of size (n_samples, n_features)
        x_Q (numpy.array): data drawn from p(x), such that [Q ,x_Q] ~ p(Q,x)
            of size (n_samples, n_features).
        clf (sklearn model, optional): the initialized classifier to use.
            needs to have a method `.fit(X,y)`. 
            Defaults to DEFAULT_CLF.

    Returns:
        (sklearn model): trained classifier (cloned from clf).
    """

    # concatenate P/Q and x_P/x_Q to get data from the joint distributions
    joint_P_x = np.concatenate([P, x_P], axis=1)
    joint_Q_x = np.concatenate([Q, x_Q], axis=1)

    # train the classifier
    clf = train_c2st(joint_P_x, joint_Q_x, clf=clf)
    return clf


def eval_lc2st(P, x_eval, clf, Q=None, single_class_eval=True):
    """Evaluates a classifier trained on data from the joint distributions

        - P,x
        - Q,x

    at a fixed observation x=x_eval. 
    
    This function is built on the `eval_c2st`, adapting it to evaluate conditional
    ditributions at a fixed observation x_eval. By default, we only evaluate on P. 
    
    Example for SBI: 
    ----------------
    We here typically do not know the true posterior and can only evaluate on data 
    generated from the approximate posterior at fixed x=x_eval.

    Args:
        P (numpy.array): data drawn from P|x_eval (or just P if independent of x)
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q|x_eval (or just Q if independent of x)
            of size (n_samples, dim).
        x_eval (numpy.array): a fixed observation
            of size (n_features,).
        clf (sklearn model, optional): needs to have a methods `.score(X,y)` and `.predict_proba(X)`.
            Defaults to DEFAULT_CLF.
    
    Returns:
        (numpy.array): predicted probabilities for class 0 (P|x) (and accuracy if y is not None).
        
    """
    # concatenate P with repeated x_eval to match training data format
    P_x_eval = np.concatenate([P, x_eval.repeat(len(P), 1)], axis=1)
    if Q is not None:
        Q_x_eval = np.concatenate([Q, x_eval.repeat(len(Q), 1)], axis=1)
    else:
        Q_x_eval = None
        single_class_eval = True  # if Q is None, we can only evaluate on P

    # evaluate the classifier: accuracy and predicted probabilities for class 0 (P|x_eval)
    accuracy, proba = eval_c2st(
        P=P_x_eval, Q=Q_x_eval, clf=clf, single_class_eval=single_class_eval
    )

    return accuracy, proba


def lc2st_scores(
    P,
    Q,
    x_P,
    x_Q,
    x_eval,
    P_eval=None,
    Q_eval=None,
    metrics=["accuracy"],
    n_folds=10,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    single_class_eval=True,
    cross_val=True,
    in_sample=False,
    n_ensemble=1,
):

    # initialize classifier
    classifier = clf_class(**clf_kwargs)

    if not cross_val:
        ens_accuracies = []
        ens_probas = []
        for _ in range(n_ensemble):
            # train classifier
            clf = train_lc2st(P, Q, x_P, x_Q, clf=classifier)

            # eval classifier
            if in_sample:
                P_eval, Q_eval = P, Q

            accuracy, proba = eval_lc2st(
                P=P_eval,
                Q=Q_eval,
                x_eval=x_eval,
                clf=clf,
                single_class_eval=single_class_eval,
            )

            ens_accuracies.append(accuracy)
            ens_probas.append(proba)

        # compute accuracy and proba of ensemble model
        accuracy = np.mean(ens_accuracies, axis=0)
        probas = np.mean(ens_probas, axis=0)

        # compute metrics
        scores = {}
        for m in metrics:
            if "accuracy" in m:
                scores[m] = accuracy
            else:
                scores[m] = compute_metric(
                    probas, metrics=[m], single_class_eval=single_class_eval
                )[m]

    else:
        # initialize scores as dict of empty lists
        scores = dict(zip(metrics, [[] for _ in range(len(metrics))]))
        probas = []

        # cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(P):
            # split data into train and val sets for n^th cv-fold
            P_train, x_P_train = P[train_index], x_P[train_index]
            P_val = P[val_index]
            if P_eval is not None:
                P_val = P_eval[val_index]

            Q_train, x_Q_train = Q[train_index], x_Q[train_index]
            Q_val = Q[val_index]
            if Q_eval is not None:
                Q_val = Q_eval[val_index]

            # train n^th classifier
            clf_n = train_lc2st(
                P=P_train, Q=Q_train, x_P=x_P_train, x_Q=x_Q_train, clf=classifier
            )
            # eval n^th classifier
            accuracy, proba = eval_lc2st(
                P=P_val,
                Q=Q_val,
                x_eval=x_eval,
                clf=clf_n,
                single_class_eval=single_class_eval,
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
            probas.append(proba)

    return scores, probas


def lc2st_htest(
    P,
    Q,
    x_P,
    x_Q,
    P_eval,
    x_eval,
    list_null_samples_P,
    list_null_samples_x_P,
    test_stats=["probas_mean"],
    n_ensemble=10,
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    precomputed_probas_null=None,
    Q_eval=None,
    P_eval_null=None,
    single_class_eval=True,
):

    t_stats_ensemble, proba_ensemble = lc2st_scores(
        P,
        Q,
        x_P,
        x_Q,
        x_eval,
        P_eval,
        Q_eval,
        metrics=test_stats,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        single_class_eval=single_class_eval,
        cross_val=False,
        in_sample=False,
        n_ensemble=n_ensemble,
    )

    t_stats_null = dict(zip(test_stats, [[] for _ in range(len(test_stats))]))
    n_trials_null = len(list_null_samples_P)
    probas_null = []
    for t in range(n_trials_null):
        if precomputed_probas_null is not None:
            probas_null.append(precomputed_probas_null[t])
            # compute test stat
            scores_t = compute_metric(
                probas_null[t], metrics=test_stats, single_class_eval=single_class_eval
            )
        else:
            scores_t, proba_t = lc2st_scores(
                P=P,
                Q=list_null_samples_P[t],
                x_P=x_P,
                x_Q=list_null_samples_x_P[t],
                x_eval=x_eval,
                P_eval=P_eval,
                Q_eval=P_eval_null,
                metrics=test_stats,
                clf_class=clf_class,
                clf_kwargs=clf_kwargs,
                single_class_eval=single_class_eval,
                cross_val=False,
                in_sample=False,
                n_ensemble=1,
            )
            probas_null.append(proba_t)

        # append test stat to list
        for m in test_stats:
            t_stats_null[m].append(scores_t[m])

    p_values = {}
    for m in test_stats:
        p_values[m] = compute_pvalue(t_stats_ensemble[m], np.array(t_stats_null[m]))

    return p_values, t_stats_ensemble, proba_ensemble, probas_null, t_stats_null

