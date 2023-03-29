# ==== C2ST: local version ====
# Implementation based on the vanilla C2ST method.
# Author: Julia Linhart
# Institution: Inria Paris-Saclay (MIND team)

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

from .test_utils import compute_pvalue

from .vanillaC2ST import train_c2st, eval_c2st, compute_metric

# define default classifier
DEFAULT_CLF = MLPClassifier(alpha=0, max_iter=25000)


# ==== train / eval functions for the classifier used in L-C2ST ====


def train_lc2st(P, Q, x_P, x_Q, clf=DEFAULT_CLF):
    """ Trains a classifier to distinguish between data from two joint distributions
    
        - P,x = P * x|P (where x|P is denoted as x_P)
        - Q,x = Q * x|Q (where x|Q is denoted as x_Q)
    
    This function is built on the original `train_c2st`, adapting it to joint distributions.
    
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


# ==== L-C2ST test functions ====
# - estimate the test statistics by computing the c2st metrics on a data sample
#   (ensemble in-sample / out-sample or cross-validation)
# - perform the test by computing
#       * the (ensemble) L-C2ST test statistics on observed data (not cross-val, not in-sample)
#       * a sample of test statistics under the null hypothesis (not cross-val, not in-sample)
# - infer test statistics on observed data and under the null (+ probabilities used to compute them)


def lc2st_scores(
    P,
    Q,
    x_P,
    x_Q,
    x_eval,
    P_eval=None,
    Q_eval=None,
    metrics=["accuracy"],
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    single_class_eval=True,
    cross_val=True,
    n_folds=10,
    n_ensemble=1,
    in_sample=False,
):
    """Computes the scores of a classifier 
        - trained on data from the joint distributions P,x and Q,x
        - evaluated on data from the conditional distributions P|x and/or Q|x 
        at a fixed observation x=x_eval.
    
    They represent the test statistics of the local C2ST test between P|x and Q|x at x=x_eval.

    If at least one of the classes (P or Q) is independent of x, we don't need extra data 
    P_eval and/or Q_eval during cross-validation. We can directly use the validation split of 
    P and/or Q to evaluate the classifier. This is the default behavior.

    By default, we only evaluate on P|x: `single_class_eval` is set to `True`.
    This is typically the case in SBI, where we generally do not have access to data from the 
    class representing the true posterior.


    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        x_P (numpy.array): data drawn from p(x), such that [P ,x_P] ~ p(P,x)
            of size (n_samples, n_features)
        x_Q (numpy.array): data drawn from p(x), such that [Q ,x_Q] ~ p(Q,x)
            of size (n_samples, n_features).
        x_eval (numpy.array): a fixed observation
            of size (n_features,).
        P_eval (numpy.array, optional): data drawn from P|x_eval (or just P if independent of x)
            of size (n_test_samples, dim). 
            Has to be provided if P is not independent of x.
            Defaults to None.
        Q_eval (numpy.array, optional): data drawn from Q|x_eval (or just Q if independent of x)
            of size (n_test_samples, dim).
            Defaults to None.
        metrics (list of str, optional): list of metric names to compute.
            Defaults to ["accuracy"].
        clf_class (sklearn model class, optional): the class of the lassifier to use.
            Defaults to MLPClassifier.
        clf_kwargs (dict, optional): the keyword arguments for the classifier.
            Defaults to {"alpha": 0, "max_iter": 25000}.
        single_class_eval (bool, optional): whether to evaluate on P only (True) or on P and Q (False).
            Defaults to True.
        cross_val (bool, optional): whether to perform cross-validation (True) or not (False).
            Defaults to True.
        n_folds (int, optional): number of folds for cross-validation.
            Defaults to 10.
        n_ensemble (int, optional): number of classifiers to train and average over to build an ensemble model.
            Defaults to 1.
        in_sample (bool, optional): whether to evaluate on training data (True) or not (False).
            Can only be used if P is independent of x.
            Defaults to False. 
    
    Returns:
        (dict): dictionary of scores (accuracy, proba, etc.) for each metric.
    """

    # initialize classifier
    classifier = clf_class(**clf_kwargs)

    if not cross_val:
        ens_accuracies = []
        ens_probas = []
        for _ in range(n_ensemble):
            # train classifier
            clf = train_lc2st(P, Q, x_P, x_Q, clf=classifier)

            # eval classifier
            if in_sample:  # evaluate on training data
                P_eval, Q_eval = P, None  # ok if P is independent of x

            elif P_eval is None:
                raise ValueError(
                    "If cross_val=False and in-sample=False, at least P_eval must be provided.\
                    In this case an out-of-sample evaluation is performed (single-class if Q_eval=None)."
                )

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
            P_val = P[val_index]  # ok if P is independent of x
            if P_eval is not None:
                P_val = P_eval[val_index]

            Q_train, x_Q_train = Q[train_index], x_Q[train_index]
            if Q_eval is not None:
                Q_val = Q_eval[val_index]
            else:
                Q_val = None

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


def t_stats_lc2st(
    P,
    Q,
    x_P,
    x_Q,
    P_eval,
    x_eval,
    Q_eval=None,
    test_stats=["probas_mean"],
    n_ensemble_obs=10,
    n_trials_null=100,
    t_stats_null=None,
    list_P_null=None,
    list_x_P_null=None,
    list_P_eval_null=None,
    single_class_eval=True,
    return_probas=True,
    **kwargs,
):
    """Performs hypothesis test for LC2ST.
    We compute the test statistic for the observed data and compare it to the test statistic of the null 
    distribution. 

    - For the observed data, we compute the test statistic using `lc2st_scores`.
    - Under the null distribution, we either use the pre-computed scores (if `t_stats_null` is provided)
    or we compute the test statistics for each trial using `lc2st_scores` on each element of the provided
    lists of null samples. 
    
    In sbi, we typically do not have access to data from both classes during evaluation, therefore we cannot 
    use the permutation method to simulate the null hypothesis as in the classical c2st setting.
    This is why this method is not implemented here. (we could add it in the future if needed with a 
    statement "if Q_eval is not None: ... else: ...").

    Args:
        P (numpy.array): data drawn from P
            of size (n_samples, dim).
        Q (numpy.array): data drawn from Q
            of size (n_samples, dim).
        x_P (numpy.array): data drawn from from p(x), such that [P ,x_P] ~ p(P,x)
            of size (n_samples, n_features).
        x_Q (numpy.array): data drawn from from p(x), such that [Q ,x_Q] ~ p(Q,x)
            of size (n_samples, n_features).
        P_eval (numpy.array): data drawn from P|x_eval (or just P if independent of x)
            of size (n_test_samples, dim).
        x_eval (numpy.array): observed data
            of size (n_features,).
        Q_eval (numpy.array, optional): data drawn from Q|x_eval (or just Q if independent of x)
            of size (n_test_samples, dim). If None, Q_eval, we only evaluate on P_eval (single_class_eval=True).
            Defaults to None.
        test_stats (list of str, optional): list of names of test statistics to compute.
            Defaults to ["probas_mean"] (better than accuracyfor single_class_eval).
        n_ensemble_obs (int, optional): number of classifiers to build an ensemble model on observed data.
            Defaults to 10.
        n_trials_null (int, optional): number of trials to perform for null hypothesis.
            Defaults to 100.
        t_stats_null (dict, optional): dictionary of precumputed scores (accuracy, proba, etc.) for each 
            metric under the null hypothesis. 
            Defaults to None.
        list_P_null (list): list of samples from P used as "P" and "Q" to test under the null 
            hypothesis. 
            Of size (2*n_trials_null, n_samples, dim).
            Defaults to None.
        list_x_P_null (list): list of samples like x_P used as x_P and x_Q to test under the null 
            hypothesis. Of size (2*n_trials_null, n_samples, n_features).
            Defaults to None.
        list_P_eval_null (list): list of samples from P_eval used as "P_eval" and "Q_eval" to test under the 
            null hypothesis. Of size (2*n_trials_null, n_test_samples, dim).
            Defaults to None.
        single_class_eval (bool, optional): whether to evaluate the classifier only on P or on P and Q.
            Defaults to True.
        return_probas (bool, optional): whether to return predicted probabilities.
            Defaults to True.
        kwargs: keyword arguments for `lc2st_scores`.
    
    Returns:
        t_stats_ensemble (dict): test statistics for the ensemble model.
        proba_ensemble (numpy.array): predicted probabilities of class 0 for the ensemble model.
            only returned if return_probas is True.
        t_stats_null (list of dict): list of test statistics computed under the null hypothesis.
        probas_null (list of numpy.array): list of predicted probabilities under the null hypothesis.
            only returned if return_probas is True.
    """

    t_stats_ensemble, proba_ensemble = lc2st_scores(
        P,
        Q,
        x_P,
        x_Q,
        x_eval,
        P_eval,
        Q_eval,
        metrics=test_stats,
        single_class_eval=single_class_eval,
        n_ensemble=n_ensemble_obs,
        **kwargs,
    )

    probas_null = []
    if t_stats_null is None:
        t_stats_null = dict(zip(test_stats, [[] for _ in range(len(test_stats))]))
        for t in range(n_trials_null):
            scores_t, proba_t = lc2st_scores(
                P=list_P_null[t],
                Q=list_P_null[n_trials_null + t],
                x_P=list_x_P_null[t],
                x_Q=list_x_P_null[n_trials_null + t],
                x_eval=x_eval,
                P_eval=list_P_eval_null[t],  # a new sample for each trial?
                Q_eval=list_P_eval_null[
                    n_trials_null + t
                ],  # a new sample for each trial?
                metrics=test_stats,
                single_class_eval=single_class_eval,
                n_ensemble=1,  # only one classifier
                **kwargs,
            )
            probas_null.append(proba_t)

            # append test stat to list
            for m in test_stats:
                t_stats_null[m].append(
                    np.mean(scores_t[m])
                )  # compute their mean (useful if cross_val=True)

    if return_probas:
        return t_stats_ensemble, proba_ensemble, t_stats_null, probas_null
    else:
        return t_stats_ensemble, t_stats_null

