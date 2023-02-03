# The content of this file was directly copied from 
# https://github.com/mackelab/sbi/blob/3e3522f177d4f56f3a617b2f15a5b2e25360a90f/sbi/utils/metrics.py
# and modified to use classifiers such as LDA or QDA where no random seed can be defined in the clf_kwargs
# (cf. see comment "# changed")

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor

def c2st_scores(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
    clf_class: Any = RandomForestClassifier,
    clf_kwargs: Dict[str, Any] = {},
) -> Tensor:
    """
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.
    This function performs training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used which
    is recommended based on the study performed in [4].
    This can be changed using <clf_class>. This class is used in the following
    fashion:
    ``` py
    clf = clf_class(random_state=seed, **clf_kwargs)
    #...
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )
    ```
    Further configuration of the classifier can be performed using <clf_kwargs>.
    If you like to provide a custom class for training, it has to satisfy the
    internal requirements of `sklearn.model_selection.cross_val_score`.
    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross validation
        n_folds: Number of folds to use for cross validation
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        z_score: Z-scoring using X, i.e. mean and std deviation of X is used to normalize Y, i.e. Y=(Y - mean)/std
        noise_scale: If passed, will add Gaussian noise with standard deviation <noise_scale> to samples of X and of Y
        verbosity: control the verbosity of sklearn.model_selection.cross_val_score
        clf_class: a scikit-learn classifier class
        clf_kwargs: key-value arguments dictionary to the class specified by clf_class, e.g. sklearn.ensemble.RandomForestClassifier
    Return:
        np.ndarray containing the calculated <metric> scores over the test set
        folds from cross-validation
    Example:
    ``` py
    > c2st_scores(X,Y)
    [0.51904464,0.5309201,0.4959452,0.5487709,0.50682926]
    #X and Y likely come from the same PDF or ensemble
    > c2st_scores(P,Q)
    [0.998456,0.9982912,0.9980476,0.9980488,0.99805826]
    #P and Q likely come from two different PDFs or ensembles
    ```
    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    # ===== changed ===== #
    try:
        clf = clf_class(random_state=seed, **clf_kwargs)
    except TypeError:
        clf = clf_class(**clf_kwargs)
    # ===== changed ===== #

    # prepare data
    data = np.concatenate((X, Y))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity
    )

    return scores
