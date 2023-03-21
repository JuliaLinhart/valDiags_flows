import numpy as np


def compute_pvalue(t_stat_est, t_stats_null):
    """ Computes the p-value of a hypothesis test as the empirical estimate of:
                    
                        p = Prob(T > \hat{T} | H0) 
        
        which represents the probability of making a type 1 error, i.e. the probability 
        of falsly rejecting the null hypothesis (H0).
    
    -------
    inputs:
    - t_stat_est: float
        test statistic \hat{T} estimated on observed data
    - t_stats_null: list or array 
        a sample of the test statistic drawn under the null hypothesis: 
            --> t_i ~ T|(H0)
    
    -------
    returns:
    - p_value: float
    """
    return (t_stat_est < np.array(t_stats_null)).mean()


def eval_htest(t_stats_estimator, metrics, conf_alpha=0.05, **kwargs):
    """ Computes the result of a hypothesis test at a given significance level.
    A test is rejected if the p-value is lower than the given significance level,
    meaning that the probability of making a type 1 error will be "small enough". 

    -------
    inputs:
    - t_stats_estimator: function that
        * takes as input a list of metrics that computed on an observed data sample will 
        give an estimate of the corresponding test statistic
        * returns objects taken as inputs in `compute_pvalue` 
        (i.e. test statistic estimated on observed data and drawn under the null hypothesis)
    - metrics: list of str
        contains the names of the metrics used in `t_stats_estimator`
    - conf_alpha: float
        significance level of the test, yielding a confidence level of (1-alpha).
    - kwargs: dict
        --> any additional kwarg of the function `t_stats_estimator` can be added

    -------
    returns:
    - reject: bool
        True if the test is rejected, False otherwise. 
    """
    reject = {}
    t_stat_data, t_stats_null = t_stats_estimator(metrics=metrics, **kwargs)
    for m in metrics:
        p_value = compute_pvalue(t_stat_data[m], t_stats_null[m])
        reject[m] = p_value < conf_alpha  # True = reject
    return reject

