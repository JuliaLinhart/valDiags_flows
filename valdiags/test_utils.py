import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_pvalue(t_stat_est, t_stats_null):
    return (t_stat_est < t_stats_null).mean()


def eval_htest(p_value, conf_alpha=0.05):
    return p_value <= conf_alpha  # True = reject


def empirical_error_htest(
    t_stats_estimator, metrics, conf_alpha=0.05, n_runs=100, **kwargs
):
    success_rate = dict(zip(metrics, [0] * len(metrics)))
    print(f"Computing empirical error as the success rate over {n_runs} runs:")
    for _ in tqdm(range(n_runs)):
        t_stat_data, t_stats_null = t_stats_estimator(metrics=metrics, **kwargs)
        for m in metrics:
            p_value = compute_pvalue(t_stat_data[m], t_stats_null[m])
            # power = true positive (TP): count if rejected under H1
            # type 1 = false positive (FP): count if rejected under H0
            success_rate[m] += eval_htest(p_value, conf_alpha) / n_runs

    return success_rate
