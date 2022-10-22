import torch
import numpy as np


def hpd_region(
    posterior, prior, param_grid, x, confidence_level, n_p_stars=100_000, tol=0.01
):
    """High Posterior Region for learned flow posterior.
    Code adapted from WALDO git-repo."""
    if posterior is None:
        # actually using prior here; naming just for consistency (should be changed)
        posterior_probs = torch.exp(prior.log_prob(torch.from_numpy(param_grid)))
    else:
        posterior_probs = torch.exp(posterior._log_prob(inputs=param_grid, context=x))
    posterior_probs /= torch.sum(posterior_probs)  # normalize
    p_stars = np.linspace(0.99, 0, n_p_stars)  # thresholds to include or not parameters
    current_confidence_level = 1
    new_confidence_levels = []
    idx = 0
    while np.abs(current_confidence_level - confidence_level) > tol:
        if idx == n_p_stars:  # no more to examine
            break
        new_confidence_level = torch.sum(
            posterior_probs[posterior_probs >= p_stars[idx]]
        ).item()
        new_confidence_levels.append(new_confidence_level)
        if np.abs(new_confidence_level - confidence_level) < np.abs(
            current_confidence_level - confidence_level
        ):
            current_confidence_level = new_confidence_level
        idx += 1
    # all params such that p(params|x) > p_star, where p_star is the last chosen one
    return (
        current_confidence_level,
        param_grid[posterior_probs >= p_stars[idx - 1], :],
        new_confidence_levels,
    )


def waldo_confidence_region(
    posterior_samples, critical_values, param_grid, grid_sample_size
):
    """Calibrated Confidence Set for learned flow posterior using Waldo statistic.
    Code adapted from WALDO git-repo."""

    # compute posterior mean and variance
    posterior_mean = torch.mean(posterior_samples, dim=0)
    posterior_var = torch.cov(torch.transpose(posterior_samples, 0, 1))
    confidence_set_i = []
    for j in range(grid_sample_size):
        # compute waldo stats for every theta in the grid
        obs_statistics = waldo_stats(
            posterior_mean,
            posterior_var,
            param=torch.from_numpy(param_grid[j, :]).float(),
        )
        # compare to critical value
        if obs_statistics <= critical_values[j]:
            confidence_set_i.append(param_grid[j, :].reshape(1, 2))
    confidence_set = np.vstack(confidence_set_i)
    return confidence_set


def waldo_stats(posterior_mean, posterior_var, param):
    stats = torch.matmul(
        torch.matmul(
            torch.t(
                torch.subtract(posterior_mean.reshape(-1, 1), param.reshape(-1, 1))
            ),
            torch.linalg.inv(posterior_var),
        ),
        torch.subtract(posterior_mean.reshape(-1, 1), param.reshape(-1, 1)),
    ).item()
    return stats


@torch.no_grad()
def highest_density_level(pdf, alpha, bias=0.0, min_epsilon=10e-17, region=False):
    """ Code from https://github.com/montefiore-ai/hypothesis/blob/master/hypothesis/stat/constraint.py
    Used by Hermans et al. (2021), "Averting a Crisis in SBI"
    """
    # Check if a proper bias has been specified.
    if bias >= alpha:
        raise ValueError("The bias cannot be larger or equal to the specified alpha level.")
    # Detect numpy type
    if type(pdf).__module__ != np.__name__:
        pdf = pdf.cpu().clone().numpy()
    else:
        pdf = np.array(pdf)
    total_pdf = pdf.sum()
    pdf /= total_pdf
    # Compute highest density level and the corresponding mask
    n = len(pdf)
    optimal_level = pdf.max().item()
    epsilon = 10e-02
    while epsilon >= min_epsilon:
        area = float(0)
        while area <= (alpha + bias):
            # Compute the integral
            m = (pdf >= optimal_level).astype(np.float32)
            area = np.sum(m * pdf)
            # Compute the error and apply gradient descent
            optimal_level -= epsilon
        optimal_level += 2 * epsilon
        epsilon /= 10
    optimal_level *= total_pdf
    if region:
        return optimal_level, torch.from_numpy(m)
    else:
        return optimal_level

def highest_density(flow, testset, n_samples=1000):
    """ Highest Density Levels for coverage tests:

    We check if a true sample x_0 is in the highest density region of the flow-estimator q
    at level 1-alpha, which is equivalent to the proportion of samples x ~ q 
    having a higher estimated density than x_0: E_x[I_{q(x)>q(x_0)}].
    
    By computing this for a large number of x_0, covering the space of the true distribution p(x),
    we get the expected coverage (or levels) over all possible covergage levels in [0,1].

    If q = p, these should be uniformly distributed over [0,1].

    Following the implementation from
    https://github.com/francois-rozet/lampe/blob/master/lampe/diagnostics.py
    adapted to non-sbi distributions. 
    """
    ranks = []

    with torch.no_grad():
        for x_0 in testset:

            samples = flow.sample(n_samples)
            mask = flow.log_prob(x_0[None,:]) < flow.log_prob(samples)
            rank = mask.sum() / mask.numel()

            ranks.append(rank)

    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, torch.tensor([0.0, 1.0])))


    levels = torch.sort(ranks).values
    coverages = torch.linspace(0.0, 1.0, len(ranks))
    return levels, coverages