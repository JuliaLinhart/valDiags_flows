import torch
import numpy as np

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

def highest_density(flow, theta_test, x_test=None, n_samples=1000, nflows_flow=True):
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
        if x_test is None:
            for theta_0 in theta_test:
                samples = flow.sample(n_samples)
                mask = flow.log_prob(theta_0[None,:]) < flow.log_prob(samples)
                rank = mask.sum() / mask.numel()
                ranks.append(rank)
        else:
            for theta_0, x_0 in zip(theta_test, x_test):
                theta_0, x_0 = theta_0[None,:], x_0[None,:]
                samples = flow.sample(len(theta_test), context = x_0)
                if not nflows_flow:
                    samples = samples[0]
                repeat_dims = tuple([samples.shape[0]]+[1 for _ in range(x_0.ndim-1)])
                mask = flow.log_prob(theta_0, context=x_0) < flow.log_prob(samples, context=x_0.repeat(repeat_dims))
                rank = mask.sum() / mask.numel()
                ranks.append(rank)

    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, torch.tensor([0.0, 1.0])))


    levels = torch.sort(ranks).values
    coverages = torch.linspace(0.0, 1.0, len(ranks))
    return levels, coverages

