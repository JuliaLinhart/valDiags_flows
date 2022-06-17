import torch
import numpy as np

from scipy.stats import norm
from torch.distributions.multivariate_normal import MultivariateNormal


class ConditionalGaussian1d:
    """1D Conditional Gaussian Distribution with 2D gaussian feature space:
    X ~ N(mu,Sigma), X = [X_1, X_2]
    Theta ~ N(X_1 + X_2, 1)
    """

    def __init__(self, mu=[0, 0], sigma=[1, 1], rho=0.8) -> None:
        # Distribution of X: 2d gaussian
        self.mean = torch.FloatTensor(mu)
        self.cov = torch.FloatTensor(
            [
                [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
                [rho * sigma[0] * sigma[1], sigma[1] ** 2],
            ]
        )

        self.x_dist = MultivariateNormal(loc=self.mean, covariance_matrix=self.cov)

    def sample_x(self, n):
        return self.x_dist.sample((n,))

    def sample_theta(self, x):
        samples_x = x.numpy()
        samples_theta = np.random.normal(
            loc=samples_x[:, 0] + samples_x[:, 1], scale=1
        ).reshape(-1, 1)
        return torch.FloatTensor(samples_theta)

    def get_joint_data(self, n):
        # samples from P(X)
        samples_x = self.sample_x(n)
        # samples from P(Theta|X)
        samples_theta = self.sample_theta(samples_x)
        return samples_x, samples_theta

    def true_pit_values(self, samples_theta, samples_x, local=False):
        # Joint PIT-values [PIT(Theta_i, X_i, f_{Theta|X})]
        if local:
            pit_values = norm.cdf(
                samples_theta, loc=samples_x[:, 0] + samples_x[:, 1], scale=1
            )
        else:
            pit_values = np.array(
                [
                    norm.cdf(samples_theta[i], loc=x[0] + x[1], scale=1)
                    for i, x in enumerate(samples_x)
                ]
            )
        return pit_values


class SBIGaussian2d:
    def __init__(self, prior, x_correlation_factor=0.8) -> None:
        """2d Gaussian.
        Inference of mean under given prior.
        """
        self.prior = prior
        self.x_correlation_factor = x_correlation_factor

    def simulator(self, theta):
        # Distribution parameters
        rho = self.x_correlation_factor
        cov = torch.FloatTensor([[1, rho], [rho, 1]])
        # Sample X
        samples_x = MultivariateNormal(loc=theta, covariance_matrix=cov).sample((1,))[0]
        return samples_x

    def get_joint_data(self, n):
        samples_prior = self.prior.sample((n,))
        samples_x = self.simulator(samples_prior)
        return samples_prior, samples_x

    def true_posterior_pdf(self, x_obs):
        def true_posterior_prob(samples):
            log_p_prior = self.prior.log_prob(samples)
            rho = self.x_correlation_factor
            cov = torch.FloatTensor([[1, rho], [rho, 1]])
            if x_obs.ndim > 1:
                log_p_x = MultivariateNormal(
                    loc=samples, covariance_matrix=cov
                ).log_prob(torch.mean(x_obs, axis=0))
            else:
                log_p_x = MultivariateNormal(
                    loc=samples, covariance_matrix=cov
                ).log_prob(x_obs)
            log_p = log_p_prior + log_p_x
            return get_proba(log_p)
        return true_posterior_prob
    
    def true_posterior(self, x_obs):
        rho = self.x_correlation_factor
        cov = torch.FloatTensor([[1, rho], [rho, 1]])
        return MultivariateNormal(loc=x_obs, covariance_matrix=cov)


def get_proba(log_p):
    """Compute probability from log_prob in a safe way.

    Parameters
    ----------
    log_p: ndarray, shape (n_samples*n_samples,)
        Values of log_p for each point (a,b) of the samples (n_samples, n_samples).
    """
    if isinstance(log_p, np.ndarray):
        log_p = torch.tensor(log_p)
    log_p = log_p.to(dtype=torch.float64)
    log_p -= torch.logsumexp(log_p, dim=-1)
    return torch.exp(log_p)
