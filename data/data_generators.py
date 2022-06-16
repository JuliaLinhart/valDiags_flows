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
        self.covariance = torch.FloatTensor(
            [
                [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
                [rho * sigma[0] * sigma[1], sigma[1] ** 2],
            ]
        )

        self.x_dist = MultivariateNormal(
            loc=self.mean, covariance_matrix=self.covariance
        )

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
                samples_theta, loc=samples_x[:,0] + samples_x[:,1], scale=1
            )
        else:
            pit_values = np.array(
                [
                    norm.cdf(samples_theta[i], loc=x[0] + x[1], scale=1)
                    for i, x in enumerate(samples_x)
                ]
            )
        return pit_values
