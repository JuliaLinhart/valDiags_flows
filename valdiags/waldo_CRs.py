# FREQUENTIST CONFIDENCE REGIONS for SBI-tasks using NEURAL POSTERIOR ESTIMATION (NPE)
# Code adapted from WALDO git-repo:
#   - HPD (Highest Predictive Density)
#   - WALDO (Calibrated CR based on the WALD statistic): quantile regression for critical value estimation.


import numpy as np
import torch
import torch.nn as nn

from itertools import chain
from functools import partial
from tqdm import tqdm


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


# Quantile regression code taken from https://colab.research.google.com/drive/1nXOlrmVHqCHiixqiMF6H8LSciz583_W2
# Copied from Waldo code


class q_model(nn.Module):
    def __init__(self, quantiles, neur_shapes, in_shape=1, dropout=0.5, seed=7):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.neur_shapes = neur_shapes
        self.in_shape = in_shape
        self.seed = seed
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.neur_shapes[0]),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
            nn.Linear(self.neur_shapes[0], self.neur_shapes[1]),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
        )
        final_layers = [
            nn.Linear(self.neur_shapes[1], 1) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)

    def init_weights(self):
        torch.manual_seed(self.seed)
        for m in chain(self.base_model, self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        tmp_ = self.base_model(x)
        return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class Learner:
    def __init__(self, model, optimizer_class, loss_func, device="cpu", seed=7):
        self.model = model.to(device)
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(device)
        self.device = device
        self.seed = seed
        self.loss_history = []

    def fit(self, x, y, epochs, batch_size):
        torch.manual_seed(self.seed)
        self.model.train()
        for e in tqdm(range(epochs)):
            shuffle_idx = np.arange(x.shape[0])
            np.random.shuffle(shuffle_idx)
            x = x[shuffle_idx]
            y = y[shuffle_idx]
            epoch_losses = []
            for idx in range(0, x.shape[0], batch_size):
                self.optimizer.zero_grad()
                batch_x = (
                    torch.from_numpy(x[idx : min(idx + batch_size, x.shape[0]), :])
                    .float()
                    .to(self.device)
                    .requires_grad_(False)
                )
                batch_y = (
                    torch.from_numpy(y[idx : min(idx + batch_size, y.shape[0])])
                    .float()
                    .to(self.device)
                    .requires_grad_(False)
                )
                preds = self.model(batch_x)
                loss = self.loss_func(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.cpu().detach().numpy())
            epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(epoch_loss)

    def predict(self, x, mc=False):
        if mc:
            self.model.train()
        else:
            self.model.eval()
        return (
            self.model(torch.from_numpy(x).to(self.device).requires_grad_(False))
            .cpu()
            .detach()
            .numpy()
        )


def train_qr_algo(
    dim,
    theta_mat,
    stats_mat,
    learner_kwargs,
    pytorch_kwargs,
    alpha,
    prediction_grid,
    nn_dropout=0.1,
):
    # Train the regression quantiles algorithms

    model = q_model([alpha], dropout=nn_dropout, in_shape=dim, **pytorch_kwargs)
    loss_func = QuantileLoss(quantiles=[alpha])
    learner = Learner(
        model,
        partial(torch.optim.Adam, weight_decay=1e-6),
        loss_func,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    learner.fit(theta_mat.reshape(-1, dim), stats_mat.reshape(-1), **learner_kwargs)
    pred_vec = learner.predict(prediction_grid.reshape(-1, dim).astype(np.float32))
    model = learner  # just for returning it

    return model, pred_vec
