import torch
import numpy as np

from valdiags.localC2ST import eval_lc2st


## Sampling


def rejection_sampling(proposal_sampler, f, num_samples=10000, approximate=False):
    """Sample from proposal_pdf*f via rejection sampling.
    (f only needs to be defined upto a multiplication constant)

    Example: To sample from a distribution p we have two options:
    1. proposal = uniform, f = p (because pdf_unifrom is a constant)
    2. proposal = q, f = p/q
    """
    acc_rej_samples = []
    nb_acc_samples = 0
    while nb_acc_samples < num_samples:
        proposal_samples = proposal_sampler(num_samples)

        f_values = f(proposal_samples)
        if f_values.max() > 1:
            print(f"WARNING: some values are > 1.")
            if approximate:
                print("Approximate Rejection Sampling with weight clipping at 1.")
                f_values = f_values * (f_values <= 1)
            else:
                print(f"Deviding by max_f = {f_values.max()}")
                f_values /= f_values.max()
        u_rand = torch.rand(f_values.shape).numpy()
        acc_rej_samples.append(proposal_samples[f_values > u_rand])
        nb_acc_samples += len(acc_rej_samples[-1])
        print(f"NB of accepted samples: {len(acc_rej_samples[-1])}")

    acc_rej_samples = np.concatenate(acc_rej_samples, axis=0)
    print(f"Total NB of accepted samples: {nb_acc_samples}")
    return acc_rej_samples


## functions f


def clf_ratio_obs(batch, x_obs, clfs, inv_flow_transform=None):
    if inv_flow_transform is not None:
        batch = inv_flow_transform(batch)
    proba = np.mean(
        [eval_lc2st(batch.numpy(), x_obs, clf=clf) for clf in clfs],
        axis=0,
    )
    return (1 - proba) / proba


def corrected_pdf_old(
    batch,
    dist,
    x_obs,
    clfs,
    inv_flow_transform=None,
):
    dist_pdf = dist.log_prob(batch).exp().detach().numpy()

    if inv_flow_transform is not None:
        batch = inv_flow_transform(batch)

    ratio = clf_ratio_obs(batch, x_obs, clfs)
    return ratio * dist_pdf


def corrected_pdf(batch, dist, x_obs, clfs, inv_flow_transform=None, z_space=True):
    if inv_flow_transform is not None:
        dist_pdf = dist.log_prob(inv_flow_transform(batch)).exp().detach().numpy()
        if z_space:
            ratio = clf_ratio_obs(inv_flow_transform(batch), x_obs, clfs)
        else:
            ratio = clf_ratio_obs(batch, x_obs, clfs)
    else:
        dist_pdf = dist.log_prob(batch).exp().detach().numpy()
        ratio = clf_ratio_obs(batch, x_obs, clfs)
    return ratio * dist_pdf


## samplers


def flow_sampler(num_samples, base_dist_sampler, flow_transform):
    zs = base_dist_sampler(num_samples)
    return flow_transform(zs)


def uniform_sampler(num_samples, dim, low=-5, high=5, flow_transform=None):
    us = torch.distributions.Uniform(
        torch.ones(dim) * low, torch.ones(dim) * high
    ).sample((num_samples,))
    if flow_transform is not None:
        us = flow_transform(us)
    return us
