import torch

from tqdm import tqdm

from .npe_utils import inv_flow_transform_obs, sample_from_npe_obs


def generate_task_data(
    n_samples,
    task,
    num_observation_list,
    sample_from_joint=True,
    sample_from_reference=True,
):
    """Generate data for a given task.
    This data is fixed and independent of the used sbi-algorithm.

    Args:
        n_samples (int): Number of samples to generate
        task (str): sbibm task name
        num_observation_list: List of observation numbers for which we want to sample
            from the reference posterior.
        seed: Seed

    """

    # Get simulator and prior
    simulator = task.get_simulator()
    prior = task.get_prior()

    # Generate data from reference posterior
    if sample_from_reference:
        print("Samples from reference posterior:")
        reference_posterior_samples = {}
        for num_observation in num_observation_list:
            reference_posterior_samples[
                num_observation
            ] = task._sample_reference_posterior(
                num_samples=n_samples,
                num_observation=num_observation,
            )

    else:
        reference_posterior_samples = None

    # Generate data from joint
    if sample_from_joint:
        theta = prior(num_samples=n_samples)
        x = simulator(theta)
    else:
        theta = None
        x = None

    return reference_posterior_samples, theta, x


def generate_npe_data_for_c2st(
    npe,
    base_dist_samples,
    reference_posterior_samples,
    observation_list,
    nf_case=True,
):
    """Generate data for a given task and npe-flow that is used in the C2ST(-NF) methods.
            - sample from the npe at given observations (using the forward flow transformation)
                (C2ST compares them to the reference posterior samples),
            - compute inverse npe-flow-transformation on reference posterior samples
                (C2ST-NF compares them to the samples from the base distribution (normal))
        This data is dependent on the flow defining the npe.

    Args:
        npe (sbi.DirectPosterior): neural posterior estimator (normalizing flow).
        base_dist_samples (torch.Tensor): samples from the base distribution of the
            flow. This is used to generate flow samples.
        reference_posterior_samples (dict): dict of samples from the reference posterior
            for the considered observations. The dict keys are the observation numbers.
            The dict values are torch tensors of shape (n_samples, dim)
        observation_list (list): list of observations the reference posterior samples correspond to.
            observation = task.get_observation(num_observation)
    """
    npe_samples_obs = {}
    reference_inv_transform_samples = {}
    for i, observation in tqdm(
        enumerate(observation_list),
        desc="Computing npe-dependant samples for every observation x_0",
    ):
        # Set default x_0 for npe
        npe.set_default_x(observation)
        # Sample from npe at x_0
        npe_samples_obs[i + 1] = sample_from_npe_obs(
            npe=npe, observation=observation, base_dist_samples=base_dist_samples
        )
        # Compute inverse flow transformation of npe on reference posterior samples at x_0
        if nf_case and reference_posterior_samples is not None:
            reference_inv_transform_samples[i + 1] = inv_flow_transform_obs(
                reference_posterior_samples[i + 1],
                observation,
                npe.posterior_estimator,
            )
        else:
            reference_inv_transform_samples[i + 1] = None
    return npe_samples_obs, reference_inv_transform_samples


def generate_npe_data_for_lc2st(
    npe,
    base_dist_samples,
    joint_samples,
    nf_case=True,
):
    """Generate data for a given task and npe that is used in the LC2ST(-NF) methods.
        - sample from the npe for every observation x in `joint_samples` (using the forward-flow transformation).
            (LC2ST compares them to the joint samples [theta, x]),
        - compute inverse npe-flow-transformation on the joint samples
            (LC2ST-NF compares them to the samples from the base distribution (z),
            concatenated with x: [z,x])
    This data is dependent on the flow.

    Args:
        npe (sbi.DirectPosterior): neural posterior estimator.
        joint_samples (dict[torch.Tensor]): dict of samples from the joint distribution of the
            flow: {'theta':theta, 'x':x} where
                - theta is a torch.Tensor of shape (n_samples, dim)
                - x is a torch.Tensor of shape (n_samples, dim_x)
        base_dist_samples (torch.Tensor): samples from the base distribution of the
            npe-flow. This is used to generate flow samples.
            of shape (n_samples, dim)
    """
    npe_samples_joint = []
    inv_transform_samples_joint = []
    for theta, x, z in tqdm(
        zip(joint_samples["theta"], joint_samples["x"], base_dist_samples),
        desc=f"Computing npe-dependant samples for every x in joint dataset",
    ):
        x, theta, z = x[None, :], theta[None, :], z[None, :]

        # Sample from flow
        npe_samples_joint.append(
            sample_from_npe_obs(npe=npe, observation=x, base_dist_samples=z)
        )
        # Compute inverse flow transformation of flow on joint samples
        if nf_case:
            # Set default x for npe
            npe.set_default_x(x)
            # compute inverse flow transformation of flowon (theta, x)
            inv_transform_samples_joint.append(
                inv_flow_transform_obs(
                    theta,
                    x,
                    npe.posterior_estimator,
                )
            )
    npe_samples_joint = torch.stack(npe_samples_joint)[:, 0, :]
    if nf_case:
        inv_transform_samples_joint = torch.stack(inv_transform_samples_joint)[:, 0, :]
    else:
        inv_transform_samples_joint = None

    return npe_samples_joint, inv_transform_samples_joint
