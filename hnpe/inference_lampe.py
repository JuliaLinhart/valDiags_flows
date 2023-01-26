from pathlib import Path

import numpy as np
import torch

import sys
import os
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from nde.train import train_lampe_npe


def run_inference_lampe(
    simulator,
    prior,
    estimator,
    meta_parameters,
    ground_truth=None,
    summary_extractor=None,
    save_rounds=False,
    seed=42,
    device="cpu",
    max_num_epochs=10_000,
    training_batch_size=100,
    dataset_train = None, 
):

    # set seed for numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    if save_rounds:
        # save the ground truth and the parameters
        folderpath = Path.cwd() / meta_parameters["label"]
        folderpath.mkdir(exist_ok=True, parents=True)
        if ground_truth is not None:
            path = folderpath / "ground_truth.pkl"
            torch.save(ground_truth, path)
        path = folderpath / "parameters.pkl"
        torch.save(meta_parameters, path)

    # loop over rounds
    posteriors = []
    proposal = prior
    if ground_truth is not None:
        ground_truth_obs = ground_truth["x"]

    for round_ in range(meta_parameters["n_rd"]):

        # simulate / load the necessary data
        if dataset_train is not None:
            # load the necessary data
            theta, x = dataset_train['theta'], dataset_train['x']
        else:
            # simulate the necessary data
            theta = proposal.sample((meta_parameters["n_sr"],))
            x = simulator(theta)
            

            # extract summary features
            if summary_extractor is not None:
                x = summary_extractor(x)
            

        # define inference object
        inference = estimator(theta, x)

        # train the neural posterior with the loaded data
        _, epochs = train_lampe_npe(
            inference,
            theta,
            x,
            num_epochs=max_num_epochs,
            batch_size=training_batch_size,
            lr= 5e-4, # default learning rate from sbi training function
            clip=5.0, # default clip from sbi training function 
            optimizer=torch.optim.AdamW,
            validation=True,
            epochs_until_converge=20,
        )
        print(f'inference done in {epochs} epochs')

        inference.eval()
        posteriors.append(inference)
        # save the parameters of the neural posterior for this round
        if save_rounds:
            path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
            torch.save(inference, path)
            print('saved')

        # define proposal for next round
        if meta_parameters["n_rd"] > 1:
            assert ground_truth is not None
            # set the proposal prior for the next round
            proposal = inference.flow(ground_truth_obs)

    return posteriors


if __name__ == "__main__":
    from simulator import prior_JRNMM, simulator_JRNMM
    from summary import summary_JRNMM
    from functools import partial
    from misc import make_label
    from posterior import NPE_JRNMM_lampe_base

    PATH_EXPERIMENT = "saved_experiments/JR-NMM/fixed_gain_3d/"
    N_EXTRA = 0

    meta_parameters = {}
    # Data features
    meta_parameters["t_recording"] = 8
    meta_parameters["n_extra"] = N_EXTRA
    # Summary Features
    meta_parameters["summary"] = "Fourier"
    meta_parameters["n_sf"] = 33
    # Training Features
    meta_parameters["n_rd"] = 1  # amortized flow
    meta_parameters["n_sr"] = 50_000  # simulations per round

    # example cases we are considering here
    meta_parameters["case"] = (
        PATH_EXPERIMENT + "test_lampe/JRNMM_nextra_{:02}_"
        "naive_{}_"
        "single_rec_{}".format(N_EXTRA, True, False)
    )

    # label for saving directory of experiments
    meta_parameters["label"] = make_label(meta_parameters)
    folderpath = Path.cwd() / meta_parameters["label"]


    # Prior
    prior = prior_JRNMM(
        parameters=[("C", 10.0, 250.0), ("mu", 50.0, 500.0), ("sigma", 100.0, 5000.0)]
    )

    # Simulator
    simulator = partial(
        simulator_JRNMM,
        input_parameters=["C", "mu", "sigma"],
        t_recording=meta_parameters["t_recording"],
        n_extra=N_EXTRA,
        p_gain=prior,
    )

    summary_extractor = summary_JRNMM(
        n_extra=N_EXTRA,
        d_embedding=meta_parameters["n_sf"],
        n_time_samples=int(128 * meta_parameters["t_recording"]),
        type_embedding=meta_parameters["summary"],
    )

    summary_extractor.embedding.net.logscale = True  # log-PSD

    # train data 
    dataset_train = torch.load('/data/parietal/store/work/jlinhart/git_repos/valDiags_flows/saved_experiments/JR-NMM/fixed_gain_3d/datasets_train.pkl')


    # ground truth for rounds > 1
    gt_theta = prior.sample((1,))
    gt_x = summary_extractor(simulator(gt_theta))[0]
    print(gt_x.shape)
    ground_truth = {'theta': gt_theta, 'x':gt_x}

    _ = run_inference_lampe(
        simulator,
        prior,
        dataset_train=dataset_train,
        estimator=NPE_JRNMM_lampe_base,
        meta_parameters=meta_parameters,
        ground_truth=ground_truth,
        summary_extractor=summary_extractor,
        save_rounds=True,
        training_batch_size=100,
    )

