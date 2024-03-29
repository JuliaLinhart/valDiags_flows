from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sbi import inference as sbi_inference
from sbi.utils import get_log_root

from sbi.utils.sbiutils import standardizing_net

def summary_plcr(prefix):
    logdir = Path(
        get_log_root(),
        prefix,
        datetime.now().isoformat().replace(":", "_"),
    )
    return SummaryWriter(logdir)


def run_inference(simulator, prior, build_nn_posterior, ground_truth,
                  meta_parameters, summary_extractor=None, save_rounds=False,
                  seed=42, device="cpu", num_workers=1, max_num_epochs=None,
                  stop_after_epochs=20, training_batch_size=100, build_aggregate_before=None): 

    # set seed for numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make a SBI-wrapper on the simulator object for compatibility
    simulator, prior = sbi_inference.prepare_for_sbi(simulator, prior)

    if save_rounds:
        # save the ground truth and the parameters
        folderpath = Path.cwd() / "results" / meta_parameters["label"]
        print(folderpath)
        folderpath.mkdir(exist_ok=True, parents=True)
        path = folderpath / "ground_truth.pkl"
        torch.save(ground_truth, path)
        path = folderpath / "parameters.pkl"
        torch.save(meta_parameters, path)

    # setup the inference procedure
    inference = sbi_inference.SNPE(
        prior=prior,
        density_estimator=build_nn_posterior,
        show_progress_bars=True,
        device=device,
        summary_writer=summary_plcr(meta_parameters["label"])
    )

    # loop over rounds
    posteriors = []
    proposal = prior
    if ground_truth is not None:
        ground_truth_obs = ground_truth["observation"]
    
    for round_ in range(meta_parameters["n_rd"]):
        
        # simulate the necessary data
        theta, x = sbi_inference.simulate_for_sbi(
            simulator, proposal, num_simulations=meta_parameters["n_sr"],
            num_workers=num_workers,
        )

        if 'cuda' in device:
            torch.cuda.empty_cache()

        # extract summary features
        if summary_extractor is not None:
            x = summary_extractor(x)

        if (x[0].shape[0] == 3) and meta_parameters["norm_before"]:
            print('norm_before')
            x0_n = torch.cat([x[:,0].reshape(-1,1), x[:,2].reshape(-1,1)],dim=1)
            stand = standardizing_net(x0_n)
            x0_n = stand(x0_n)

            x[:,0] = x0_n[:,0]
            x[:,2] = x0_n[:,1]

            path = folderpath / f"stand_net_round_{round_:02}.pkl"
            torch.save(stand.state_dict(), path)
        
        ## ------- added --------- ##
        # standardize data wrt x and aggregate extra observations
        if build_aggregate_before is not None:
            aggregate_before = build_aggregate_before(x_ref=x) # standardize data wrt x
            x = aggregate_before(x)
            ground_truth_obs = aggregate_before(ground_truth["observation"])
        else:
            aggregate_before = None
        ## ----------------------- ##
        # train the neural posterior with the loaded data
        nn_posterior = inference.append_simulations(theta, x).train(
            num_atoms=10,
            training_batch_size=training_batch_size,
            use_combined_loss=True,
            discard_prior_samples=True,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True
        )
        nn_posterior.zero_grad()
        posterior = inference.build_posterior(nn_posterior)
        posteriors.append(posterior)
        # save the parameters of the neural posterior for this round
        if save_rounds:
            path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
            posterior.net.save_state(path)
            ## --------------- added -------------- ##
            # save aggregate net parameters: mean and std based on training simulations 
            if aggregate_before is not None:
                path = folderpath / f"norm_agg_before_net_round_{round_:02}.pkl"
                torch.save(aggregate_before.state_dict(), path)
            ## ------------------------------------ ##

        if meta_parameters['n_rd'] > 1:
            # set the proposal prior for the next round
            proposal = posterior.set_default_x(ground_truth_obs)

    return posteriors
