from functools import partial
import submitit
import torch

from hnpe.misc import make_label

from tasks.jrnmm.inference import run_inference_sbi, run_inference_lampe
from tasks.jrnmm.posterior import build_flow, IdentityJRNMM, NPE_JRNMM_lampe_base
from tasks.jrnmm.summary import summary_JRNMM
from tasks.jrnmm.simulator import prior_JRNMM, simulator_JRNMM



# we train for one round : amortized
NB_ROUNDS = 1

# number of training samples
N_SIM = 50_000

# choose the naive or factorized flow
NAIVE = True

# list of (trec, nextra)
LIST_TREC_NEXTRA = [(8, 0)]  # , (8,9)] #[(2,3), (8,10)]   #(8,3), (2,10) #(8,4) (2,4)

# extra obs have all parameters in common with x_0 or only the global one (gain)
# during inference only
LIST_SINGLE_REC = [False]

# fixed gain
FIXED_GAIN = True

# lampe ou pas 
LAMPE = True

# target folder path inside results folder for saving
PATH = "saved_experiments/JR-NMM/fixed_gain_3d/Flows_amortized/lampe/randperm/epochs_until_conv_100/"


def get_executor_marg(job_name, timeout_hour=60, n_cpus=40):

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f"{timeout_hour}:00:00",
        slurm_additional_parameters={
            "ntasks": 1,
            "cpus-per-task": n_cpus,
            "distribution": "block:block",
        },
    )
    return executor


def setup_inference(t_rec, n_extra, single_rec, num_workers=20):

    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = n_extra
    # what kind of summary features to use
    meta_parameters["summary"] = "Fourier"

    # whether to do naive implementation
    meta_parameters["naive"] = NAIVE

    # which example case we are considering here
    meta_parameters["case"] = (
        PATH + "JRNMM_nextra_{:02}_trec_{}"
        "naive_{}_"
        "single_rec_{}".format(
            meta_parameters["n_extra"], t_rec, meta_parameters["naive"], single_rec
        )
    )

    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = NB_ROUNDS
    # number of simulations per round
    meta_parameters["n_sr"] = N_SIM
    # number of summary features to consider
    meta_parameters["n_sf"] = 33
    # how many seconds the simulations should have (fs = 128 Hz)
    meta_parameters["t_recording"] = t_rec
    meta_parameters["n_ss"] = int(128 * meta_parameters["t_recording"])
    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)
    # run example with the chosen parameters
    device = "cpu"

    # set prior distribution for the parameters
    input_parameters = ["C", "mu", "sigma", "gain"]
    prior = prior_JRNMM(
        parameters=[
            ("C", 10.0, 250.0),
            ("mu", 50.0, 500.0),
            ("sigma", 100.0, 5000.0),
            ("gain", -20.0, +20.0),
        ]
    )
    if FIXED_GAIN:
        input_parameters = ["C", "mu", "sigma"]
        prior = prior_JRNMM(
            parameters=[
                ("C", 10.0, 250.0),
                ("mu", 50.0, 500.0),
                ("sigma", 100.0, 5000.0),
            ]
        )

    # choose how to setup the simulator for training
    simulator = partial(
        simulator_JRNMM,
        input_parameters=input_parameters,
        t_recording=meta_parameters["t_recording"],
        n_extra=meta_parameters["n_extra"],
        p_gain=prior,
    )

    # choose how to get the summary features
    summary_extractor = summary_JRNMM(
        n_extra=meta_parameters["n_extra"],
        d_embedding=meta_parameters["n_sf"],
        n_time_samples=meta_parameters["n_ss"],
        type_embedding=meta_parameters["summary"],
    )

    # let's use the log power spectral density instead
    summary_extractor.embedding.net.logscale = True

    if not LAMPE:
        # choose a function which creates a neural network density estimator
        build_nn_posterior = partial(
            build_flow,
            embedding_net=IdentityJRNMM(),
            naive=meta_parameters["naive"],
            aggregate=True,
            z_score_theta=True,
            z_score_x=True,
            n_layers=10,
        )

        _ = run_inference_sbi(
            simulator=simulator,
            prior=prior,
            build_nn_posterior=build_nn_posterior,
            ground_truth=None,
            meta_parameters=meta_parameters,
            summary_extractor=summary_extractor,
            save_rounds=True,
            device=device,
            num_workers=num_workers,
            max_num_epochs=100000,
        )
    else:
        # train data
        dataset_train = torch.load(
            "/data/parietal/store/work/jlinhart/git_repos/valDiags_flows/saved_experiments/JR-NMM/fixed_gain_3d/datasets_train.pkl"
        )
        _ = run_inference_lampe(
            simulator=simulator,
            prior=prior,
            dataset_train=dataset_train,
            estimator=partial(NPE_JRNMM_lampe_base, randperm=True),
            meta_parameters=meta_parameters,
            summary_extractor=summary_extractor,
            save_rounds=True,
            # training_batch_size=100,
            # lr=5e-4,
            epochs_until_convergence=100,
        )


executor = get_executor_marg(f"work_inference")
# launch batches
with executor.batch():
    print("Submitting jobs...", end="", flush=True)
    tasks = []
    for single_rec in LIST_SINGLE_REC:
        for (t_rec, n_extra) in LIST_TREC_NEXTRA:
            kwargs = {"t_rec": t_rec, "n_extra": n_extra, "single_rec": single_rec}
            tasks.append(executor.submit(setup_inference, **kwargs))

