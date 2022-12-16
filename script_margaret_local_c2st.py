import submitit

import numpy as np
import pandas as pd
import torch
import torch.distributions as D
from sbi.utils.metrics import c2st

from data.data_generators import ConditionalGaussian1d, SBIGaussian2d
from data.feature_transforms import first_dim_only

# Simulator and flows (trained on 10_000 samples)
data_gen = ConditionalGaussian1d()
# maf_good = torch.load('saved_experiments/Gaussian1d_localPIT/maf_good_layers10.pkl')
# maf_bad = torch.load('saved_experiments/Gaussian1d_localPIT/maf_bad_layers10.pkl')

# Data dimensions
DIM = 1 # target data
N_LIST = [1000, 2000, 5000, 10000] # calibration dataset size

# Reference samples (base distribution)
P = D.MultivariateNormal(loc=torch.zeros(DIM), covariance_matrix=torch.eye(DIM))

# # Calibration dataset (used to compute transformed flow samples)
# x_cal, theta_cal = data_gen.get_joint_data(n=N)

# Flow transform 
# flow_values_train_good = maf_good._transform(theta_cal, context=x_cal)[0]
# flow_values_train_bad = maf_bad._transform(theta_cal, context=first_dim_only(x_cal))[0]

# # Observation x_0
# x_0 = torch.FloatTensor([[0, 1]])

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

def eval_classifier_for_lc2st(shifts, shift_object = 'mean', clfs=['rf', 'mlp'], nb_samples = 1000, n_trials=10):
    
    x_samples, _ = data_gen.get_joint_data(n=nb_samples)
    P_samples = P.rsample((nb_samples,))
    P_joint = torch.cat([P_samples,x_samples], dim=1)
    clf_method = []
    shift = []
    scores = []
    if shift_object == 'mean':
        for clf in clfs:
            for m in shifts:
                for _ in range(n_trials):
                    Q = D.MultivariateNormal(loc=torch.FloatTensor([m]*DIM), covariance_matrix=torch.eye(DIM)).rsample((nb_samples,))
                    Q_joint = torch.cat([Q,x_samples], axis=1)
                    score = c2st(P_joint, Q_joint,classifier=clf).item()
                    clf_method.append(clf)
                    shift.append(m)
                    scores.append(score)
    else:
        for clf in clfs:
            for s in shifts:
                for _ in range(n_trials):
                    Q = D.MultivariateNormal(loc=torch.zeros(DIM), covariance_matrix=torch.eye(DIM)*s).rsample((n,))
                    Q_joint = torch.cat([Q,x_samples], axis=1)
                    score = c2st(P_joint, Q_joint,classifier=clf).item()
                    clf_method.append(clf)
                    shift.append(s)
                    scores.append(score)
    df = pd.DataFrame({f'{shift_object}_shift': shift, 'score': scores, 'classifier':clf_method,})
    filename = f'saved_experiments/Gaussian1d_localPIT/lc2st_eval_clfs/df_{shift_object}.pkl'
    torch.save(df, filename)

# def score_lc2st(flow, x_obs, nb_samples):
    
executor = get_executor_marg(f"eval_lc2st_clfs")
# launch batches
with executor.batch():
    print("Submitting jobs...", end="", flush=True)
    tasks = []
    for n in N_LIST:
        for shifts, s_object in zip([np.linspace(0,5,20),'mean'], [np.linspace(0.01,5,20),'scale']):
            kwargs = {
                "shifts": shifts,
                "shift_object": s_object,
                "nb_samples": n,
            }
            tasks.append(executor.submit(eval_classifier_for_lc2st, **kwargs))


