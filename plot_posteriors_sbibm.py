import argparse
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from valdiags.graphical_diagnostics import multi_corner_plots

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="lc2st")
parser.add_argument(
    "--task",
    type=str,
    default="gaussian_mixture",
    choices=["two_moons", "slcp", "gaussian_mixture", "gaussian_linear_uniform"],
)
parser.add_argument(
    "--n_train", type=int, default=1000, choices=[100, 1000, 10000, 100000]
)
parser.add_argument(
    "--num_observation",
    "-no",
    type=int,
    nargs="+",
    default=1,
    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)
args = parser.parse_args()

PATH_EXPERIMENT = Path(f"saved_experiments/lc2st_2023/exp_2/{args.task}")

# Load data
if args.method == "lc2st_nf":
    label = r"$z$"

    ref_samples = torch.load(PATH_EXPERIMENT / "base_dist_samples_n_cal_10000.pkl")
    npe_samples = {}
    for n in args.num_observation:
        npe_samples[n] = torch.load(
            PATH_EXPERIMENT
            / f"npe_{args.n_train}"
            / "reference_inv_transform_samples_n_cal_10000.pkl"
        )[n]

else:
    label = r"$\theta$"

    ref_samples = {}
    npe_samples = {}
    for n in args.num_observation:
        ref_samples[n] = torch.load(
            PATH_EXPERIMENT / "reference_posterior_samples_n_eval_10000.pkl"
        )[n]
        npe_samples[n] = torch.load(
            PATH_EXPERIMENT / f"npe_{args.n_train}" / "npe_samples_obs_n_eval_10000.pkl"
        )[n]


dim_theta = ref_samples[args.num_observation[0]].shape[-1]
labels = [f"{label}" + rf"$_{i}$" for i in range(dim_theta)]


# plot posteriors
for n in args.num_observation:
    multi_corner_plots(
        [ref_samples[n], npe_samples[n]],
        ["Reference", "NPE"],
        ["blue", "orange"],
        labels=labels,
        title=f"{args.task} - npe_{args.n_train} - obs_{n}" + f"\n({args.method})",
    )
    plt.show()
