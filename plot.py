from turtle import color
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch.distributions as D

def plot_pdf_1D(target_dist, x_samples, x_i, x_f, flow=None):
    eval_x = torch.linspace(x_i, x_f).reshape(-1,1)

    fig = plt.figure(figsize=(6, 2))
    plt.plot(x_samples, np.zeros_like(x_samples), 'bx', alpha=0.5, markerfacecolor='none', markersize=6)
    
    try:
        p_x_true = torch.exp(target_dist.log_prob(eval_x))
        plt.plot(eval_x.numpy(), p_x_true.detach().numpy(),'--', color='blue')
    except ValueError:  # in case of exponential distribution 
        eval_x_pos = torch.linspace(0.01, x_f).reshape(-1,1)
        p_x_true = torch.exp(target_dist.log_prob(eval_x_pos))
        plt.plot(eval_x_pos.numpy(), p_x_true.detach().numpy(),'--', color='blue')

    if flow is not None:
        p_x_learned = torch.exp(flow.log_prob(eval_x))
        plt.plot(eval_x.numpy(), p_x_learned.detach().numpy(), color='orange')

    plt.legend(["Samples", "True", "Learned"], loc="upper right")

    _ = plt.xlim([x_i, x_f]); _ = plt.ylim([-0.12, 3.2])

    plt.show()

def plot_cdf_1D(target_dist, flow, x_i, x_f, base_dist=D.Normal(0,1)):
    x_eval = torch.linspace(x_i,x_f).reshape(-1,1)
    
    try:
        cdf_estimate = cdf_flow(x_eval, flow, base_dist)
        plt.plot(x_eval,target_dist.cdf(x_eval), color='blue')
    except ValueError:  # in case of exponential distribution 
        x_eval_pos = torch.linspace(0.01, x_f).reshape(-1,1)
        cdf_estimate = cdf_flow(x_eval_pos, flow, base_dist)
        plt.plot(x_eval,target_dist.cdf(x_eval_pos), color='blue')

    plt.plot(x_eval, cdf_estimate.detach().numpy(), color='orange')
    plt.legend(["True", "Learned"], loc="upper left")
    plt.show()

def PP_plot_1D(target_dist, x_samples, flow, base_dist=D.Normal(0,1)):
    alphas = np.linspace(0,1)
    cdf = lambda x: cdf_flow(x, flow, base_dist)
    z_true = PP_vals(target_dist.cdf, x_samples, alphas)
    z_estimate = PP_vals(cdf, x_samples, alphas)
    plt.plot(alphas, z_true, color='blue')
    plt.plot(alphas, z_estimate, color='orange')
    plt.legend(["True", "Learned"], loc="upper left")
    plt.show()



def PP_vals(cdf, x_samples, alphas):
    """Return estimated vs true quantiles between pdist and samples.
    inputs:
    - cdf: cdf of the estimate (flow) 
    - x_samples: torch.tensor of samples of the target distribution
    - alphas: array of values to evaluate the PP-vals 
    """
    F = cdf(x_samples).detach().numpy()
    z = [np.mean(F < alpha)  for alpha in alphas]
    return z


    
def cdf_flow(x, flow, base_dist=D.Normal(0,1)):
    """ Return the cdf of the flow evaluated in x
    input:
    - x: torch.tensor
    - flow: nflows.Flow 
    - base_dist: torch.distributions object
        """
    return base_dist.cdf(flow._transform(x)[0])