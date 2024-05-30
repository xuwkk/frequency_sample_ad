import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from model import (SimpleModel_omega_omegadot_feedback, 
                    AugementModel, SimpleModel_omega_q, 
                    SimpleModel_omega_omegadot)

def to_numpy(tensor):
    """
    convert a tensor to numpy
    """
    return tensor.detach().cpu().numpy()

def initialize_model(name, system_params, hyperparams):

    if name == "SimpleModel_omega_omegadot_feedback":
        cons_params = {
            "delta_P": system_params.delta_P,
            "tau": system_params.tau,
            "r": system_params.r,
            "M": system_params.M0,  # ! regard as constant parameter
            "D": system_params.D0,
        }

        model = SimpleModel_omega_omegadot_feedback(cons_params, device = hyperparams.device)

    elif name == "AugementModel":
        cons_params = {
            "delta_P": system_params.delta_P,
            "tau": system_params.tau,
            "r": system_params.r,
            "M": system_params.M0,  # ! regard as constant parameter
            "D": system_params.D0,
        }
        
        model = AugementModel(cons_params, device = hyperparams.device)
        
    elif name == "SimpleModel_omega_q":
        
        cons_params = {
        "delta_P": system_params.delta_P,
        "tau": system_params.tau,
        "r": system_params.r,
        }

        model = SimpleModel_omega_q(cons_params, device = hyperparams.device)

    elif name == "SimpleModel_omega_omegadot":

        cons_params = {
        "delta_P": system_params.delta_P,
        "tau": system_params.tau,
        "r": system_params.r,
        }

        model = SimpleModel_omega_omegadot(cons_params, device = hyperparams.device)
        
    return model

def get_K_threshold(system_params):
    return system_params.M0 **2 / (4 * system_params.delta_P)

def get_K(hyperparams, system_params):

    threshold = get_K_threshold(system_params)
    K = torch.randn((hyperparams.sample_no, 4), device = hyperparams.device) * hyperparams.k_range

    if system_params.delta_P > 0:
        K[:, 1] = torch.clamp(K[:, 1], -1e6, threshold * 0.8) # ! to avoid the rounding error
    else:
        K[:, 1] = torch.clamp(K[:, 1], threshold * 0.8, 1e6)
    
    return K

def plot(omega, d_omega, title, cfg):

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']

    linestep = np.linspace(hyperparams["t0"], hyperparams["t1"], 
                        int((hyperparams["t1"] - hyperparams["t0"]) / hyperparams["dt"]))
    
    base = system_params.base
    ss_max = base + system_params.max_ss_dev
    ss_min = base - system_params.max_ss_dev
    rocof_max = system_params.max_rocof
    rocof_min = -system_params.max_rocof
    nadir_max = base + system_params.max_nadir_dev
    nadir_min = base - system_params.max_nadir_dev

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(linestep, (1+omega) * base)
    axs[0].set_title("omega plot")
    axs[0].hlines(y = ss_max, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = 'r', label = "Steady state constraint")
    axs[0].hlines(y = ss_min, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = 'r')
    axs[0].hlines(y = nadir_max, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = 'g', label = "Nadir constraint")
    axs[0].hlines(y = nadir_min, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = 'g')
    axs[0].grid()
    axs[0].legend()
    
    axs[1].plot(linestep, d_omega * base)
    axs[1].set_title("d_omega plot")
    axs[1].hlines(y = rocof_max, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = 'r', label = "RoCoF constraint")
    axs[1].hlines(y = rocof_min, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = 'r')
    axs[1].legend()
    axs[1].grid()
    plt.savefig(f"{title}.pdf")

def set_random_seed(seed = 100):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False