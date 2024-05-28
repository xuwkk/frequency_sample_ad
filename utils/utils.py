import numpy as np
import torch
import random
import matplotlib.pyplot as plt

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