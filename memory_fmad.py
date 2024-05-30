"""
return the memory usage
"""

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_random_seed, initialize_model, get_K
from ode_solver import solve
import time
import os, psutil
from model import fmad_method, finite_method, unrolling_method

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):

    # prepare the data
    set_random_seed(cfg.hyperparams.random_seed)

    torch.set_default_dtype(torch.float64)

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']

    K_all = get_K(hyperparams, system_params)
    batch_size = hyperparams.batch_size
    no_batch = len(K_all) // batch_size

    # algorithms

    for i in range(no_batch):
        print('batch:', i)

        K = K_all[i * batch_size : (i + 1) * batch_size]

        memory_fmad = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        model_fmad = initialize_model("AugementModel", system_params, hyperparams)
        
        time_fmad = time.time()
        ss_fmad, grad_ss_fmad, nadir_fmad, grad_nadir_fmad, rocof_fmad, grad_rocof_fmad = fmad_method(
            model = model_fmad, K = K, solve = solve, cfg = cfg)
        print('time_fmad:', time.time() - time_fmad)
        print('memory fmad:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2 - memory_fmad)

if __name__ == "__main__":
    main()