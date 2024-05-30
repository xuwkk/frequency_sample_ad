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

        # finite difference
        memory_finite = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        model_finite = initialize_model('SimpleModel_omega_omegadot_feedback', system_params, hyperparams)

        time_finite = time.time()
        ss_finite, grad_ss_finite, nadir_finite, grad_nadir_finite, rocof_finite, grad_rocof_finite = finite_method(
            model = model_finite, K = K, solve = solve, cfg = cfg)
        print('time_finite:', time.time() - time_finite)
        print('memory finite:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2 - memory_finite)
        

if __name__ == "__main__":
    main()
    