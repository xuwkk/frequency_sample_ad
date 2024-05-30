"""
compare the solution, gradient, time, and memory usage of the three methods
1. fmad
2. unrolling
3. finite difference
"""

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_random_seed, initialize_model, get_K
from ode_solver import solve
import time
from torch.func import vmap, grad
from functools import partial
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

    precision = {'atol': 1e-14, 'rtol': 1e-14}

    # algorithms

    for i in range(no_batch):

        print('batch:', i)

        K = K_all[i * batch_size : (i + 1) * batch_size]

        # unrolling
        memory_unrolling = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        
        # this is only for evaluation the memory usage
        model_unrolling = initialize_model('SimpleModel_omega_omegadot_feedback', system_params, hyperparams)

        time_unrolling = time.time()
        ss_unroll, grad_ss_unroll, nadir_unroll, grad_nadir_unroll, rocof_unroll, grad_rocof_unroll = unrolling_method(
            model = model_unrolling, K = K, solve = solve, cfg = cfg)
        print('time_unrolling:', time.time() - time_unrolling)
        print('memory unrolling:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2 - memory_unrolling)

        # fmad
        
        memory_fmad = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        model_fmad = initialize_model("AugementModel", system_params, hyperparams)
        
        time_fmad = time.time()
        ss_fmad, grad_ss_fmad, nadir_fmad, grad_nadir_fmad, rocof_fmad, grad_rocof_fmad = fmad_method(
            model = model_fmad, K = K, solve = solve, cfg = cfg)
        print('time_fmad:', time.time() - time_fmad)
        print('memory fmad:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2 - memory_fmad)

        # finite difference
        memory_finite = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        model_finite = initialize_model('SimpleModel_omega_omegadot_feedback', system_params, hyperparams)

        time_finite = time.time()
        ss_finite, grad_ss_finite, nadir_finite, grad_nadir_finite, rocof_finite, grad_rocof_finite = finite_method(
            model = model_finite, K = K, solve = solve, cfg = cfg)
        print('time_finite:', time.time() - time_finite)
        print('memory finite:', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2 - memory_finite)

        assert torch.allclose(ss_unroll, ss_fmad, **precision)
        assert torch.allclose(ss_unroll, ss_finite, **precision)
        assert torch.allclose(nadir_unroll, nadir_fmad, **precision)
        assert torch.allclose(nadir_unroll, nadir_finite, **precision)
        assert torch.allclose(rocof_unroll, rocof_fmad, **precision)
        assert torch.allclose(rocof_unroll, rocof_finite, **precision)

        assert torch.allclose(grad_ss_unroll, grad_ss_fmad, **precision)
        assert torch.allclose(grad_nadir_unroll, grad_nadir_fmad, **precision)
        assert torch.allclose(grad_rocof_unroll, grad_rocof_fmad, **precision)

        print('max finite error ss:', torch.max(torch.abs(grad_ss_unroll - grad_ss_finite) / torch.abs(grad_ss_unroll) ))
        print('max finite error nadir:', torch.max(torch.abs(grad_nadir_unroll - grad_nadir_finite) / torch.abs(grad_nadir_unroll) ))
        print('max finite error rocof:', torch.max(torch.abs(grad_rocof_unroll[:, [1,3]] - grad_rocof_finite[:, [1,3]]) / torch.abs(grad_rocof_unroll[:, [1,3]]) ))

        
if __name__ == "__main__":
    main()