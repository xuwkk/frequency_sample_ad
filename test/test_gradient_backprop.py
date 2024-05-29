"""
test with the backward propagation
"""

import sys
sys.path.append('.')
from model import AugementModel
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_random_seed, initialize_model
from ode_solver import solve
import time
from torch.autograd.functional import jacobian
from functools import partial

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    set_random_seed(100)

    torch.set_default_dtype(torch.float64)

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']

    model_name = "SimpleModel_omega_omegadot_feedback"

    model = initialize_model(model_name, system_params, hyperparams)
    
    K = torch.randn(hyperparams.batch_size, 4).to(hyperparams.device) * 20
    
    diff_params = {
        "K": K,
    }

    watched_idx = 0

    """
    backward propagation
    """


    K.requires_grad = True

    for i in range(hyperparams.max_iter):
        with torch.enable_grad():
            print('iteration:', i)

            initial_state = model.get_initial_state(K)
            
            backward_solution_time = time.time()
            output_1 = solve(model, **hyperparams, **diff_params, y0 = initial_state)
            print('backward_solution_time:', time.time() - backward_solution_time)

            backward_backward_time = time.time()    
            freq_ss_1 = torch.abs(output_1[watched_idx][-1, 0])
            freq_ss_1.backward(retain_graph=True)
            grad_freq_ss_1 = K.grad.clone()[watched_idx]
            K.grad.zero_()
            
            freq_nadir_1 = torch.max(torch.abs(output_1[watched_idx][:, 0]))
            freq_nadir_1.backward(retain_graph=True)
            grad_freq_nadir_1 = K.grad.clone()[watched_idx]
            K.grad.zero_()

            freq_rocof_1 = torch.max(torch.abs(output_1[watched_idx][:, 1]))
            freq_rocof_1.backward()
            grad_freq_rocof_1 = K.grad.clone()[watched_idx]
            K.grad.zero_()

            print('backward_backward_time:', time.time() - backward_backward_time)

            print('freq_ss_1:', freq_ss_1)
            print('freq_nadir_1:', freq_nadir_1)
            print('freq_rocof_1:', freq_rocof_1)
            print('grad_freq_ss_1:', grad_freq_ss_1)
            print('grad_freq_nadir_1:', grad_freq_nadir_1)
            print('grad_freq_rocof_1:', grad_freq_rocof_1)

        # try to increase the rocof
        K.data[watched_idx] = K.data[watched_idx] + grad_freq_rocof_1 * 1e5


if __name__ == "__main__":
    main()