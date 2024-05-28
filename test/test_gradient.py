"""
test with the backward propagation
"""

import sys
sys.path.append('.')
from model import SimpleModel_omega_omegadot_feedback, AugementModel
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.autograd.forward_ad as fwAD
from torch.autograd.functional import jacobian
from torch.func import jvp
from functools import partial
from utils import set_random_seed
from ode_solver import solve
import time

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    set_random_seed(100)

    print(OmegaConf.to_yaml(cfg))

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']
    """
    the omega-omegadot model with state feedback
    """

    cons_params = {
        "delta_P": system_params.delta_P,
        "tau": system_params.tau,
        "r": system_params.r,
        "M": system_params.M0,  # ! regard as constant parameter
        "D": system_params.D0,
    }

    K = torch.randn(1, 4) * 20

    diff_params = {
        "K": K,
    }

    """
    backward propagation
    """
    model = SimpleModel_omega_omegadot_feedback(cons_params)
    initial_state = model.get_initial_state(K)
    
    K.requires_grad = True

    
    with torch.enable_grad():
        backward_solution_time = time.time()
        output_1 = solve(model, **hyperparams, **diff_params, y0 = initial_state)
        print('backward_solution_time:', time.time() - backward_solution_time)

        backward_backward_time = time.time()    
        freq_ss = system_params.base * (1 + output_1[0][-1, 0])
        freq_ss.backward(retain_graph=True)
        grad_freq_ss = K.grad.clone()
        K.grad.zero_()
        
        freq_nadir = system_params.base * (1 + torch.max(torch.abs(output_1[0][:, 0])))
        freq_nadir.backward(retain_graph=True)
        grad_freq_nadir = K.grad.clone()
        K.grad.zero_()

        freq_rocof = system_params.base * torch.max(torch.abs(output_1[0][:, 1]))
        freq_rocof.backward()
        grad_freq_rocof = K.grad.clone()
        K.grad.zero_()

        print('backward_backward_time:', time.time() - backward_backward_time)
    
    print('output_1:', output_1.shape)
    print('freq_ss:', freq_ss)
    print('grad_freq_ss:', grad_freq_ss)
    print('freq_nadir:', freq_nadir)
    print('grad_freq_nadir:', grad_freq_nadir)
    print('freq_rocof:', freq_rocof)
    print('grad_freq_rocof:', grad_freq_rocof)


    print("====================================")
    """
    fmad
    """
    model = AugementModel(cons_params)
    initial_state = model.get_initial_state(K.requires_grad_(False))

    time_fmad = time.time()
    with torch.no_grad():
        output_2 = solve(model, **hyperparams, **diff_params, y0 = initial_state) # (batch_size, time_steps, state_dim)
    print('time_fmad_all:', time.time() - time_fmad)

    freq_ss = system_params.base * (1 + output_2[0][-1, 0])
    grad_freq_ss = output_2[0][-1].reshape(-1,2)[1:,0] * system_params.base

    nadir_idx = torch.argmax(torch.abs(output_2[0][:, 0]))
    freq_nadir = system_params.base * (1 + torch.abs(output_2[0][nadir_idx, 0]))
    grad_freq_nadir = output_2[0][nadir_idx].reshape(-1,2)[1:,0] * system_params.base

    rocof_idx = torch.argmax(torch.abs(output_2[0][:, 1]))
    freq_rocof = system_params.base * torch.abs(output_2[0][rocof_idx, 1])
    grad_freq_rocof = output_2[0][rocof_idx].reshape(-1,2)[1:,1] * system_params.base

    

    print('freq_ss:', freq_ss)
    print('grad_freq_ss:', grad_freq_ss)
    print('freq_nadir:', freq_nadir)
    print('grad_freq_nadir:', grad_freq_nadir)
    print('freq_rocof:', freq_rocof)
    print('grad_freq_rocof:', grad_freq_rocof)

if __name__ == "__main__":
    main()