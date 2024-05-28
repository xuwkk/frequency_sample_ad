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

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    set_random_seed(100)

    print(OmegaConf.to_yaml(cfg))

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']

    model_name = "SimpleModel_omega_omegadot_feedback"

    model = initialize_model(model_name, system_params, hyperparams)
    
    batch_size = 5
    K = torch.randn(batch_size, 4).to(hyperparams.device) * 20

    diff_params = {
        "K": K,
    }

    watched_idx = 0

    """
    backward propagation
    """
    initial_state = model.get_initial_state(K)

    K.requires_grad = True

    with torch.enable_grad():
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
    
    """
    fmad
    """

    def modify_grad(output, time_idx, state_idx):
        value, grad = AugementModel.pick_value_grad(output = output, time_idx = time_idx, state_idx = state_idx)
        value_abs, grad_abs = AugementModel.abs_loss(value)
        
        return value_abs, grad * grad_abs.unsqueeze(1)
    
    model_2 = initialize_model("AugementModel", system_params, hyperparams)

    initial_state = model_2.get_initial_state(K.requires_grad_(False))

    time_fmad = time.time()
    with torch.no_grad():
        output_2 = solve(model_2, **hyperparams, **diff_params, y0 = initial_state) # (batch_size, time_steps, state_dim)
    print('time_fmad_all:', time.time() - time_fmad)

    # ss
    freq_ss_2, grad_freq_ss_2 = modify_grad(output_2, time_idx = -1, state_idx = 0)

    # nadir
    nadir_idx = torch.argmax(torch.abs(output_2[:, :, 0]), dim=1)
    freq_nadir_2, grad_freq_nadir_2 = modify_grad(output_2, time_idx = nadir_idx, state_idx = 0)

    # rocof
    rocof_idx = torch.argmax(torch.abs(output_2[:,:, 1]), dim=1)
    freq_rocof_2, grad_freq_rocof_2 = modify_grad(output_2, time_idx = rocof_idx, state_idx = 1)

    assert torch.isclose(freq_ss_1, freq_ss_2[watched_idx])
    assert torch.allclose(grad_freq_ss_1, grad_freq_ss_2[watched_idx])
    assert torch.isclose(freq_nadir_1, freq_nadir_2[watched_idx])
    assert torch.allclose(grad_freq_nadir_1, grad_freq_nadir_2[watched_idx])
    assert torch.isclose(freq_rocof_1, freq_rocof_2[watched_idx])
    assert torch.allclose(grad_freq_rocof_1, grad_freq_rocof_2[watched_idx])

if __name__ == "__main__":
    main()