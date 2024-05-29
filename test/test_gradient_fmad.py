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
    torch.set_default_dtype(torch.float64)
    set_random_seed(100)

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']

    K = torch.randn(hyperparams.batch_size, 4).to(hyperparams.device) * 20

    diff_params = {
        "K": K,
    }

    def modify_grad(output, time_idx, state_idx):
        value, grad = AugementModel.pick_value_grad(output = output, time_idx = time_idx, state_idx = state_idx)
        value_abs, grad_abs = AugementModel.abs_loss(value)
        return value_abs, grad * grad_abs.unsqueeze(1)
    
    model_2 = initialize_model("AugementModel", system_params, hyperparams)

    for i in range(hyperparams.max_iter):
        print('iteration:', i)
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

        print('freq_ss', freq_ss_2)
        print('freq_nadir', freq_nadir_2)
        print('freq_rocof', freq_rocof_2)
        print('grad_freq_ss', grad_freq_ss_2)
        print('grad_freq_nadir', grad_freq_nadir_2)
        print('grad_freq_rocof', grad_freq_rocof_2)

        K = K + grad_freq_ss_2 * 1e9 


if __name__ == "__main__":

    main()