import sys
sys.path.append('.')
from model import SimpleModel_omega_q, SimpleModel_omega_omegadot, SimpleModel_omega_omegadot_feedback
from ode_solver import solve
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import plot

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def test_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']
    
    """
    the omega-q model
    """

    cons_params = {
        "delta_P": system_params.delta_P,
        "tau": system_params.tau,
        "r": system_params.r,
    }

    model_1 = SimpleModel_omega_q(cons_params, device = hyperparams.device)
    M = system_params.M0 * torch.ones(hyperparams.batch_size, 1)
    D = system_params.M0 * torch.ones(hyperparams.batch_size, 1)

    diff_params = {
        "M": M,
        "D": D,
    }

    initial_state = model_1.get_initial_state(M,D).to(hyperparams.device)
    output_1 = solve(model_1, **hyperparams, **diff_params, y0 = initial_state)
    d_omega = model_1.cal_d_omega(omega = output_1[:,:,0], q = output_1[:,:,1], M = M, D = D)
    sample_idx = 1
    plot(output_1[sample_idx, :, 0].detach().numpy(), d_omega[sample_idx, :].detach().numpy(), "test/omega_q", cfg)

    exit()
    """
    the omega-omegadot model
    """

    model_2 = SimpleModel_omega_omegadot(cons_params)
    initial_state = model_2.get_initial_state(M,D)
    output_2 = solve(model_2, **hyperparams, **diff_params, y0 = initial_state)

    plot(output_2[sample_idx, :, 0].detach().numpy(), output_2[sample_idx, :, 1].detach().numpy(), 
        "test/omega_omegadot", cfg)

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

    K = torch.zeros(hyperparams.batch_size, 4)

    diff_params = {
        "K": K,
    }

    model_3 = SimpleModel_omega_omegadot_feedback(cons_params)
    initial_state = model_3.get_initial_state(K)

    output_3 = solve(model_3, **hyperparams, **diff_params, y0 = initial_state)

    plot(output_3[sample_idx, :, 0].detach().numpy(), output_3[sample_idx, :, 1].detach().numpy(), 
        "test/omega_omegadot_feedback", cfg)
    
    """
    change the K values
    """
    K = - torch.randn(hyperparams.batch_size, 4) * 100
    K[:, 1:2] = - torch.ones_like(K[:, 1:2]) * 10
    diff_params = {
        "K": K,
    }

    initial_state = model_3.get_initial_state(K)
    output_3 = solve(model_3, **hyperparams, **diff_params, y0 = initial_state)

    plot(output_3[sample_idx, :, 0].detach().numpy(), output_3[sample_idx, :, 1].detach().numpy(), 
        "test/omega_omegadot_feedback_K1", cfg)
    

if __name__ == "__main__":
    test_model()