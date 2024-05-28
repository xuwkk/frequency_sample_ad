import sys
sys.path.append('.')
from model import SimpleModel_omega_omegadot_feedback
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.autograd.forward_ad as fwAD
from torch.autograd.functional import jacobian
from torch.func import jvp
from functools import partial
from utils import set_random_seed

def jvp_fmad(state, K, model, x_dot, e):

    no_sys_state = state.shape[1]

    jvp_all = []
    # consider df_dx * x_dot_i + df_dtheta * e_i
    for i in range(K.shape[1]):
        with fwAD.dual_level():
            dual_x_dot = fwAD.make_dual(state, x_dot[:,i*no_sys_state:(i+1)*no_sys_state])
            dual_theta = fwAD.make_dual(K, e[:,i])
            dual_output = model(0, dual_x_dot, dual_theta)

            output, jvp_single = fwAD.unpack_dual(dual_output)

            jvp_all.append(jvp_single)
    
    jvp_all = torch.concat(jvp_all, dim=1)

    return output, jvp_all

def jvp_fmad_simple(state, K, model, x_dot, e):

    no_sys_state = state.shape[1]

    jvp_all = []
    # consider df_dx * x_dot_i + df_dtheta * e_i
    for i in range(K.shape[1]):
        output, jvp_single = jvp(partial(model, 0.0), (state, K), (x_dot[:,i*no_sys_state:(i+1)*no_sys_state], e[:,i]))
        jvp_all.append(jvp_single)
    
    jvp_all = torch.concat(jvp_all, dim=1)

    return output, jvp_all

def jvp_jac(state, K, model, x_dot, e):

    no_sys_state = state.shape[1]

    jvp_all = []

    for i in range(K.shape[0]):
        df_dx, df_dtheta = jacobian(partial(model, 0.0), (state[i:i+1], K[i:i+1]))
        df_dx = df_dx.squeeze((0,2))
        df_dtheta = df_dtheta.squeeze((0,2))
        
        # need to flatten along the column
        jvp_single = df_dx @ x_dot[i].reshape(-1, no_sys_state).T + df_dtheta @ e[i]
        jvp_all.append(jvp_single.T.flatten().unsqueeze(0))
    
    jvp_all = torch.concat(jvp_all, dim=0)

    return model(0, state, K), jvp_all


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

    K = torch.zeros(hyperparams.batch_size, 4)

    diff_params = {
        "K": K,
    }

    model = SimpleModel_omega_omegadot_feedback(cons_params)
    initial_state = model.get_initial_state(K)
    t = 0.

    no_sys_state = 2
    no_aug_state = 8

    # tangent
    x_dot = torch.randn((hyperparams.batch_size, no_sys_state * K.shape[1])) # (batch_size, no_sys_state * no_diff_params)
    e = torch.eye(K.shape[1]).unsqueeze(0).repeat(hyperparams.batch_size, 1, 1) 
    
    update_sys_1, update_aug_1 = jvp_fmad(initial_state, K, model, x_dot, e)
    update_sys_2, update_aug_2 = jvp_fmad_simple(initial_state, K, model, x_dot, e)

    update_sys_3, update_aug_3 = jvp_jac(initial_state, K, model, x_dot, e)

    assert torch.allclose(update_sys_1, update_sys_2)
    assert torch.allclose(update_aug_1, update_aug_2)

    assert torch.allclose(update_sys_1, update_sys_3)
    assert torch.allclose(update_aug_1, update_aug_3)

if __name__ == "__main__":
    main()

        
