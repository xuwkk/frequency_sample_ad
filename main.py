from model import AugementModel, fmad_sampler
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


@hydra.main(version_base=None, config_path="conf/", config_name="config")
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

    K = torch.randn((hyperparams.sample_no, 4), device = hyperparams.device) * 100

    diff_params = {
        "K": K,
    }

    model = AugementModel(cons_params, device = hyperparams.device)

    fmad_sampler(model, K, cfg, solve)




if __name__ == "__main__":
    main()