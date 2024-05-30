from model import AugementModel, fmad_sampler
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_random_seed, to_numpy, initialize_model, get_K, get_K_threshold
from ode_solver import solve
import numpy as np


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig):

    set_random_seed(100)

    print(OmegaConf.to_yaml(cfg))

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']

    # K = torch.randn((hyperparams.sample_no, 4), device = hyperparams.device) * 100
    K = get_K(hyperparams, system_params)
    K_threshold = get_K_threshold(system_params)

    print('initial data size:', K.shape)

    diff_params = {
        "K": K,
    }

    model = initialize_model(name = 'AugementModel', 
                            system_params=system_params, hyperparams=hyperparams)

    K_all, stability_all = fmad_sampler(model, K, cfg, solve, K_threshold)

    K_all, stability_all = to_numpy(K_all), to_numpy(stability_all)

    np.save('K_all.npy', K_all)
    np.save('stability_all.npy', stability_all)

if __name__ == "__main__":
    main()