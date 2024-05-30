from model import AugementModel, fmad_sampler
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_random_seed, to_numpy, initialize_model, plot
from ode_solver import solve
import numpy as np
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig):

    # Plot options
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 14
    # plt.rcParams['font.weight'] = 'bold'
    # plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['figure.figsize'] = (6,4)
    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fontsize'] = 14
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix",
        }
    plt.rcParams.update(rc)

    hyperparams = cfg['hyperparams']
    system_params = cfg['system_params']

    linestep = np.linspace(hyperparams["t0"], hyperparams["t1"], 
                    int((hyperparams["t1"] - hyperparams["t0"]) / hyperparams["dt"]))

    base = system_params.base
    ss_max = base + system_params.max_ss_dev
    ss_min = base - system_params.max_ss_dev
    rocof_max = system_params.max_rocof
    rocof_min = -system_params.max_rocof
    nadir_max = base + system_params.max_nadir_dev
    nadir_min = base - system_params.max_nadir_dev

    def my_plot(before_sample_output, after_sample_output, sample_idx):

        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        axs[0].plot(linestep, (1+before_sample_output[0,:,0]) * base, label = 'Original Sample', color = plt.cm.tab20c(8))
        axs[0].plot(linestep, (1+after_sample_output[0,:,0]) * base, label = 'New Sample', color = plt.cm.tab20c(12))
        axs[0].hlines(y = ss_max, xmin = hyperparams['t0'], xmax = hyperparams["t1"], label = "SS Constraint", linestyle = '--', color = plt.cm.tab20c(0))
        axs[0].hlines(y = ss_min, xmin = hyperparams['t0'], xmax = hyperparams["t1"], linestyle = '--', color = plt.cm.tab20c(0))
        axs[0].hlines(y = nadir_max, xmin = hyperparams['t0'], xmax = hyperparams["t1"], label = "Nadir Constraint", linestyle = '--', color = plt.cm.tab20c(4))
        axs[0].hlines(y = nadir_min, xmin = hyperparams['t0'], xmax = hyperparams["t1"], linestyle = '--', color = plt.cm.tab20c(4))
        axs[0].grid()
        axs[0].legend(labelspacing=0.2, loc = 'best')
        axs[0].set_ylim([48.9, 51.1])
        axs[0].set_xticks(np.arange(0, 21, 2))
        axs[0].set_xticklabels([])
        axs[0].set_ylabel('Frequency (Hz)')

        axs[1].plot(linestep, before_sample_output[0,:,1] * base, color = plt.cm.tab20c(8))
        axs[1].plot(linestep, after_sample_output[0,:,1] * base, color = plt.cm.tab20c(12))
        axs[1].hlines(y = rocof_max, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = plt.cm.tab20c(4), label = "RoCoF Constraint", linestyle = '--')
        axs[1].hlines(y = rocof_min, xmin = hyperparams['t0'], xmax = hyperparams["t1"], color = plt.cm.tab20c(4), linestyle = '--')
        axs[1].legend(loc = 'best')
        axs[1].set_ylim([-1.2, 1.2])
        axs[1].set_xticks(np.arange(0, 21, 2))
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('RoCoF (Hz/s)')
        axs[1].grid()

        plt.savefig(f'fig/compare_samples_{sample_idx}.pdf', bbox_inches = 'tight')

        plt.close()

    model = initialize_model(name = 'SimpleModel_omega_omegadot_feedback', 
                        system_params=system_params, hyperparams=hyperparams)

    K_all = np.load('K_all.npy')
    stability_all = np.load('stability_all.npy')

    for sample_idx in [41, 48]:
    
        before_sample_idx = 0
        stability = stability_all[sample_idx]
        K = torch.tensor(K_all[sample_idx])

        try:
            after_sample_idx = np.where(
            stability != stability[before_sample_idx]
            )[0][0]
        except:
            after_sample_idx = -1

        before_sample_K = K[before_sample_idx].unsqueeze(0)
        initial_state = model.get_initial_state(before_sample_K)
        before_sample_output = solve(model, **cfg['hyperparams'], K = before_sample_K, y0 = initial_state)

        after_sample_K = K[after_sample_idx].unsqueeze(0)
        initial_state = model.get_initial_state(after_sample_K)

        after_sample_output = solve(model, **cfg['hyperparams'], K = after_sample_K, y0 = initial_state)

        my_plot(before_sample_output, after_sample_output, sample_idx)

if __name__ == "__main__":
    main()