import torch
import time
import sys
sys.path.append('.')
from model import AugementModel
from functools import partial
from torch.func import vmap, grad

def modify_grad(output, time_idx, state_idx):
    # find the value and grad after abs() function
    value, grad = AugementModel.pick_value_grad(output = output, time_idx = time_idx, state_idx = state_idx)
    value_abs, grad_abs = AugementModel.abs_loss(value)
    
    return value_abs, grad * grad_abs.unsqueeze(1)


def gradient_surgery(G: list, method: str, cos: torch.nn.CosineSimilarity):

    G = torch.stack(G, dim = 0) # 3 x batch_size x param_size

    if method == 'average':
        G = G / (torch.norm(G, p=2, dim=2, keepdim=True) + 1e-7)
        # ! do not consider the ss gradient
        grad = torch.mean(G[1:], dim = 0) # batch_size x param_size 
    
    elif method == 'ss':
        Warning('ss method is deprecated because the ss does not depend on the parameters.')
        grad = G[0].clone() / (torch.norm(G[0], p=2, dim=1, keepdim=True) + 1e-7)
    
    elif method == 'nadir':
        grad = G[1].clone() / (torch.norm(G[1], p=2, dim=1, keepdim=True) + 1e-7)

    elif method == 'rocof':
        grad = G[2].clone() / (torch.norm(G[2], p=2, dim=1, keepdim=True) + 1e-7)
    
    elif method == 'surgery':
        G = G / (torch.norm(G, p=2, dim=2, keepdim=True) + 1e-7)
        grad = torch.zeros_like(G[0]).to(G[0].device)

        for i in range(len(G)):
            g_p = G[i].clone() # the gradient of the i-th loss
            G_remain = torch.cat((G[:i], G[i+1:]), dim = 0) # (2 x batch_size x param_size)

            for j in range(len(G_remain)):
                inner_prod = torch.einsum('ij,ij->i', g_p, G_remain[j]) # batcwise inner product
                # print('inner_prod:', inner_prod)
                negative_idx = torch.where(inner_prod < 0)[0]
                if len(negative_idx) > 0:
                    g_p[negative_idx] = g_p[negative_idx] - inner_prod[negative_idx] / torch.norm(G_remain[j, negative_idx], p=2, dim=1) * G_remain[j, negative_idx]
                    Warning(f'negative inner product occurs: {len(negative_idx)}')
            
            grad = grad + g_p

    return grad


def fmad_method(model: torch.nn.Module, K: torch.Tensor, solve: callable, cfg: dict):
    """
    single iteration of fmad
    """
    initial_state = model.get_initial_state(K.requires_grad_(False))
    with torch.no_grad():
        output = solve(model, **cfg['hyperparams'], K = K, y0 = initial_state)
    
    freq_ss, grad_freq_ss = modify_grad(output, time_idx = -1, state_idx = 0)

    nadir_idx = torch.argmax(torch.abs(output[:, :, 0]), dim=1)
    freq_nadir, grad_freq_nadir = modify_grad(output, time_idx = nadir_idx, state_idx = 0)

    rocof_idx = torch.argmax(torch.abs(output[:,:, 1]), dim=1)
    freq_rocof, grad_freq_rocof = modify_grad(output, time_idx = rocof_idx, state_idx = 1)

    return freq_ss, grad_freq_ss, freq_nadir, grad_freq_nadir, freq_rocof, grad_freq_rocof


def unrolling_method(model: torch.nn.Module, K: torch.Tensor, solve: callable, cfg: dict):
    """
    single iteration of unrolling
    """
    hyperparams = cfg['hyperparams']

    """
    define the loss functions for a *single* sample
    """
    def loss_ss(model, K):
        # single sample ss loss
        initial_state = model.get_initial_state_single(K) # ! the initial value is also a function of K
        output = solve(model.forward_single, **hyperparams, K = K, y0 = initial_state)
        return torch.abs(output[-1, 0])

    def loss_nadir(model, K):
        # single sample nadir loss
        initial_state = model.get_initial_state_single(K)
        output = solve(model.forward_single, **hyperparams, K = K, y0 = initial_state)

        return torch.max(torch.abs(output[:, 0]))
    
    def loss_rocof(model, K):
        # single sample rocof loss
        initial_state = model.get_initial_state_single(K)
        output = solve(model.forward_single, **hyperparams, K = K, y0 = initial_state)
        return torch.max(torch.abs(output[:, 1]))
    
    # with torch.enable_grad():

    freq_ss = vmap(loss_ss, in_dims = (None, 0))(model, K)
    grad_freq_ss = vmap(grad(partial(loss_ss, model)))(K)

    freq_nadir = vmap(partial(loss_nadir, model))(K)
    grad_freq_nadir = vmap(grad(partial(loss_nadir, model)))(K)

    freq_rocof = vmap(partial(loss_rocof, model))(K)
    grad_freq_rocof = vmap(grad(partial(loss_rocof, model)))(K)
    
    return freq_ss, grad_freq_ss, freq_nadir, grad_freq_nadir, freq_rocof, grad_freq_rocof


def finite_method(model: torch.nn.Module, K: torch.Tensor, solve: callable, cfg: dict):

    """
    single iteration of finite difference
    """
    
    eps = cfg.hyperparams.finite_eps

    """
    original ODE
    """
    initial_state = model.get_initial_state(K)
    output = solve(model, **cfg['hyperparams'], K = K, y0 = initial_state)

    freq_ss = torch.abs(output[:, -1, 0])

    freq_nadir_idx = torch.argmax(torch.abs(output[:, :, 0]), dim = 1)
    freq_nadir = torch.abs(output[torch.arange(K.shape[0]), freq_nadir_idx, 0])

    freq_rocof_idx = torch.argmax(torch.abs(output[:, :, 1]), dim = 1)
    freq_rocof = torch.abs(output[torch.arange(K.shape[0]), freq_rocof_idx, 1])

    """
    perturb K
    """
    perturb = torch.zeros_like(K)
    output_all = []
    for j in range(4):
        # finite difference
        perturb_j = perturb.clone()
        perturb_j[:,j] = eps
        K_perturb = K + perturb_j
        initial_state = model.get_initial_state(K_perturb) # ! the initial value is also a function of K
        output_perturb = solve(model, **cfg.hyperparams, K = K_perturb, y0 = initial_state)
        output_all.append(output_perturb)
    
    freq_ss_grad = torch.zeros_like(K)
    freq_nadir_grad = torch.zeros_like(K)
    freq_rocof_grad = torch.zeros_like(K)

    for j in range(4):
        freq_ss_grad[:, j] = (freq_ss - torch.abs(output_all[j][:, -1, 0])) / eps
        freq_nadir_grad[:, j] = (freq_nadir - torch.abs(output_all[j][torch.arange(K.shape[0]), freq_nadir_idx, 0])) / eps
        freq_rocof_grad[:, j] = (freq_rocof - torch.abs(output_all[j][torch.arange(K.shape[0]), freq_rocof_idx, 1])) / eps

    return freq_ss, freq_ss_grad, freq_nadir, freq_nadir_grad, freq_rocof, freq_rocof_grad
    

@torch.no_grad()
def fmad_sampler(model: torch.nn.Module, K: torch.Tensor, cfg, solve: callable, K_threshold):

    """
    the complete sampling algorithm
    """
    
    batch_size = cfg['hyperparams']['batch_size']
    no_batch = int(K.shape[0] / batch_size)

    system_params = cfg['system_params']
    tau_ss = system_params.max_ss_dev / system_params.base
    tau_nadir = system_params.max_nadir_dev / system_params.base
    tau_rocof = system_params.max_rocof / system_params.base

    print('critical absolute measure:')
    print('tau_ss:', tau_ss)
    print('tau_nadir:', tau_nadir)
    print('tau_rocof:', tau_rocof)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # K_max = K_threshold(system_params)

    print('K_threshold:', K_threshold)

    # step size decay
    step_size_all = torch.linspace(cfg.hyperparams.step_size, cfg.hyperparams.step_size * 0.1, cfg.hyperparams.max_iter)

    stability_all__batch_all = []
    K__batch_all = []

    if hasattr(cfg, 'watched_idx'):
        watched_idx = cfg.watched_idx

    for i in range(no_batch):
        print(f'batch {i}')
        if i == no_batch - 1:
            K_ = K[i*batch_size:]
        else:
            K_ = K[i*batch_size:(i+1)*batch_size]
        
        K__batch = []
        stability_all__batch = []
        
        for j in range(cfg.hyperparams.max_iter):
            
            initial_state = model.get_initial_state(K_)

            time_fmad = time.time()
            
            freq_ss, grad_freq_ss, freq_nadir, grad_freq_nadir, freq_rocof, grad_freq_rocof = fmad_method(model = model, K = K_, solve = solve, cfg = cfg)

            # output = solve(model, **cfg['hyperparams'], K = K_, y0 = initial_state)

            # steady state
            # freq_ss, grad_freq_ss = modify_grad(output, time_idx = -1, state_idx = 0)
            stability_ss = (freq_ss <= tau_ss).float()

            # nadir
            # nadir_idx = torch.argmax(torch.abs(output[:, :, 0]), dim=1)
            # freq_nadir, grad_freq_nadir = modify_grad(output, time_idx = nadir_idx, state_idx = 0)
            stability_nadir = (freq_nadir <= tau_nadir).float()
            
            # rocof
            # rocof_idx = torch.argmax(torch.abs(output[:,:, 1]), dim=1)
            # freq_rocof, grad_freq_rocof = modify_grad(output, time_idx = rocof_idx, state_idx = 1)    
            stability_rocof = (freq_rocof <= tau_rocof).float() # 1 if stable, 0 if unstable

            # unstable if any of the three is unstable (0)
            stability_all = (stability_ss * stability_nadir * stability_rocof).float()

            # record each iteration
            K__batch.append(K_)
            stability_all__batch.append(stability_all)

            # gradient surgery
            # the gradients point to the direction of increase the abs of the loss
            G = [grad_freq_ss, grad_freq_nadir, grad_freq_rocof]

            # an aggregate gradient to INCREASE the abs of the loss
            grad = gradient_surgery(G, method = 'surgery', cos=cos) 

            # our goal is to change from stable (1) to unstable (0) or vice versa
            # from stable (1) to unstable (0): gradient ascent (+)
            # from unstable (0) to stable (1): gradient descent (-)
            gradient_sign = torch.sign(
                stability_all - 0.5
            )
            grad = grad * gradient_sign.unsqueeze(1)

            if hasattr(cfg, 'watched_idx'):
                print('stability:', stability_all[watched_idx].item())
                print('freq_ss:', freq_ss[watched_idx].item())
                print('freq_nadir:', freq_nadir[watched_idx].item())
                print('freq_rocof:', freq_rocof[watched_idx].item())

            K_ = K_ + step_size_all[j] * grad

            if system_params.delta_P > 0:
                K_[:, 1] = torch.clamp(K_[:, 1], min = -1e6, max = K_threshold)
            else:
                K_[:, 1] = torch.clamp(K_[:, 1], min = K_threshold, max = 1e6)

        K__batch_all.append(torch.stack(K__batch, dim = 1))
        stability_all__batch_all.append(torch.stack(stability_all__batch, dim = 1))
    
    return torch.concatenate(K__batch_all, dim = 0), torch.concatenate(stability_all__batch_all, dim = 0)
