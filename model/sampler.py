import torch
import time
import sys
sys.path.append('.')
from model import AugementModel

def modify_grad(output, time_idx, state_idx):
    # find the value and grad after abs() function
    value, grad = AugementModel.pick_value_grad(output = output, time_idx = time_idx, state_idx = state_idx)
    value_abs, grad_abs = AugementModel.abs_loss(value)
    
    return value_abs, grad * grad_abs.unsqueeze(1)

def gradient_surgery(G: list, method: str):

    if method == 'average':
        grad = torch.stack(G).mean(dim=0)
    
    return grad / torch.norm(grad, p=2, dim=1, keepdim=True)

def fmad_sampler(model: torch.nn.Module, K: torch.Tensor, cfg, solve: callable):
    
    batch_size = cfg['hyperparams']['batch_size']
    no_batch = int(K.shape[0] / batch_size)

    system_params = cfg['system_params']
    tau_ss = system_params.max_ss_dev / system_params.base
    tau_nadir = system_params.max_nadir_dev / system_params.base
    tau_rocof = system_params.max_rocof / system_params.base

    print('tau_ss:', tau_ss)
    print('tau_nadir:', tau_nadir)
    print('tau_rocof:', tau_rocof)

    for i in range(no_batch):
        print(f'batch {i}')
        if i == no_batch - 1:
            K_ = K[i*batch_size:]
        else:
            K_ = K[i*batch_size:(i+1)*batch_size]
        
        for j in range(cfg.hyperparams.max_iter):
            
            initial_state = model.get_initial_state(K_)

            time_fmad = time.time()
            with torch.no_grad():
                output = solve(model, **cfg['hyperparams'], K = K_, y0 = initial_state)
                print('time_fmad_all:', time.time() - time_fmad)

                print('output:', output.shape)  

                # steady state
                freq_ss, grad_freq_ss = modify_grad(output, time_idx = -1, state_idx = 0)
                stability_ss = (freq_ss <= tau_ss).float()

                # nadir
                nadir_idx = torch.argmax(torch.abs(output[:, :, 0]), dim=1)
                freq_nadir, grad_freq_nadir = modify_grad(output, time_idx = nadir_idx, state_idx = 0)
                stability_nadir = (freq_nadir <= tau_nadir).float()
                
                # rocof
                rocof_idx = torch.argmax(torch.abs(output[:,:, 1]), dim=1)
                freq_rocof, grad_freq_rocof = modify_grad(output, time_idx = rocof_idx, state_idx = 1)    
                stability_rocof = (freq_rocof <= tau_rocof).float() # 1 if stable, 0 if unstable

                # unstable if any of the three is unstable (0)
                stability_all = (stability_ss * stability_nadir * stability_rocof).float()

                # our goal is to change from stable (1) to unstable (0) or vice versa
                # from stable (1) to unstable (0): gradient ascent (+)
                # from unstable (0) to stable (1): gradient descent (-)

                gradient_sign = torch.sign(
                    stability_all - 0.5
                )

                grad_freq_ss = grad_freq_ss * gradient_sign.unsqueeze(1)
                grad_freq_nadir = grad_freq_nadir * gradient_sign.unsqueeze(1)
                grad_freq_rocof = grad_freq_rocof * gradient_sign.unsqueeze(1)

                # gradient surgery
                G = [grad_freq_ss, grad_freq_nadir, grad_freq_rocof]
                grad = gradient_surgery(G, method = 'average')

                print('stability_all:', stability_all[0])
                print('freq_ss:', freq_ss[0])
                print('freq_nadir:', freq_nadir[0])
                print('freq_rocof:', freq_rocof[0])

                K_ = K_ + cfg.hyperparams.step_size * grad
