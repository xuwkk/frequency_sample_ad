"""
power system dynamic model written in PyTorch

Reference:
[1] Fast Frequency Control Scheme through Adaptive Virtual Inertia Emulation
"""

import torch
from torch.func import jvp
import torch.nn as nn
import abc
from torch.func import jvp
from functools import partial
import time

class DynModel(nn.Module):

    def __init__(self, cons_params: dict, device):
        """
        cons_params: the parameters that do not need to be optimized
        """

        super().__init__()
        for key, value in cons_params.items():
            setattr(self, key, torch.as_tensor([value]).to(device))

    @abc.abstractmethod
    def get_initial_state(self):
        """
        return the initial state of the system
        """
        return NotImplementedError
    
    @abc.abstractmethod
    def forward(self, t: float, state: torch.Tensor, diff_params: dict):
        """
        rhs of the ODE, the f function in the ODE dy/dt = f(t, y, theta)
        """
        raise NotImplementedError

class SimpleModel_omega_q(DynModel):

    """
    implement the ODE in (1)-(2) of [1]
    """

    def get_initial_state(self, M: torch.Tensor, D: torch.Tensor):

        # initial conditions are constantly equal to 0
        omega_init = torch.zeros_like(M)
        q = torch.zeros_like(M)
    
        initial_state = torch.concat([omega_init, q], dim=1)
        
        return initial_state
    
    def forward(self, t: float, state: torch.Tensor, M: torch.Tensor, D: torch.Tensor, **kwargs):
        omega, q = state[:,0:1], state[:,1:2]
        d_omega = (-D * omega + q + self.delta_P) / M
        d_q = (-1/self.r * omega - q) / self.tau

        return torch.concat([d_omega, d_q], dim=1)
    
    def cal_d_omega(self, omega: torch.Tensor, q: torch.Tensor, 
                    M: torch.Tensor, D: torch.Tensor):
        """
        calculate the derivative of omega
        """
        return (-D * omega +q + self.delta_P) / M

class SimpleModel_omega_omegadot(DynModel):

    """
    implement the ODE in (35) of [1]
    """

    def get_initial_state(self, M: torch.Tensor, D: torch.Tensor):
        omega_init = torch.zeros_like(M)
        omega_dot_init = self.delta_P / M # eq.(1) of [1]
        initial_state = torch.concat([omega_init, omega_dot_init], dim=1)
        
        return initial_state
    
    def forward(self, t: float, state: torch.Tensor, M: torch.Tensor, D: torch.Tensor, **kwargs):
        omega, omega_dot = state[:,0:1], state[:,1:2]
        d_omega = omega_dot
        d_omega_dot = (
            -(1/(self.r * self.tau * M ) + D / (self.tau * M)) * omega
            - (D / M + 1/self.tau) * omega_dot
            + self.delta_P / (self.tau * M)
                        )
        return torch.concat([d_omega, d_omega_dot], dim=1)

class SimpleModel_omega_omegadot_feedback(DynModel):

    """
    implement the ODE in (35) of [1] with state feedback on M and D
    """

    # the initial M and D should be the constant parameters

    def get_initial_state(self, K: torch.Tensor):
        omega_init = torch.zeros((K.shape[0], 1)).to(K.device)
        # we can only have the minus sign here
        non_zero_idx = torch.where(K[:, 1:2] != 0)[0]
        omega_dot_init = torch.zeros((K.shape[0], 1)).to(K.device)
        omega_dot_init[non_zero_idx] = (self.M - torch.sqrt(self.M ** 2 - 4 * K[non_zero_idx, 1:2] * self.delta_P) ) / (2 * K[non_zero_idx, 1:2])
        zero_idx = torch.where(K[:, 1:2] == 0)[0]
        omega_dot_init[zero_idx] = self.delta_P / self.M
        # omega_dot_init = (self.M - torch.sqrt(self.M ** 2 - 4 * K[:, 1:2] * self.delta_P) ) / (2 * K[:, 1:2])
        # # eliminate nan
        # omega_dot_init[torch.isnan(omega_dot_init)] = self.delta_P / self.M

        assert torch.isnan(omega_dot_init).sum() == 0, 'nan found in omega_dot_init'

        initial_state = torch.concat([omega_init, omega_dot_init], dim=1)

        return initial_state
    
    def get_initial_state_single(self, K: torch.Tensor):
        omega_init = torch.zeros(1).to(K.device)
        # todo: in vmap we cannot use the if else statement, let's just assume K12 is not zero
        # see: https://github.com/pytorch/functorch/issues/257
        omega_dot_init = (self.M - torch.sqrt(self.M ** 2 - 4 * K[1] * self.delta_P) ) / (2 * K[1])
        
        return torch.concat([omega_init, omega_dot_init])

    
    def forward(self, t: float, state: torch.Tensor, K: torch.Tensor, **kwargs):
        omega, omega_dot = state[:,0:1], state[:,1:2]
        d_omega = omega_dot

        # the feedback on M and D
        M = self.M - K[:, 0:1] * omega - K[:, 1:2] * omega_dot # self.M is scalar
        D = self.D - K[:, 2:3] * omega - K[:, 3:] * omega_dot

        # same as before
        d_omega_dot = (
            -(1/(self.r * self.tau * M ) + D / (self.tau * M)) * omega
            - (D / M + 1/self.tau) * omega_dot
            + self.delta_P / (self.tau * M)
                        )
        return torch.concat([d_omega, d_omega_dot], dim=1)
    
    def forward_single(self, t: float, state: torch.Tensor, K: torch.Tensor, **kwargs):
        """
        forward pass for one sample
        """

        omega, omega_dot = state[0], state[1]
        d_omega = omega_dot

        # the feedback on M and D
        M = self.M - K[0] * omega - K[1] * omega_dot
        D = self.D - K[2] * omega - K[3] * omega_dot

        # same as before
        d_omega_dot = (
            -(1/(self.r * self.tau * M ) + D / (self.tau * M)) * omega
            - (D / M + 1/self.tau) * omega_dot
            + self.delta_P / (self.tau * M)
                        )

        return torch.concat([d_omega.unsqueeze(0), d_omega_dot])

class AugementModel(DynModel):

    """
    the FMAD augmented model in the paper
    """

    def get_initial_state(self, K: torch.Tensor):

        omega_init = torch.zeros((K.shape[0], 1)).to(K.device)
        # we can only have the minus sign here, see the proof in the paper

        # def get_init_omega_dot(K12):
        #     return (self.M - torch.sqrt(self.M ** 2 - 4 * K12 * self.delta_P) ) / (2 * K12)
        
        non_zero_idx = torch.where(K[:, 1:2] != 0)[0]
        omega_dot_init = torch.zeros((K.shape[0], 1)).to(K.device)
        omega_dot_init[non_zero_idx] = (self.M - torch.sqrt(self.M ** 2 - 4 * K[non_zero_idx, 1:2] * self.delta_P) ) / (2 * K[non_zero_idx, 1:2])

        zero_idx = torch.where(K[:, 1:2] == 0)[0]
        omega_dot_init[zero_idx] = self.delta_P / self.M

        assert torch.isnan(omega_dot_init).sum() == 0, 'nan found in omega_dot_init'

        # omega_dot_init = (self.M - torch.sqrt(self.M ** 2 - 4 * K[:, 1:2] * self.delta_P) ) / (2 * K[:, 1:2])
        # omega_dot_init = get_init_omega_dot(K[:, 1:2])
        # eliminate nan: nan may be caused by the square root or because K12 is zero
        # omega_dot_init[torch.isnan(omega_dot_init)] = self.delta_P / self.M

        # tangent state
        tangent_state_init = torch.zeros((omega_dot_init.shape[0], K.shape[1] * 2)).to(K.device)
        # non_zero_entry = torch.autograd.functional.jacobian(get_init_omega_dot, K[:, 1:2])

        # the initial value of thete_2 is different
        non_zero_entry = (
            (2*torch.pow(self.M**2 - 4 * K[:, 1:2] * self.delta_P, -0.5)*self.delta_P * K[:, 1:2] - (self.M - torch.sqrt(self.M**2 - 4 * K[:, 1:2] * self.delta_P))
                ) / (2 * torch.square(K[:, 1:2]))
                )
        
        assert torch.isnan(non_zero_entry).sum() == 0, 'nan found in non_zero_entry'

        # non_zero_entry_test = torch.autograd.functional.jacobian(get_init_omega_dot, K[:, 1:2])
        
        tangent_state_init[:, 3:4] = non_zero_entry # ! either by AD or by analytical calculation
        
        return torch.concat([omega_init, omega_dot_init, tangent_state_init], dim=1)
    
    def forward(self, t: float, state: torch.Tensor, K: torch.Tensor, **kwargs):
        
        def forward_(t, sys_state, K):

            # for the original system
            omega, omega_dot = sys_state[:,0:1], sys_state[:,1:2]
            d_omega = omega_dot

            # the feedback on M and D
            M = self.M - K[:, 0:1] * omega - K[:, 1:2] * omega_dot # self.M is scalar
            D = self.D - K[:, 2:3] * omega - K[:, 3:] * omega_dot

            # same as before
            d_omega_dot = (
            -(1/(self.r * self.tau * M ) + D / (self.tau * M)) * omega
            - (D / M + 1/self.tau) * omega_dot
            + self.delta_P / (self.tau * M)
                        )
            
            return torch.concat([d_omega, d_omega_dot], dim=1) 

        no_params = K.shape[1]
        no_sys_state = int(state.shape[1] / (no_params + 1))
        no_aug_state = int(no_sys_state * no_params)
        e = torch.eye(no_params).unsqueeze(0).repeat(K.shape[0], 1, 1)
        sys_state, aug_state = state[:, :no_sys_state], state[:, no_sys_state:]

        aug_state_all = []
        for i in range(no_params):
            # forward pass
            output, jvp_single = jvp(
                partial(forward_, t),
                (sys_state, K),
                (aug_state[:, i*no_sys_state: (i+1)*no_sys_state], e[:, i])
            )
            aug_state_all.append(jvp_single)
        
        aug_state_all = torch.concat(aug_state_all, dim=1)

        return torch.concat([output, aug_state_all], dim=1)
    
    @staticmethod
    def abs_loss(output):
        value = torch.abs(output)
        grad = torch.sign(output)

        return value, grad
    
    @staticmethod
    def pick_value_grad(output, time_idx, state_idx):
        # ! output: (batch_size, no_sys_state * (no_params + 1))
        batch_size = output.shape[0]
        output_reshape = output[torch.arange(batch_size), time_idx].reshape(output.shape[0], -1, 2)
        value = output_reshape[:, 0, state_idx]
        grad = output_reshape[:, 1:, state_idx]
        
        return value, grad