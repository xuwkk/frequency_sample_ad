import torch
import time

def euler_step(f, t, y, dt, **kwargs):
    """
    one step of the Euler method
    """
    return y + dt * f(t, y, **kwargs)

def rk4_step(f, t, y, dt, **kwargs):
    """
    one step of the Runge-Kutta 4th order method
    """
    k1 = f(t, y, **kwargs)
    k2 = f(t + dt/2, y + dt/2 * k1, **kwargs)
    k3 = f(t + dt/2, y + dt/2 * k2, **kwargs)
    k4 = f(t + dt, y + dt * k3, **kwargs)
    
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def solve(f, t0, y0, t1, dt, method = 'euler', **kwargs):
    """
    solve the ODE y'(t) = f(t, y(t)) from t0 to t1 with initial condition y0
    """
    
    t = t0
    y = y0
    y_summary = []
    while t < t1:
        if method == 'euler':
            y = euler_step(f, t, y, dt, **kwargs)
        elif method == 'rk4': 
            y = rk4_step(f, t, y, dt, **kwargs)
        t += dt
        y_summary.append(y)
    
    if isinstance(y0, torch.Tensor):
        y_summary = torch.stack(y_summary).transpose(0,1)
    else:
        raise NotImplementedError
    
    return y_summary