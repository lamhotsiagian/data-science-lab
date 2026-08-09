import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def quadratic(x):
    return x**2

def grad_quadratic(x):
    return 2*x

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    df_dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    df_dx1 = 200*(x[1] - x[0]**2)
    return np.array([df_dx0, df_dx1])

def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def grad_beale(x):
    """Analytical gradient of the Beale function.

    f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Each term t_i = (c_i - x + x*y^i) contributes 2*t_i * d(t_i)/d(param):
        dt_i/dx = -1 + y^i
        dt_i/dy = i * x * y^(i-1)
    """
    x0, y0 = float(x[0]), float(x[1])
    t1 = 1.5 - x0 + x0 * y0
    t2 = 2.25 - x0 + x0 * y0**2
    t3 = 2.625 - x0 + x0 * y0**3

    df_dx = 2 * t1 * (y0 - 1) + 2 * t2 * (y0**2 - 1) + 2 * t3 * (y0**3 - 1)
    df_dy = 2 * t1 * x0 + 2 * t2 * (2 * x0 * y0) + 2 * t3 * (3 * x0 * y0**2)
    return np.array([df_dx, df_dy])

def gradient_descent(f, grad_f, x0, lr, n_iter):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(n_iter):
        x = x - lr * np.array(grad_f(x))
        traj.append(x.copy())
    return traj

def gradient_descent_momentum(f, grad_f, x0, lr, beta, n_iter):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    traj = [x.copy()]
    for _ in range(n_iter):
        v = beta * v + lr * np.array(grad_f(x))
        x = x - v
        traj.append(x.copy())
    return traj

def forward_pass(x, w1, w2):
    h = np.maximum(0, w1 * x) # relu
    y = w2 * h
    return y, h

def backward_pass(x, w1, w2):
    h = np.maximum(0, w1 * x)
    dy_dw2 = h
    dy_dh = w2
    dh_dw1 = x if w1 * x > 0 else 0
    dy_dw1 = dy_dh * dh_dw1
    return dy_dw1, dy_dw2

def plot_trajectory_2d(f, trajectory):
    pass

def plot_convergence(losses):
    pass
    
def plot_lr_comparison(f, grad_f, x0, learning_rates):
    pass
