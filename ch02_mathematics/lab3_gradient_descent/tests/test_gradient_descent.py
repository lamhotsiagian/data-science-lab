import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from gradient_descent import quadratic, grad_quadratic, gradient_descent, gradient_descent_momentum, backward_pass, forward_pass

def test_gradient_descent_quadratic():
    traj = gradient_descent(quadratic, grad_quadratic, 10.0, 0.1, 100)
    assert abs(traj[-1]) < 1e-1

def test_gradient_descent_momentum():
    traj = gradient_descent_momentum(quadratic, grad_quadratic, 10.0, 0.1, 0.9, 100)
    assert abs(traj[-1]) < 1e-1

def test_backward_pass():
    dy_dw1, dy_dw2 = backward_pass(1.0, 2.0, 3.0)
    assert dy_dw2 == 2.0
    assert dy_dw1 == 3.0 * 1.0
    
def test_forward_pass():
    y, h = forward_pass(1.0, 2.0, 3.0)
    assert h == 2.0
    assert y == 6.0
