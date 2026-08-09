import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from memory_profiling import *

def test_memory_functions():
    n_params = 1000
    assert calculate_parameter_memory(n_params, 4) == 4000
    assert calculate_gradient_memory(n_params, 4) == 4000
    assert calculate_optimizer_memory(n_params, "adam") == 8000
    
    breakdown = total_training_memory(n_params, 1, 10, 128, 2, 4)
    assert breakdown.total == breakdown.params + breakdown.gradients + breakdown.optimizer_states + breakdown.activations
    assert breakdown.params > 0

def test_model_profiling():
    model = create_small_transformer()
    res = profile_model_empirically(model)
    assert res['n_params'] > 0
    assert res['param_memory_bytes'] > 0
    
    comp = compare_theoretical_vs_empirical(model)
    assert 'empirical' in comp
