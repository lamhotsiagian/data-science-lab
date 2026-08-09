import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import torch
from lora import *

def test_lora():
    model = create_small_language_model(100, 32, 2, 1)
    orig_params = count_total_params(model)
    orig_trainable = count_trainable_params(model)
    
    model = apply_lora(model, target_modules=['fc'], rank=2)
    new_trainable = count_trainable_params(model)
    
    assert new_trainable < orig_trainable
    assert get_param_reduction_ratio(model) < 1.0
    
    x = torch.randint(0, 100, (1, 10))
    out = model(x)
    assert out.shape == (1, 10, 100)

def test_training():
    res = compare_lora_vs_full(epochs=2)
    assert len(res["full_history"]) == 2
    assert len(res["lora_history"]) == 2
