import sys, os
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
from gradient_checkpointing import DeepNetwork, measure_peak_memory

def test_gradient_checkpointing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepNetwork(5, 10).to(device)
    X1 = torch.randn(10, 10, device=device, requires_grad=True)
    X2 = X1.clone().detach().requires_grad_(True)
    
    out1 = model.forward_standard(X1)
    out1.sum().backward()
    
    out2 = model.forward_checkpointed(X2)
    out2.sum().backward()
    
    # check equality
    assert torch.allclose(out1, out2, atol=1e-5)
    assert torch.allclose(X1.grad, X2.grad, atol=1e-5)
    
    mem_std = measure_peak_memory(model, X1, 'standard', device)
    mem_cp = measure_peak_memory(model, X2, 'checkpointed', device)
    
    if torch.cuda.is_available():
        assert mem_cp <= mem_std
