import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import time

class DeepNetwork(nn.Module):
    def __init__(self, n_layers, hidden_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(n_layers)
        ])
        
    def forward_standard(self, x):
        for block in self.blocks:
            x = block(x)
        return x
        
    def forward_checkpointed(self, x):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return x

def measure_peak_memory(model, X, forward_fn_name, device):
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    X.requires_grad = True
    
    if forward_fn_name == 'standard':
        out = model.forward_standard(X)
    else:
        out = model.forward_checkpointed(X)
        
    loss = out.sum()
    loss.backward()
    
    return torch.cuda.max_memory_allocated() / (1024**2)

def compare_memory(n_layers, hidden_dim, batch_size, device):
    model = DeepNetwork(n_layers, hidden_dim).to(device)
    X = torch.randn(batch_size, hidden_dim, device=device)
    
    mem_std = measure_peak_memory(model, X, 'standard', device)
    mem_cp = measure_peak_memory(model, X, 'checkpointed', device)
    
    return {'standard': mem_std, 'checkpointed': mem_cp}

def compare_speed(n_layers, hidden_dim, batch_size, device):
    model = DeepNetwork(n_layers, hidden_dim).to(device)
    X = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)
    
    t0 = time.time()
    out = model.forward_standard(X)
    out.sum().backward()
    t_std = time.time() - t0
    
    X.grad = None
    t0 = time.time()
    out = model.forward_checkpointed(X)
    out.sum().backward()
    t_cp = time.time() - t0
    
    return {'standard': t_std, 'checkpointed': t_cp}
