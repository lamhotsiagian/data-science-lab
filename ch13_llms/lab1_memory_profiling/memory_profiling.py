import torch
import torch.nn as nn

class MemoryBreakdown:
    def __init__(self, params, gradients, optimizer_states, activations):
        self.params = params
        self.gradients = gradients
        self.optimizer_states = optimizer_states
        self.activations = activations
        self.total = params + gradients + optimizer_states + activations
        
    def to_table(self):
        return (f"Params: {self.params} bytes\n"
                f"Gradients: {self.gradients} bytes\n"
                f"Optimizer: {self.optimizer_states} bytes\n"
                f"Activations: {self.activations} bytes\n"
                f"Total: {self.total} bytes")

def calculate_parameter_memory(n_params, dtype_bytes=4):
    return n_params * dtype_bytes

def calculate_gradient_memory(n_params, dtype_bytes=4):
    return n_params * dtype_bytes

def calculate_optimizer_memory(n_params, optimizer_type="adam"):
    if optimizer_type.lower() == "adam":
        return n_params * 8
    elif optimizer_type.lower() == "sgd":
        return n_params * 4
    return 0

def calculate_activation_memory(batch_size, seq_len, hidden_dim, n_layers, n_heads):
    return batch_size * seq_len * hidden_dim * n_layers * 4 * 2

def total_training_memory(n_params, batch_size, seq_len, hidden_dim, n_layers, n_heads, dtype="fp32", optimizer="adam"):
    dtype_bytes = 4 if dtype == "fp32" else 2
    params = calculate_parameter_memory(n_params, dtype_bytes)
    grads = calculate_gradient_memory(n_params, dtype_bytes)
    opt = calculate_optimizer_memory(n_params, optimizer)
    acts = calculate_activation_memory(batch_size, seq_len, hidden_dim, n_layers, n_heads)
    return MemoryBreakdown(params, grads, opt, acts)

def create_small_transformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, max_seq_len=128):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True), num_layers=n_layers)

def profile_model_empirically(model):
    n_params = sum(p.numel() for p in model.parameters())
    dtype_bytes = 4
    mem = calculate_parameter_memory(n_params, dtype_bytes)
    return {"n_params": n_params, "param_memory_bytes": mem}

def compare_theoretical_vs_empirical(model, d_model=128, n_heads=4, n_layers=2):
    empirical = profile_model_empirically(model)
    theoretical_params = d_model * d_model * 4 * n_layers
    return {"empirical": empirical, "theoretical_params": theoretical_params}
