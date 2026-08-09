import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_simple_model(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    ).to(device)

def generate_labeled_data(n_samples, n_features, n_classes):
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    # Add signal
    X += y.unsqueeze(1) * 0.5
    return X, y

def create_sorted_dataloader(X, y, batch_size):
    sorted_idx = torch.argsort(y)
    ds = TensorDataset(X[sorted_idx], y[sorted_idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def create_shuffled_dataloader(X, y, batch_size):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def train_with_tracking(model, dataloader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    batch_losses = []
    epoch_losses = []
    grad_norms = []
    
    for epoch in range(epochs):
        epoch_l = 0
        for X_b, y_b in dataloader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            
            # Record grad norm
            norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(norm ** 0.5)
            
            optimizer.step()
            batch_losses.append(loss.item())
            epoch_l += loss.item()
        epoch_losses.append(epoch_l / len(dataloader))
        
    return batch_losses, epoch_losses, grad_norms

def compare_orderings(X, y, epochs, batch_size, lr):
    model1 = create_simple_model(X.shape[1], 16, len(torch.unique(y)))
    model2 = create_simple_model(X.shape[1], 16, len(torch.unique(y)))
    model2.load_state_dict(model1.state_dict())
    
    dl_sorted = create_sorted_dataloader(X, y, batch_size)
    dl_shuf = create_shuffled_dataloader(X, y, batch_size)
    
    _, ep_sorted, _ = train_with_tracking(model1, dl_sorted, epochs, lr)
    _, ep_shuf, _ = train_with_tracking(model2, dl_shuf, epochs, lr)
    
    return ep_sorted, ep_shuf

def compare_batch_sizes(X, y, epochs, batch_sizes, lr):
    res = {}
    for bs in batch_sizes:
        m = create_simple_model(X.shape[1], 16, len(torch.unique(y)))
        dl = create_shuffled_dataloader(X, y, bs)
        _, ep_l, _ = train_with_tracking(m, dl, epochs, lr)
        res[bs] = ep_l
    return res

def plot_ordering_comparison(sorted_losses, shuffled_losses):
    fig, ax = plt.subplots()
    ax.plot(sorted_losses, label='Sorted')
    ax.plot(shuffled_losses, label='Shuffled')
    ax.legend()
    return fig

def plot_batch_size_comparison(results):
    fig, ax = plt.subplots()
    for bs, loss in results.items():
        ax.plot(loss, label=f'BS={bs}')
    ax.legend()
    return fig

def plot_gradient_norms(gradient_norms):
    fig, ax = plt.subplots()
    ax.plot(gradient_norms)
    ax.set_title("Gradient Norms")
    return fig
