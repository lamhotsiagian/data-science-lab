import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

def create_model(input_dim, hidden_dim, n_layers, output_dim):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

def generate_data(n_samples, input_dim, output_dim):
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    w = torch.randn(input_dim, output_dim)
    y = X @ w + torch.randn(n_samples, output_dim) * 0.1
    return TensorDataset(X, y)

def train_fp32(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    metrics = {'loss': [], 'time': []}
    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        metrics['loss'].append(epoch_loss / len(dataloader))
        metrics['time'].append(time.time() - t0)
    return metrics

def train_mixed_precision(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda' if 'cuda' in str(device) else 'cpu', enabled=torch.cuda.is_available())
    
    metrics = {'loss': [], 'time': []}
    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if 'cuda' in str(device) else 'cpu', enabled=torch.cuda.is_available()):
                out = model(X)
                loss = criterion(out, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        metrics['loss'].append(epoch_loss / len(dataloader))
        metrics['time'].append(time.time() - t0)
    return metrics

def compare_precision(input_dim, hidden_dim, n_layers, n_samples, epochs, device):
    dataset = generate_data(n_samples, input_dim, 1)
    loader = DataLoader(dataset, batch_size=32)
    
    m1 = create_model(input_dim, hidden_dim, n_layers, 1)
    fp32_res = train_fp32(m1, loader, epochs, 0.01, device)
    
    m2 = create_model(input_dim, hidden_dim, n_layers, 1)
    amp_res = train_mixed_precision(m2, loader, epochs, 0.01, device)
    
    return {'fp32': fp32_res, 'amp': amp_res}

def measure_memory(model, batch, device):
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    X, y = batch
    X, y = X.to(device), y.to(device)
    out = model(X)
    loss = out.sum()
    loss.backward()
    
    return torch.cuda.max_memory_allocated() / (1024**2)
