import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def create_model(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    )

def generate_data(n_samples, input_dim):
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    w = torch.randn(input_dim, 2)
    logits = X @ w
    y = torch.argmax(logits, dim=1)
    return TensorDataset(X, y)

def train_standard(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
    return losses

def train_with_accumulation(model, dataloader, epochs, lr, accumulation_steps, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * accumulation_steps
        losses.append(epoch_loss / len(dataloader))
    return losses

def compare_accumulation_steps(input_dim, n_samples, physical_batch_size, accumulation_configs, epochs, lr, device):
    dataset = generate_data(n_samples, input_dim)
    loader = DataLoader(dataset, batch_size=physical_batch_size)
    
    results = {}
    for acc_steps in accumulation_configs:
        model = create_model(input_dim, 2)
        if acc_steps == 1:
            losses = train_standard(model, loader, epochs, lr, device)
        else:
            losses = train_with_accumulation(model, loader, epochs, lr, acc_steps, device)
        results[acc_steps] = losses
    return results

def plot_accumulation_comparison(results):
    for steps, losses in results.items():
        plt.plot(losses, label=f"Accumulation: {steps}")
    plt.legend()
    plt.title("Training Loss")
    
def plot_effective_batch_analysis(results):
    plot_accumulation_comparison(results)
