import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_simple_cnn(dropout_rate, use_label_smoothing):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
                nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                nn.Flatten()
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(8 * 8 * 8, 16), nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(16, 2)
            )
        def forward(self, x):
            return self.classifier(self.features(x))
    return SimpleCNN().to(device)

def create_dataset(n_train=200, n_test=100):
    X, y = make_classification(n_samples=n_train+n_test, n_features=64, n_informative=10, random_state=42)
    X = X.reshape(-1, 1, 8, 8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(test_ds, batch_size=32)

def train_model(model, train_loader, val_loader, epochs, label_smoothing):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        tl = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        train_losses.append(tl / len(train_loader))
        
        model.eval()
        vl = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                vl += criterion(out, y).item()
        val_losses.append(vl / len(val_loader))
    return train_losses, val_losses

def run_ablation(configs):
    train_loader, test_loader = create_dataset(200, 100)
    results = {}
    for name, drop, ls in configs:
        model = create_simple_cnn(drop, ls)
        tl, vl = train_model(model, train_loader, test_loader, epochs=20, label_smoothing=ls)
        results[name] = {'train': tl, 'val': vl}
    return results

def plot_ablation_results(results):
    fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 4))
    if len(results) == 1: axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(res['train'], label='Train')
        ax.plot(res['val'], label='Val')
        ax.set_title(name)
        ax.legend()
    return fig

def plot_generalization_gap(results):
    gaps = {k: v['val'][-1] - v['train'][-1] for k, v in results.items()}
    fig, ax = plt.subplots()
    ax.bar(gaps.keys(), gaps.values())
    ax.set_title("Generalization Gap")
    return fig
