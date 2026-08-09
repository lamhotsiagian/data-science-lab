import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MultitaskNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head_reg = nn.Linear(hidden_dim, 1)
        self.head_cls = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        feat = self.shared(x)
        return self.head_reg(feat), self.head_cls(feat)

def generate_multitask_data(n_samples, input_dim=10):
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    w_reg = torch.randn(input_dim, 1)
    w_cls = torch.randn(input_dim, 1)
    
    y_reg = X @ w_reg + torch.randn(n_samples, 1) * 0.1
    y_cls = ((X @ w_cls) > 0).float()
    return X, y_reg, y_cls

def train_multitask(model, dataloader, epochs, lr, loss_weights, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    w_reg, w_cls = loss_weights
    
    history = {'loss_reg': [], 'loss_cls': []}
    for epoch in range(epochs):
        model.train()
        epoch_loss_reg = 0
        epoch_loss_cls = 0
        for X, y_reg, y_cls in dataloader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            optimizer.zero_grad()
            out_reg, out_cls = model(X)
            
            loss_reg = mse_criterion(out_reg, y_reg)
            loss_cls = bce_criterion(out_cls, y_cls)
            loss = w_reg * loss_reg + w_cls * loss_cls
            loss.backward()
            optimizer.step()
            
            epoch_loss_reg += loss_reg.item()
            epoch_loss_cls += loss_cls.item()
            
        history['loss_reg'].append(epoch_loss_reg / len(dataloader))
        history['loss_cls'].append(epoch_loss_cls / len(dataloader))
    return history

def train_single_task(X, y, task_type, epochs, lr, device):
    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if task_type == 'regression' else nn.BCEWithLogitsLoss()
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16)
    
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return model

def compare_mtl_vs_single(X, y_reg, y_cls, epochs, device):
    dataset = TensorDataset(X, y_reg, y_cls)
    loader = DataLoader(dataset, batch_size=16)
    
    mtl_model = MultitaskNetwork(X.shape[1])
    mtl_hist = train_multitask(mtl_model, loader, epochs, 0.01, (1.0, 1.0), device)
    
    reg_model = train_single_task(X, y_reg, 'regression', epochs, 0.01, device)
    cls_model = train_single_task(X, y_cls, 'classification', epochs, 0.01, device)
    
    return {'mtl': mtl_hist}
