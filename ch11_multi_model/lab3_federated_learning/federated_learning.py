import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np

def create_simple_model(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

def partition_data_iid(X, y, n_clients):
    indices = torch.randperm(len(X))
    X, y = X[indices], y[indices]
    splits_X = torch.chunk(X, n_clients)
    splits_y = torch.chunk(y, n_clients)
    return [(splits_X[i], splits_y[i]) for i in range(n_clients)]

def partition_data_non_iid(X, y, n_clients, shards_per_client):
    # simple sort by class
    indices = torch.argsort(y)
    X, y = X[indices], y[indices]
    n_shards = n_clients * shards_per_client
    shards_X = torch.chunk(X, n_shards)
    shards_y = torch.chunk(y, n_shards)
    
    client_data = []
    shard_idx = 0
    for i in range(n_clients):
        cx = torch.cat(shards_X[shard_idx:shard_idx+shards_per_client])
        cy = torch.cat(shards_y[shard_idx:shard_idx+shards_per_client])
        client_data.append((cx, cy))
        shard_idx += shards_per_client
    return client_data

def train_local(model, dataloader, epochs, lr, device):
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())

def fedavg_aggregate(global_model, client_state_dicts, client_sizes):
    total_size = sum(client_sizes)
    weights = [size / total_size for size in client_sizes]
    
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_dict[k] * weight for client_dict, weight in zip(client_state_dicts, weights)], dim=0).sum(dim=0)
    
    global_model.load_state_dict(global_dict)
    return global_model

def federated_training_round(global_model, client_dataloaders, epochs, lr, device):
    client_state_dicts = []
    client_sizes = []
    for loader in client_dataloaders:
        local_model = copy.deepcopy(global_model)
        state_dict = train_local(local_model, loader, epochs, lr, device)
        client_state_dicts.append(state_dict)
        client_sizes.append(len(loader.dataset))
        
    global_model = fedavg_aggregate(global_model, client_state_dicts, client_sizes)
    return global_model

def run_federated(X, y, n_clients, n_rounds, local_epochs, lr, iid, device):
    if iid:
        client_data = partition_data_iid(X, y, n_clients)
    else:
        client_data = partition_data_non_iid(X, y, n_clients, 2)
        
    client_dataloaders = [DataLoader(TensorDataset(cx, cy), batch_size=16, shuffle=True) for cx, cy in client_data]
    
    global_model = create_simple_model(X.shape[1], 16, len(torch.unique(y))).to(device)
    
    accs = []
    for r in range(n_rounds):
        global_model = federated_training_round(global_model, client_dataloaders, local_epochs, lr, device)
        
        global_model.eval()
        with torch.no_grad():
            out = global_model(X.to(device))
            preds = out.argmax(dim=1)
            acc = (preds == y.to(device)).float().mean().item()
        accs.append(acc)
    return accs

def train_centralized(X, y, epochs, lr, device):
    model = create_simple_model(X.shape[1], 16, len(torch.unique(y))).to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out = model(X.to(device))
        acc = (out.argmax(dim=1) == y.to(device)).float().mean().item()
    return acc

def compare_federated_vs_centralized(X, y, device):
    fed_acc = run_federated(X, y, 5, 5, 2, 0.1, True, device)
    cent_acc = train_centralized(X, y, 10, 0.1, device)
    return {'federated': fed_acc, 'centralized': cent_acc}
