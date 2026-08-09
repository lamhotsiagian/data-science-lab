import os
import json

base_dir = "/Users/lamhots/ai/book-project/data-science/data-science-lab"

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def write_notebook(path, title):
    nb = {
     "cells": [
      {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}"]},
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Start coding here\n", "import torch\n", "print(torch.__version__)"]}
     ],
     "metadata": {
      "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.13.9"}
     },
     "nbformat": 4,
     "nbformat_minor": 5
    }
    write_file(path, json.dumps(nb, indent=1))

# ch11 lab1
path_ch11_l1 = os.path.join(base_dir, "ch11_multi_model/lab1_transfer_learning")
write_file(os.path.join(path_ch11_l1, "transfer_learning.py"), """import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def create_synthetic_image_dataset(n_per_class, n_classes, img_size=(3, 32, 32)):
    torch.manual_seed(42)
    n_samples = n_per_class * n_classes
    X = torch.randn(n_samples, *img_size)
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    return TensorDataset(X, y)

def get_pretrained_model(model_name, n_classes, freeze=True):
    if model_name != "resnet18":
        raise ValueError("Only resnet18 is supported")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

def train_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        if val_loader:
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    loss = criterion(out, y)
                    val_loss += loss.item() * X.size(0)
                    preds = out.argmax(dim=1)
                    correct += (preds == y).sum().item()
            val_loss /= len(val_loader.dataset)
            val_acc = correct / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
    return history

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_frozen_vs_unfrozen(train_loader, val_loader, n_classes, epochs, device):
    frozen_model = get_pretrained_model("resnet18", n_classes, freeze=True)
    unfrozen_model = get_pretrained_model("resnet18", n_classes, freeze=False)
    
    print("Training Frozen Model")
    frozen_hist = train_model(frozen_model, train_loader, val_loader, epochs, 0.001, device)
    
    print("Training Unfrozen Model")
    unfrozen_hist = train_model(unfrozen_model, train_loader, val_loader, epochs, 0.001, device)
    
    return {'frozen': frozen_hist, 'unfrozen': unfrozen_hist}
""")

write_file(os.path.join(path_ch11_l1, "tests/test_transfer_learning.py"), """import sys, os
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
from transfer_learning import create_synthetic_image_dataset, get_pretrained_model, count_trainable_params, train_model

def test_transfer_learning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 2
    dataset = create_synthetic_image_dataset(20, n_classes, (3, 32, 32))
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    frozen = get_pretrained_model("resnet18", n_classes, freeze=True)
    unfrozen = get_pretrained_model("resnet18", n_classes, freeze=False)
    
    assert count_trainable_params(frozen) < count_trainable_params(unfrozen)
    
    # Forward pass
    X, y = next(iter(loader))
    out = frozen(X)
    assert out.shape == (10, n_classes)
    
    hist = train_model(frozen, loader, loader, 2, 0.01, device)
    assert len(hist['train_loss']) == 2
""")
write_notebook(os.path.join(path_ch11_l1, "lab1_transfer_learning.ipynb"), "Lab 1: Transfer Learning")
write_file(os.path.join(path_ch11_l1, "README.md"), "# Lab 1: Transfer Learning")

# ch11 lab2
path_ch11_l2 = os.path.join(base_dir, "ch11_multi_model/lab2_multitask_learning")
write_file(os.path.join(path_ch11_l2, "multitask_learning.py"), """import torch
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
""")
write_file(os.path.join(path_ch11_l2, "tests/test_multitask_learning.py"), """import sys, os
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
from multitask_learning import generate_multitask_data, MultitaskNetwork, train_multitask

def test_multitask():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y_reg, y_cls = generate_multitask_data(100)
    model = MultitaskNetwork(X.shape[1])
    
    # check shapes
    out_reg, out_cls = model(X)
    assert out_reg.shape == (100, 1)
    assert out_cls.shape == (100, 1)
    
    loader = DataLoader(TensorDataset(X, y_reg, y_cls), batch_size=16)
    hist = train_multitask(model, loader, 2, 0.01, (1.0, 1.0), device)
    
    assert len(hist['loss_reg']) == 2
    assert len(hist['loss_cls']) == 2
    assert hist['loss_reg'][1] < hist['loss_reg'][0] or hist['loss_cls'][1] < hist['loss_cls'][0]
""")
write_notebook(os.path.join(path_ch11_l2, "lab2_multitask_learning.ipynb"), "Lab 2: Multitask Learning")
write_file(os.path.join(path_ch11_l2, "README.md"), "# Lab 2: Multitask Learning")

# ch11 lab3
path_ch11_l3 = os.path.join(base_dir, "ch11_multi_model/lab3_federated_learning")
write_file(os.path.join(path_ch11_l3, "federated_learning.py"), """import torch
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
""")
write_file(os.path.join(path_ch11_l3, "tests/test_federated_learning.py"), """import sys, os
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
from federated_learning import partition_data_iid, run_federated, fedavg_aggregate, create_simple_model

def test_federated():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    partitions = partition_data_iid(X, y, 5)
    assert len(partitions) == 5
    assert sum(len(p[0]) for p in partitions) == 100
    
    accs = run_federated(X, y, n_clients=2, n_rounds=2, local_epochs=1, lr=0.1, iid=True, device=device)
    assert len(accs) == 2
""")
write_notebook(os.path.join(path_ch11_l3, "lab3_federated_learning.ipynb"), "Lab 3: Federated Learning")
write_file(os.path.join(path_ch11_l3, "README.md"), "# Lab 3: Federated Learning")

# ch12 lab1
path_ch12_l1 = os.path.join(base_dir, "ch12_dl_optimization/lab1_mixed_precision")
write_file(os.path.join(path_ch12_l1, "mixed_precision.py"), """import torch
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
""")
write_file(os.path.join(path_ch12_l1, "tests/test_mixed_precision.py"), """import sys, os
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
from mixed_precision import create_model, generate_data, train_fp32, train_mixed_precision

def test_mixed_precision():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = generate_data(100, 10, 1)
    loader = DataLoader(dataset, batch_size=16)
    
    m1 = create_model(10, 16, 2, 1)
    res1 = train_fp32(m1, loader, 2, 0.01, device)
    assert len(res1['loss']) == 2
    
    m2 = create_model(10, 16, 2, 1)
    res2 = train_mixed_precision(m2, loader, 2, 0.01, device)
    assert len(res2['loss']) == 2
    
    X, _ = next(iter(loader))
    assert m1(X.to(device)).shape == (16, 1)
""")
write_notebook(os.path.join(path_ch12_l1, "lab1_mixed_precision.ipynb"), "Lab 1: Mixed Precision")
write_file(os.path.join(path_ch12_l1, "README.md"), "# Lab 1: Mixed Precision")

# ch12 lab2
path_ch12_l2 = os.path.join(base_dir, "ch12_dl_optimization/lab2_gradient_checkpointing")
write_file(os.path.join(path_ch12_l2, "gradient_checkpointing.py"), """import torch
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
""")
write_file(os.path.join(path_ch12_l2, "tests/test_gradient_checkpointing.py"), """import sys, os
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
""")
write_notebook(os.path.join(path_ch12_l2, "lab2_gradient_checkpointing.ipynb"), "Lab 2: Gradient Checkpointing")
write_file(os.path.join(path_ch12_l2, "README.md"), "# Lab 2: Gradient Checkpointing")

# ch12 lab3
path_ch12_l3 = os.path.join(base_dir, "ch12_dl_optimization/lab3_gradient_accumulation")
write_file(os.path.join(path_ch12_l3, "gradient_accumulation.py"), """import torch
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
""")
write_file(os.path.join(path_ch12_l3, "tests/test_gradient_accumulation.py"), """import sys, os
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
from gradient_accumulation import create_model, generate_data, train_standard, train_with_accumulation

def test_gradient_accumulation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = generate_data(100, 10)
    loader = DataLoader(dataset, batch_size=8)
    
    model = create_model(10, 2)
    losses_std = train_standard(model, loader, 2, 0.01, device)
    assert len(losses_std) == 2
    
    model2 = create_model(10, 2)
    losses_acc = train_with_accumulation(model2, loader, 2, 0.01, 2, device)
    assert len(losses_acc) == 2
    assert losses_acc[1] < losses_acc[0]
""")
write_notebook(os.path.join(path_ch12_l3, "lab3_gradient_accumulation.ipynb"), "Lab 3: Gradient Accumulation")
write_file(os.path.join(path_ch12_l3, "README.md"), "# Lab 3: Gradient Accumulation")

print("All labs generated successfully!")
