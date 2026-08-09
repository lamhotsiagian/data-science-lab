import torch
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
