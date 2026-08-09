import sys, os
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
