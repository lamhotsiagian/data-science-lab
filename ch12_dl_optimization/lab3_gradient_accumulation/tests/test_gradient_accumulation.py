import sys, os
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
