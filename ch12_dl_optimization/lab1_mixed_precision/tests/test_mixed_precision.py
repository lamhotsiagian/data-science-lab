import sys, os
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
