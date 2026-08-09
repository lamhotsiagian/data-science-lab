import sys, os
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
