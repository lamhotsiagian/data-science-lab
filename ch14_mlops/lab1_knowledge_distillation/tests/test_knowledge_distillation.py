import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.data import TensorDataset, DataLoader
from knowledge_distillation import *

def test_knowledge_distillation():
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=10)
    
    res = compare_distillation(dl, dl, 10, 2, 2, 'cpu')
    assert res['teacher_params'] > res['student_params']
    
    s_logits = torch.tensor([[1.0, 0.0]])
    t_logits = torch.tensor([[2.0, -2.0]])
    labels = torch.tensor([0])
    loss = distillation_loss(s_logits, t_logits, labels, 2.0, 0.5)
    assert loss.item() > 0
    
    fig = plot_distillation_comparison(res)
    assert fig is not None
