import sys, os
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
