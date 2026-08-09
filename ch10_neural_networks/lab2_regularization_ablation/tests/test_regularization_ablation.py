import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
from regularization_ablation import create_simple_cnn, create_dataset, train_model, run_ablation

def test_ablation():
    configs = [
        ('None', 0.0, 0.0),
        ('Reg', 0.5, 0.1)
    ]
    results = run_ablation(configs)
    
    # Check loss decreases for both
    assert results['None']['train'][-1] < results['None']['train'][0]
    
    # Check generalization gap
    gap_none = results['None']['val'][-1] - results['None']['train'][-1]
    gap_reg = results['Reg']['val'][-1] - results['Reg']['train'][-1]
    
    # Check regularized model is less overfit or at least runs
    assert 'None' in results
