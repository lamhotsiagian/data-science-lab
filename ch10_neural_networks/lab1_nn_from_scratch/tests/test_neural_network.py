import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from neural_network import NeuralNetwork, gradient_check, train_xor

def test_gradient_check():
    X = np.random.randn(5, 3)
    y = np.random.randint(0, 2, (5, 1))
    nn = NeuralNetwork([3, 4, 1])
    assert gradient_check(nn, X, y)
    
def test_train_xor():
    nn, history = train_xor()
    assert history[-1] < 0.1
    
def test_shapes_and_probs():
    X = np.random.randn(10, 5)
    nn = NeuralNetwork([5, 8, 2, 1])
    probs = nn.forward(X)
    assert probs.shape == (10, 1)
    assert np.all((probs >= 0) & (probs <= 1))
