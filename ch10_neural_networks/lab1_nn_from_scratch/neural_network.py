import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        rng = np.random.default_rng(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(rng.standard_normal((layer_sizes[i], layer_sizes[i+1])) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
            
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
        
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def sigmoid_derivative(x):
        s = NeuralNetwork.sigmoid(x)
        return s * (1 - s)
        
    def forward(self, X):
        self.A = [X]
        self.Z = []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            self.Z.append(z)
            self.A.append(self.relu(z))
            
        z = np.dot(self.A[-1], self.weights[-1]) + self.biases[-1]
        self.Z.append(z)
        self.A.append(self.sigmoid(z))
        return self.A[-1]
        
    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    def backward(self, X, y_true):
        m = X.shape[0]
        self.dW = []
        self.db = []
        
        dz = self.A[-1] - y_true
        self.dW.insert(0, np.dot(self.A[-2].T, dz) / m)
        self.db.insert(0, np.sum(dz, axis=0, keepdims=True) / m)
        
        for i in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[i+1].T)
            dz = da * self.relu_derivative(self.Z[i])
            self.dW.insert(0, np.dot(self.A[i].T, dz) / m)
            self.db.insert(0, np.sum(dz, axis=0, keepdims=True) / m)
            
    def update_weights(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.dW[i]
            self.biases[i] -= lr * self.db[i]
            
    def train(self, X, y, epochs, lr):
        history = []
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            history.append(loss)
            self.backward(X, y)
            self.update_weights(lr)
        return history
        
    def predict(self, X):
        return self.forward(X)

def gradient_check(network, X, y, epsilon=1e-5):
    network.forward(X)
    network.backward(X, y)
    
    for i in range(len(network.weights)):
        for r in range(network.weights[i].shape[0]):
            for c in range(network.weights[i].shape[1]):
                orig = network.weights[i][r, c]
                
                network.weights[i][r, c] = orig + epsilon
                l_plus = network.compute_loss(y, network.forward(X))
                
                network.weights[i][r, c] = orig - epsilon
                l_minus = network.compute_loss(y, network.forward(X))
                
                network.weights[i][r, c] = orig
                
                grad_approx = (l_plus - l_minus) / (2 * epsilon)
                grad_actual = network.dW[i][r, c]
                
                if abs(grad_approx - grad_actual) > 1e-4:
                    return False
    return True

def train_xor():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork([2, 4, 1])
    history = nn.train(X, y, epochs=10000, lr=0.1)
    return nn, history
