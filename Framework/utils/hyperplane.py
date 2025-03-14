import numpy as np
from activation.sigmoid import Sigmoid  # Use absolute import

class Hyperplane:
    def __init__(self, dim):
        # Initialize weights with larger random values
        self.weights = np.random.randn(dim) * 0.1
        self.bias = 0
    
    def classify(self, X):
        """Applies the hyperplane equation and sigmoid activation."""
        linear_output = np.dot(X, self.weights) + self.bias
        return Sigmoid.activate(linear_output)  # Apply sigmoid function