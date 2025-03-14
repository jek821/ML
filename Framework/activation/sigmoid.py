import numpy as np

class Sigmoid:
    @staticmethod
    def activate(x):
        """Applies the sigmoid function element-wise."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        """Computes the derivative of the sigmoid function."""
        sig = Sigmoid.activate(x)
        return sig * (1 - sig)