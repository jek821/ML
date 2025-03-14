from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def compute_gradient(self, y_true, y_pred, X):
        #Computes the gradient of the loss function.
        pass
    
    @abstractmethod
    def update_hyperplane(self, hyperplane, X, y_true, y_pred):
        #Updates the hyperplane based on the computed gradient.
        pass