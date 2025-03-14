from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def compute(self, y_true, y_pred):
        """Computes the loss given true and predicted values."""
        pass