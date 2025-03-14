import numpy as np

class CrossEntropyLoss:
    @staticmethod
    def compute(y_true, y_pred):
        # Computes the cross-entropy loss between true labels and predictions.
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Prevent log(0)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss