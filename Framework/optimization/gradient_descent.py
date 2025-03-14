import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def compute_gradient(self, y_true, y_pred, X):
        """Computes the gradient of cross-entropy loss w.r.t. the hyperplane weights and bias."""
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Prevent division by zero
        gradient_w = np.dot(X.T, (y_pred - y_true)) / len(y_true)
        gradient_b = np.mean(y_pred - y_true)
        return gradient_w, gradient_b
    
    def update_hyperplane(self, hyperplane, X, y_true, y_pred):
        """Updates the hyperplane's weights and bias using gradient descent."""
        gradient_w, gradient_b = self.compute_gradient(y_true, y_pred, X)

        # Print gradient values for debugging
        print(f"Gradient W: {gradient_w}, Gradient B: {gradient_b}")

        # Apply gradient update
        hyperplane.weights -= self.learning_rate * gradient_w
        hyperplane.bias -= self.learning_rate * gradient_b

        return hyperplane