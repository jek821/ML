import numpy as np

class Model:
    def __init__(self, hyperplane, loss_function, optimizer):
        self.hyperplane = hyperplane
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def train(self, data, learning_rate, epochs):
        # Trains the model using the provided data, optimizer, and loss function.
        # In train method of model.py
        X_train, X_test, y_train, y_test = data.partition_data()
        self.optimizer.learning_rate = learning_rate
        
        for epoch in range(epochs):
            y_pred = self.hyperplane.classify(X_train)  
            loss = self.loss_function.compute(y_train, y_pred)

            # Convert predictions to binary (0 or 1) for accuracy calculation
            y_pred_binary = (y_pred > 0.5).astype(int)
            train_accuracy = np.mean(y_pred_binary == y_train)

            # Update hyperplane weights and bias
            self.hyperplane = self.optimizer.update_hyperplane(self.hyperplane, X_train, y_train, y_pred)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Weights: {self.hyperplane.weights}, Bias: {self.hyperplane.bias}")
    
    def test(self, data):
        """Tests the trained model on test data and computes accuracy."""
        _, X_test, _, y_test = data.partition_data()

        # Get predictions (sigmoid output, values between 0 and 1)
        y_pred = self.hyperplane.classify(X_test)

        # Convert predictions to binary (0 or 1)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Compute accuracy
        accuracy = np.mean(y_pred_binary == y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy