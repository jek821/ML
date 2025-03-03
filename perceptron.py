import numpy as np
import torch
from torch import tensor
import matplotlib.pyplot as plt
import time 

class perceptron:
    def __init__(self, learning_rate=0.01):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.finalAccruacy = None 
        self.finalEpoch = None
        self.totalTime = None 
        
    def train(self, data_generator, epochs=100):
        # Set start time
        start_time = time.time()

        # Initialize weights and bias
        self.w = torch.zeros(2, dtype=torch.float32)
        self.b = torch.zeros(1, dtype=torch.float32)
        
        x = data_generator.x
        y = data_generator.labels
        n_samples = x.shape[0]
        
        for epoch in range(epochs):
            mistakes = 0
            
            # Process each data point
            for i in range(n_samples):
                # Make prediction
                prediction = 1.0 if torch.dot(self.w, x[i]) + self.b > 0 else -1.0
                
                # Update weights if prediction is wrong
                if prediction != y[i]:
                    mistakes += 1
                    self.w += self.lr * y[i] * x[i]
                    self.b += self.lr * y[i]
            
            # Print progress
            accuracy = 1 - (mistakes / n_samples)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
                
            # If perfect classification, we can stop early
            if mistakes == 0:
                print(f"Converged at epoch {epoch + 1}")
                # get end time
                end_time = time.time()
                # calculate total time
                self.totalTime = end_time - start_time
                self.finalAccruacy = accuracy
                self.finalEpoch = epoch + 1
                break
    
    def plot_results(self, data_generator):
        # Plot the data points
        plt.figure(figsize=(8, 6))
        plt.scatter(data_generator.x[data_generator.labels == 1][:, 0].detach().numpy(),
                   data_generator.x[data_generator.labels == 1][:, 1].detach().numpy(),
                   color='blue', label='Class 1')
        plt.scatter(data_generator.x[data_generator.labels == -1][:, 0].detach().numpy(),
                   data_generator.x[data_generator.labels == -1][:, 1].detach().numpy(),
                   color='red', label='Class -1')
        
        # Plot the original decision boundary
        x_line = np.linspace(0, 1, 100)
        y_true = (-data_generator.w[0].detach().numpy() * x_line - data_generator.b.detach().numpy()) / data_generator.w[1].detach().numpy()
        plt.plot(x_line, y_true, 'g--', label='True Boundary')
        
        # Plot the learned decision boundary
        if self.w[1] != 0:  # Avoid division by zero
            y_learned = (-self.w[0].detach().numpy() * x_line - self.b.detach().numpy()) / self.w[1].detach().numpy()
            plt.plot(x_line, y_learned, 'k-', label='Learned Boundary')
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.title("Perceptron Results")
        plt.savefig('perceptron_simple.png')
        #plt.show()