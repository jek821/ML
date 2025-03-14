import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        """Initialize a simple Graph utility class."""
        pass
        
    def plot_and_save(self, data, hyperplane, save_path="plot.png"):
        """
        Plot the data points and hyperplane, then save to a file.
        
        Parameters:
        -----------
        data : Data object
            Contains X (features) and y (labels)
        hyperplane : Hyperplane object
            Contains weights and bias for the decision boundary
        save_path : str
            Path where to save the resulting plot
        """
        # Only works for 2D data
        if data.X.shape[1] != 2:
            raise ValueError("This visualization only works for 2D data")
            
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot the data points
        # Class 0 in red, Class 1 in blue
        plt.scatter(data.X[data.y == 0, 0], data.X[data.y == 0, 1], color='red', label='Class 0')
        plt.scatter(data.X[data.y == 1, 0], data.X[data.y == 1, 1], color='blue', label='Class 1')
        
        # Plot the hyperplane
        # For a hyperplane w₁x₁ + w₂x₂ + b = 0, we can solve for x₂:
        # x₂ = (-w₁x₁ - b) / w₂
        
        # Get min and max for x1 (with some padding)
        x1_min, x1_max = data.X[:, 0].min() - 1, data.X[:, 0].max() + 1
        
        # Generate x1 points along the range
        x1_points = np.array([x1_min, x1_max])
        
        # Calculate corresponding x2 points for the hyperplane line
        # From w₁x₁ + w₂x₂ + b = 0 → x₂ = (-w₁x₁ - b) / w₂
        w1, w2 = hyperplane.weights
        b = hyperplane.bias
        
        # Avoid division by zero
        if abs(w2) < 1e-10:
            # If w2 is almost zero, the line is vertical
            x2_min, x2_max = data.X[:, 1].min() - 1, data.X[:, 1].max() + 1
            plt.axvline(x=-b/w1, color='black', linestyle='--', label='Decision Boundary')
        else:
            # Calculate x2 points
            x2_points = (-w1 * x1_points - b) / w2
            # Plot the hyperplane
            plt.plot(x1_points, x2_points, 'k--', label='Decision Boundary')
        
        # Add labels and legend
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Data Classification with Hyperplane')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.savefig(save_path)
        plt.close()
        
        print(f"Plot saved to {save_path}")