import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

# Create a directory for plots if it doesn't exist
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Function to generate linearly separable data
def generate_plot_linearly_separable_data(num_samples=100, separation=4):
    np.random.seed(42)

    # Generate points for class 0
    x0 = np.random.randn(num_samples, 2) + np.array([-separation / 2, -separation / 2])
    y0 = np.zeros(num_samples)

    # Generate points for class 1
    x1 = np.random.randn(num_samples, 2) + np.array([separation / 2, separation / 2])
    y1 = np.ones(num_samples)

    # Combine the data
    X = np.vstack((x0, x1))
    y = np.hstack((y0, y1))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

# Logistic regression function
def logistic_regression(X, W, b):
    """Computes predictions using logistic regression."""
    z = X @ W + b  # Linear function
    return 1 / (1 + torch.exp(-z))  # Sigmoid activation

# Least squares loss function
def least_squares_loss(y_pred, y_true):
    """Manually computes least squares loss."""
    return torch.mean((y_pred - y_true) ** 2)

# Function to compute gradients manually
def compute_gradients(X, y, W, b):
    """Manually computes gradients of W and b for least squares loss."""
    N = X.shape[0]

    y_pred = logistic_regression(X, W, b)  # Predicted outputs

    # Compute gradient w.r.t loss
    dL_dy = 2 * (y_pred - y) / N  # (N,) shape

    # Reshape dL_dy to be (N,1) so that matrix multiplication works
    dL_dy = dL_dy.unsqueeze(1)  # Now shape (N,1)

    # Compute gradient w.r.t W (Now this should work properly)
    dL_dW = (X.T @ dL_dy).squeeze()  # (2,)

    # Compute gradient w.r.t b
    dL_db = dL_dy.sum()

    return dL_dW, dL_db


# Function to plot and save decision boundary
def plot_decision_boundary(X, y, W, b, epoch):
    """Plots and saves the decision boundary image."""
    plt.figure(figsize=(8, 6))

    # Scatter plot of data
    X_np = X.detach().numpy()
    y_np = y.detach().numpy()
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm')

    # Compute decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(W[0].item() * x_vals + b.item()) / W[1].item()

    plt.plot(x_vals, y_vals, 'k-', label=f"Epoch {epoch}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary at Epoch {epoch}')
    plt.legend()
    plt.grid(True)

    # Save the plot instead of showing it
    filename = os.path.join(PLOT_DIR, f"epoch_{epoch:03d}.png")
    plt.savefig(filename)
    plt.close()  # Close to prevent memory issues

    print(f"Saved plot: {filename}")

# Training function using gradient descent
def train_logistic_regression(X, y, lr=0.1, epochs=100):
    """Trains logistic regression using least squares loss and manual gradient descent."""
    torch.manual_seed(42)
    W = torch.randn(2, dtype=torch.float32)  # Two weights
    b = torch.zeros(1, dtype=torch.float32)  # Bias

    for epoch in range(epochs):
        dL_dW, dL_db = compute_gradients(X, y, W, b)

        # Update parameters using gradient descent
        W -= lr * dL_dW
        b -= lr * dL_db

        # Compute loss
        y_pred = logistic_regression(X, W, b)
        loss = least_squares_loss(y_pred, y)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            plot_decision_boundary(X, y, W, b, epoch)  # Save image

    create_gif()  # Generate a GIF after training
    return W, b

# Function to create a GIF from saved plots
def create_gif():
    """Creates a GIF from saved plots."""
    images = []
    filenames = sorted([f for f in os.listdir(PLOT_DIR) if f.endswith(".png")])

    for filename in filenames:
        file_path = os.path.join(PLOT_DIR, filename)
        images.append(imageio.imread(file_path))

    gif_path = os.path.join(PLOT_DIR, "decision_boundary_evolution.gif")
    imageio.mimsave(gif_path, images, duration=0.5)  # 0.5s per frame

    print(f"Saved GIF: {gif_path}")

# Run the script
if __name__ == "__main__":
    # Generate dataset
    X, y = generate_plot_linearly_separable_data()

    # Train model
    W, b = train_logistic_regression(X, y, lr=0.1, epochs=100)

    print("Training completed. Check the 'plots/' folder for images and the GIF.")
