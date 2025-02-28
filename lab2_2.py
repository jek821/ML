import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the least squares cost function
def least_squares_cost(instances, labels, w):
    return torch.mean((torch.sum(instances * w) - labels) ** 2)  # Fixed shape issue

# Define training set
instances = torch.tensor([1.0, 2.0, 3.0])
labels = torch.tensor([4.0, 8.0, 12.0])

# Vectorize loss computation
w_values = torch.linspace(0, 10, 100)  # Continuous range
loss_values = [least_squares_cost(instances, labels, w.clone().detach()) for w in w_values]

# Plot loss function
plt.plot(w_values.numpy(), loss_values)
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Least Squares Cost Function')
plt.grid()
plt.savefig("loss_plot.png")  # Save plot instead of showing it
print("Loss plot saved as 'loss_plot.png'")

# Gradient computation
parameters = torch.tensor([3.0], dtype=torch.float32, requires_grad=True)
step_size = 0.01

# Compute loss and gradient
loss_value = least_squares_cost(instances, labels, parameters)
loss_value.backward()
print(f"Loss: {loss_value.item()}, Grad: {parameters.grad.item()}")

# Update parameters
with torch.no_grad():
    parameters -= step_size * parameters.grad
    parameters.grad.zero_()  # Reset gradients
print(f"Updated Parameters: {parameters.item()}")
