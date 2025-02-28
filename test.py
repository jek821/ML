import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the least squares cost function
def least_squares_cost(instances, labels, w):
    return torch.mean((torch.sum(instances * w) - labels) ** 2)

# Define training set
instances = torch.tensor([1.0, 2.0, 3.0])
labels = torch.tensor([4.0, 8.0, 12.0])

# Vectorize loss computation for plotting
w_values = torch.linspace(0, 10, 100)  # Continuous range
loss_values = [least_squares_cost(instances, labels, w.clone().detach()) for w in w_values]

# Plot loss function
plt.plot(w_values.numpy(), loss_values)
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Least Squares Cost Function')
plt.grid()
plt.savefig("loss_plot.png")
print("Loss plot saved as 'loss_plot.png'")

# Initialize parameter for gradient descent
parameters = torch.tensor([3.0], dtype=torch.float32, requires_grad=True)
step_size = 0.01
num_epochs = 1000  # Number of iterations

loss_history = []  # To keep track of loss over iterations

# Gradient descent loop
for epoch in range(num_epochs):
    # Compute the loss
    loss_value = least_squares_cost(instances, labels, parameters)
    loss_history.append(loss_value.item())
    
    # Compute the gradient
    loss_value.backward()
    
    # Update parameters and reset gradients
    with torch.no_grad():
        parameters -= step_size * parameters.grad
        parameters.grad.zero_()
    
    # Optionally, print progress every 100 iterations
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss_value.item()}, Parameter = {parameters.item()}")

# Optionally, plot the loss over epochs to see the convergence
plt.figure()
plt.plot(range(num_epochs), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.grid()
plt.savefig("loss_history.png")
print("Loss history plot saved as 'loss_history.png'")

print(f"Final Parameter: {parameters.item()}")
