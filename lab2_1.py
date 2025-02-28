import torch
from torch import tensor 

# Define a simple loss function that calculates the squared difference from 2
def silly_loss(w):
    return (w - 2) ** 2

if __name__ == '__main__':
    # Initialize parameters with a starting value of 3.0
    parameters_init = [3.0]
    # Convert the initial parameters to a tensor with gradient tracking enabled
    parameters = tensor(parameters_init, dtype=torch.float32, requires_grad=True)

    # Define the step size for the gradient descent update
    step_size = 0.01

    # Print the initial parameters
    print("parameters = ", parameters)

    # Calculate the loss value using the silly_loss function
    loss_value = silly_loss(parameters)
    print('loss value = ', loss_value)

    # Perform backpropagation to compute the gradient of the loss with respect to the parameters
    loss_value.backward()
    # Print the gradient of the parameters
    print("grad = ", parameters.grad)

    # Update the parameters using gradient descent
    with torch.no_grad():
        parameters.data = parameters - step_size * parameters.grad
    print("parameters after one step = ", parameters)

    # Calculate the loss value again using the updated parameters
    loss_value = silly_loss(parameters)
    print('loss value = ', loss_value)

    # Perform backpropagation again to compute the new gradient of the loss with respect to the updated parameters
    loss_value.backward()
    # Print the new gradient of the parameters
    print("grad = ", parameters.grad)

    # Update the parameters again using gradient descent
    with torch.no_grad():
        parameters.data = parameters - step_size * parameters.grad

    # Print the final updated parameters after two steps of gradient descent
    print("parameters after two steps = ", parameters)



