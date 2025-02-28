import torch
import math 

# Exercise 1 
# Create a 2D tensor (3x2) with specified values
t1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(t1)  # Print the original tensor

# Add an extra dimension at index 2 (making it a 3x2x1 tensor)
t12 = torch.unsqueeze(t1, 2)
print(t12)  # Print the tensor with the new dimension

# Add an extra dimension at the second-to-last position (making it a 3x1x2 tensor)
t13 = torch.unsqueeze(t1, -2)
print("here", t13)  # Print the tensor with the new dimension

# Add an extra dimension at index 2 again (same as t12)
t14 = torch.unsqueeze(t1, 2)
print(t14)  # Print the tensor with the new dimension

# Exercise 2
# Create a 2D tensor (3x1) with specified values
t2 = torch.tensor([[10.0], [20.0], [30.0]])
t2add = torch.tensor([1.0])  # Create a 1D tensor with a single value
print(t2)  # Print the original tensor

# Print the sizes of t2 and t2add for reference
print(t2.size())  # Output: torch.Size([3, 1])
print(t2add.size())  # Output: torch.Size([1])

# Expand t2add to match the number of rows in t2 (3x1)
t2add_exp = t2add.unsqueeze(0).expand(t2.size(0), 1)

# Concatenate the expanded t2add tensor with t2 along dimension 1 (column-wise)
result = torch.cat((t2add_exp, t2), dim=1)
print(result)  # Print the resulting tensor

