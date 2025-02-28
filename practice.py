import torch 
from torch import tensor
import math 
import matplotlib.pyplot as plt


# #  Tensors 
# w = tensor([3.0], requires_grad=True)

# f = lambda x: x**2
# g = lambda x: torch.sqrt(x)


# result = g(f(w))
# result.backward()

# print('result is', result)
# print('gradient is ', w.grad)


# # Plotting 
# x_range = torch.linspace(-10, 10, 1000)
# y = torch.zeros(x_range.shape)
# for i in range(len(y)): y[i] = x_range[i]**2
# plt.plot(x_range, y)
# plt.savefig("plot.png")  # Saves as a PNG file



# # switch to 3D
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # Make data.
# x = torch.arange(-5, 5, 0.25)
# y = torch.arange(-5, 5, 0.25)

# # Explicitly specify indexing="ij" to remove the warning
# x, y = torch.meshgrid(x, y, indexing="ij")

# z = x**2 + y**2

# # Plot the surface.
# surf = ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), linewidth=0, antialiased=False)

# plt.savefig('simple-surface.png', bbox_inches='tight')


# Tensor squeezing 

t1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
t2 = t1.unsqueeze(2)
print(t2)

