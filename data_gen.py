import numpy as np
import torch
from torch import tensor
import matplotlib.pyplot as plt



class data_gen:
    def __init__(self, num_instances):
        self.instances = num_instances
        # Define weights with larger values to create better separation
        self.w = None
        # Define bias with 1 dimension
        self.b = None
        # Define data matrix x with shape (num_instances, 2)
        self.x = None
        # Initialize labels tensor
        self.labels = None
   
    def set_attributes_with_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Define weights with larger values to create better separation
        self.w = tensor(np.random.uniform(-1, 1, 2), dtype=torch.float32, requires_grad=True)
        # Define bias with 1 dimension
        self.b = tensor(np.random.uniform(-1, 1, 1), dtype=torch.float32, requires_grad=True)
        # Define data matrix x with shape (num_instances, 2)
        self.x = tensor(np.random.rand(self.instances, 2), dtype=torch.float32)
        # Initialize labels tensor
        self.labels = torch.zeros(self.instances, dtype=torch.float32)

    def generate_labels(self):
        # print out the weights and bias
        print(f"w: {self.w}, b: {self.b}")
        # Compute the linear combination
        line = self.x @ self.w + self.b
        # Assign labels in a vectorized way
        self.labels = torch.where(line > 0, torch.tensor(1.0), torch.tensor(-1.0))
        
        # Check class balance
        pos_count = (self.labels == 1).sum().item()
        neg_count = (self.labels == -1).sum().item()
        print(f"Class 1: {pos_count}, Class -1: {neg_count}")

        # Check if the classes are balanced
        if pos_count < neg_count/2 or neg_count < pos_count/2:
            return("retry")





        
        return self.labels
        
    def plot_data(self):
        import matplotlib.pyplot as plt
        # plot the data points with different colors for each class
        plt.scatter(self.x[self.labels == 1][:, 0].detach().numpy(), 
                    self.x[self.labels == 1][:, 1].detach().numpy(), 
                    color='blue', label='Class 1')
        plt.scatter(self.x[self.labels == -1][:, 0].detach().numpy(), 
                    self.x[self.labels == -1][:, 1].detach().numpy(), 
                    color='red', label='Class -1')
        
        # plot the line
        x_line = np.linspace(0, 1, 100)
        y_line = (-self.w[0].detach().numpy() * x_line - self.b.detach().numpy()) / self.w[1].detach().numpy()
        plt.plot(x_line, y_line, color='green', label='Decision Boundary')
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('linear_data.png')
        #plt.show()