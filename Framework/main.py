import loss_functions.cross_entropy as cross_entropy
import optimization.gradient_descent as gradient_descent
import utils.data as data
import utils.hyperplane as hyperplane
from models.model import Model
import utils.graph as graph
import os

csv_file_path = os.path.join(os.path.dirname(__file__), "Heart.csv")
data = data.Data()
#data.generate_linearly_separable(n_samples=50, dim=2)
data.load_from_csv(csv_file_path, normalize='min_max')
hyperplane = hyperplane.Hyperplane(dim=2)
loss_function = cross_entropy.CrossEntropyLoss()
optimizer = gradient_descent.GradientDescent(learning_rate=0.01)
model = Model(hyperplane=hyperplane, loss_function=loss_function, optimizer=optimizer)
model.train(data,learning_rate=0.05, epochs=10000)
model.test(data)
graph = graph.Graph()

# Create the graphs directory if it doesn't exist
graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")

# Create the full path for the image
save_path = os.path.join(graphs_dir, "test.png")

# Save the graph
graph.plot_and_save(data, hyperplane, save_path=save_path)