# Here we will use all classes for our machine learning experiment
# import the data_gen class
import data_gen
# import the perceptron class
import perceptron


perceptron_accuracy = []
perceptron_epoch = []
perceptron_time = []
start_seed = 0


for i in range(10):
    # Create a new data_gen object
    data = data_gen.data_gen(1000)
    data.set_attributes_with_seed(start_seed)  # Set the seed for reproducibility
    # Generate the labels

    # PERCEPTRON EXPERIMENT:
    while data.generate_labels() == "retry":
        # If the classes are not balanced, retry generating the labels
        data = data_gen.data_gen(1000)
        start_seed += 1
        data.set_attributes_with_seed(start_seed)

    # Plot the data (saved to image file plot_data.png)
    data.plot_data()

    # use perceptron class
    # Create and train perceptron
    model = perceptron.perceptron(learning_rate=0.1)
    model.train(data, epochs=50)
    # append the accuracy, epoch, and time to the lists
    perceptron_accuracy.append(model.finalAccruacy)
    perceptron_epoch.append(model.finalEpoch)
    perceptron_time.append(model.totalTime)

# Print the average accuracy, epoch, and time for perceptron
print(f"Perceptron Average Accuracy: {sum(perceptron_accuracy)/len(perceptron_accuracy)}")
print(f"Perceptron Average Epoch: {sum(perceptron_epoch)/len(perceptron_epoch)}")
print(f"Perceptron Average Time: {sum(perceptron_time)/len(perceptron_time)}")
# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.hist(perceptron_accuracy, bins=20, color='blue', alpha=0.7)
plt.title('Perceptron Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.subplot(1, 3, 2)
plt.hist(perceptron_epoch, bins=20, color='green', alpha=0.7)
plt.title('Perceptron Epoch Distribution')
plt.xlabel('Epoch')
plt.ylabel('Frequency')
plt.subplot(1, 3, 3)
plt.hist(perceptron_time, bins=20, color='red', alpha=0.7)
plt.title('Perceptron Time Distribution')
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('perceptron_results.png')



 

