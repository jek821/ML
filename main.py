# Here we will use all classes for our machine learning experiment
# import the data_gen class
import data_gen
# import the perceptron class
import perceptron
import random 
import matplotlib.pyplot as plt


perceptron_accuracy = []
perceptron_epoch = []
perceptron_time = []



for i in range(1000):
    print("Iteration: "+ str(i) + "/1000") 
    start_seed = random.randint(0, 1000)
    # Create a new data_gen object
    data = data_gen.data_gen(1000)
    data.set_attributes_with_seed(start_seed+1)  # Set the seed for reproducibility
    # Generate the labels

    # PERCEPTRON EXPERIMENT:
    while data.generate_labels() == "retry":
        # If the classes are not balanced, retry generating the labels
        data = data_gen.data_gen(1000)
        start_seed += 1
        data.set_attributes_with_seed(start_seed)

    # Plot the data (saved to image file plot_data.png)
    # data.plot_data()

    # use perceptron class
    # Create and train perceptron
    model = perceptron.perceptron(learning_rate=0.01)
    model.train(data, epochs=100)
    # append the accuracy, epoch, and time to the lists
    perceptron_accuracy.append(model.finalAccruacy)
    perceptron_epoch.append(model.finalEpoch)
    perceptron_time.append(model.totalTime)

print(f"Perceptron Accuracy: {perceptron_accuracy}")
print(f"Perceptron Epoch: {perceptron_epoch}")
print(f"Perceptron Time: {perceptron_time}")

# Calculate averages
perceptron_accuracy_average = sum(perceptron_accuracy) / len(perceptron_accuracy)
perceptron_epoch_average = sum(perceptron_epoch) / len(perceptron_epoch)
perceptron_time_average = sum(perceptron_time) / len(perceptron_time)
# Print the average accuracy, epoch, and time for perceptron
print(f"Perceptron Average Accuracy: {perceptron_accuracy_average:.2%}")
print(f"Perceptron Average Number of Epochs: {[perceptron_epoch_average]}")
print(f"Perceptron Average Time: {perceptron_time_average:.2f} seconds")



# Do all experiment histogram plotting below:

# Plot Average Time
plt.figure(figsize=(5, 5))
plt.bar(['Perceptron'], [perceptron_time_average], color='blue')
plt.ylabel('Average Time (seconds)')
plt.title('Average Time for Learning Algorithm')
plt.text(0, perceptron_time_average, f'{perceptron_time_average:.2f}s', 
         ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.savefig('average_time.png')
#plt.show()

# Plot Average Accuracy
plt.figure(figsize=(5, 5))
plt.bar(['Perceptron'], [perceptron_accuracy_average], color='blue')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy for Learning Algorithm')
plt.text(0, perceptron_accuracy_average, f'{perceptron_accuracy_average:.2%}', 
         ha='center', va='bottom', fontsize=12, fontweight='bold')  # Percentage format
plt.savefig('average_accuracy.png')
#plt.show()

# Plot Average Epoch
plt.figure(figsize=(5, 5))
plt.bar(['Perceptron'], [perceptron_epoch_average], color='blue')
plt.ylabel('Average Epoch')
plt.title('Average Epoch for Learning Algorithm')
plt.text(0, perceptron_epoch_average, f'{perceptron_epoch_average:.0f}', 
         ha='center', va='bottom', fontsize=12, fontweight='bold')  # Integer format
plt.savefig('average_epoch.png')
#plt.show()

 

