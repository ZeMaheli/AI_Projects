import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork

# Define activation function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Create the XOR dataset
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_train = [[0], [1], [1], [0]]

# Create the neural network: 2 input neurons, 2 hidden neurons, 1 output neuron
nn = NeuralNetwork.NeuralNetwork([2, 2, 1], activation_function=sigmoid, activation_derivative=sigmoid_derivative)

# Train the network
epochs = 10000
learning_rate = 0.1
momentum = 0.9
error_threshold = 0.01

errors = nn.treinar(X_train, Y_train, epochs, error_threshold, learning_rate, momentum)

# Test the network
for x in X_train:
    y_pred = nn.propagate(x)
    print(f"Input: {x}, Predicted Output: {y_pred}")

# Plot the error
plt.plot(errors)
plt.title("Training Error over Time")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()