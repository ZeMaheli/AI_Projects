import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt

# Define activation function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Tanh activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Mean Squared Error Loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

# Create the XOR dataset
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_train = [[0], [1], [1], [0]]

# Create the neural network: 2 input neurons, 2 hidden neurons, 1 output neuron
nn = NeuralNetwork.NeuralNetwork([2, 4, 1], activation_function=tanh, activation_derivative=tanh_derivative)

# Train the network
epochs = 10000
learning_rate = 0.1
error_threshold = 0.05
momentum=0.9

# Initialize an empty list to store the error for each epoch
errors = []

for epoch in range(epochs):
    # Perform forward propagation and calculate predictions
    y_pred = [nn.propagate(x) for x in X_train]
    
    # Calculate error (MSE) for the current epoch
    error = np.mean((np.array(Y_train) - np.array(y_pred)) ** 2)
    errors.append(error)
    
    # Stop training if error is below the threshold
    if error < error_threshold:
        print(f"Early stopping at epoch {epoch}, error: {error}")
        break

    # Perform backpropagation and update weights
    nn.treinar(X_train, Y_train, 1, error_threshold, learning_rate, momentum)  # One epoch per call

# Test the network after training
for x in X_train:
    y_pred = nn.propagate(x)
    print(f"Input: {x}, Predicted Output: {y_pred}")

# Plot the error
plt.plot(range(len(errors)), errors)
plt.title("Training Error over Time")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()