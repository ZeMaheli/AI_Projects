from NeuralNetworkImpl.activation_functions.tanh import tanh, tanh_derivative
from NeuralNetworkImpl.neural_network import NeuralNetwork

# Create the vertical bar  dataset
X_train =  [
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 1, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 1, 0, 0, 1, 0],
[0, 0, 1, 0, 0, 1, 0, 0, 1],
[1, 1, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 1, 1]
]
Y_train = [ [0], [1], [1], [1], [0], [0], [0] ]

# Create the neural network: 9 input neurons, 3 hidden neurons, 1 output neuron
nn = NeuralNetwork([9, 3, 1], tanh, tanh_derivative)

# Train the network
epochs = 1000
learning_rate = 0.05
error_threshold = 0.05
momentum=0.9

nn.train(X_train, Y_train, epochs, error_threshold, learning_rate, momentum)

# Test the network after training
for x in X_train:
    y_pred = nn.propagate(x)
    print(f"Input: {x}, Predicted Output: {y_pred}")
