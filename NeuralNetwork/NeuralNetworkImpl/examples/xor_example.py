from NeuralNetworkImpl.activation_functions.sigmoid import sigmoid, sigmoid_derivative
from NeuralNetworkImpl.activation_functions.tanh import tanh, tanh_derivative
from NeuralNetworkImpl.neural_network import NeuralNetwork

# Create the XOR dataset
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_train = [[0], [1], [1], [0]]

# Create the neural network: 2 input neurons, 4 hidden neurons, 1 output neuron
nn = NeuralNetwork([2, 4, 1], tanh, tanh_derivative)

# Train the neural network
n_epochs = 1000  # Number of training epochs
epsilon_max = 0.05  # Error threshold
learning_rate = 0.1  # Learning rate
momentum = 0.9  # Momentum factor

nn.train(X_train, Y_train, n_epochs, epsilon_max, learning_rate, momentum)

# Test the trained neural network
for x in X_train:
    prediction = nn.propagate(x)
    print(f"Input: {x}, Predicted Output: {prediction}")