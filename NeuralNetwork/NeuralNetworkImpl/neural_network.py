from matplotlib import pyplot as plt

from .entry_layer import EntryLayer
from .hidden_layer import HiddenLayer


class NeuralNetwork():
    """
    Represents a multi-layer neural network with customizable layer structure, activation function, and learning parameters.

    Attributes:
        layers (list): A list containing instances of EntryLayer and HiddenLayer, defining the network architecture.

    Methods:
        output_delta(network_output_vector, output_training_vector):
            Calculates the error between the network's output and the expected output.

        backpropagation(lowercase_delta_N, alpha, beta):
            Performs backpropagation to update weights and biases across all layers in the network.

        adapt(x, output_training_vector, learning_rate, momentum):
            Completes a forward and backward pass, updating weights and biases, and computes the training error.

        train(X, Y, n_epochs, epsilon_max, learning_rate, momentum):
            Trains the neural network on a dataset over multiple epochs or until a specified error threshold is met.

        predict(X):
            Predicts the output for a given input dataset `X` after forward propagation.

        propagate(x):
            Performs a forward propagation of input `x` through the network layers to get the final output.
    """
    def __init__(self, network_shape, activation_function, activation_derivative):
        """
        Initializes a NeuralNetwork instance with a specified layer structure, activation function, and derivative.

        Parameters:
            network_shape (list of int): List defining the size of each layer in the network.
            activation_function (callable): The activation function to be applied at each neuron.
            activation_derivative (callable): The derivative of the activation function for backpropagation.
        """
        self.layers = []
        N = len(network_shape)

        entry_layer_size = network_shape[0]
        first_layer = EntryLayer(entry_layer_size)
        self.layers.append(first_layer)

        for n in range(1, N):
            previous_layer_output = network_shape[n - 1]
            current_layer_output = network_shape[n]
            current_layer = HiddenLayer(previous_layer_output, current_layer_output, activation_function,
                                        activation_derivative)
            self.layers.append(current_layer)

    def output_delta(self, network_output_vector, output_training_vector):
        """
        Computes the error vector between the network's output and the training target.

        Parameters:
            network_output_vector (list of float): The network's output values.
            output_training_vector (list of float): The expected output values for training.

        Returns:
            list of float: The error vector, element-wise difference between actual and expected output.
        """
        output_training_vector_len = len(output_training_vector)
        return [network_output_vector[k] - output_training_vector[k] for k in range(output_training_vector_len)]

    def backpropagation(self, lowercase_delta_N, learning_rate, momentum):
        """
        Updates weights and biases in all layers of the network by backpropagation the error.

        Parameters:
            lowercase_delta_N (list of float): Initial error vector from the output layer.
            learning_rate (float): Learning rate.
            momentum (float): Momentum factor to stabilize weight updates.
        """
        lowercase_delta_n = lowercase_delta_N  # Start with the error at the output layer

        # Loop through the layers from the last hidden layer to the first hidden layer (layer N to 0)
        for n in range(len(self.layers) - 1, 0, -1):
            current_layer = self.layers[n]
            previous_layer = self.layers[n - 1]

            previous_layer_output = previous_layer.y
            previous_layer_output_size = previous_layer.output_size
            current_layer_neurons = current_layer.neurons

            lowercase_delta_n_minus_1 = []

            for i in range(previous_layer_output_size):
                lowercase_delta_n_minus_1.append(
                    sum(neuron.weights[i] * lowercase_delta_n[j] * neuron.y_prime for j, neuron in
                        enumerate(current_layer_neurons)))

            # Adapt the current layer (adjust weights and biases using backpropagation)
            current_layer.adapt(lowercase_delta_n, previous_layer_output, learning_rate, momentum)

            # Update delta for the next iteration (moving backward)
            lowercase_delta_n = lowercase_delta_n_minus_1

    def adapt(self, x, output_training_vector, learning_rate, momentum):
        """
        Executes a forward pass and then backpropagation to update weights and biases.

        Parameters:
            x (list of float): Input vector for the network.
            output_training_vector (list of float): The target output vector.
            learning_rate (float): The learning rate for weight adjustments.
            momentum (float): The momentum factor for stabilizing training.

        Returns:
            float: The average error of the network's output relative to the target.
        """
        network_output_vector = self.propagate(x)
        lowercase_delta_N = self.output_delta(network_output_vector, output_training_vector)
        self.backpropagation(lowercase_delta_N, learning_rate, momentum)

        # Calculate average error
        epsilon = sum((delta ** 2 for delta in lowercase_delta_N)) / len(lowercase_delta_N)
        return epsilon

    def train(self, X, Y, n_epochs, epsilon_max, learning_rate, momentum):
        """
        Trains the neural network on a dataset for a specified number of epochs or until error is minimized.

        Parameters:
            X (list(list of float)): Input data, where each entry is an input vector.
            Y (list(list of float)): Target data, where each entry is the expected output for the corresponding input.
            n_epochs (int): Maximum number of training epochs.
            epsilon_max (float): Error threshold; training stops if error is below this value.
            learning_rate (float): Learning rate for training.
            momentum (float): Momentum factor to stabilize updates.
        """
        errors = []  # To store error for each epoch
        for epoch in range(n_epochs):
            epsilon = 0  # Initialize error for the epoch

            # Iterate over the training examples
            for x, y in zip(X, Y):
                epsilon_x = self.adapt(x, y, learning_rate, momentum)  # Training error for input x
                epsilon = max(epsilon, epsilon_x)  # Track the maximum error in the epoch

            errors.append(epsilon)  # Store the error for this epoch

            print(f"Epoch {epoch + 1}, Error: {epsilon}")

            # Check if the error is below the allowed threshold
            if epsilon <= epsilon_max:
                print(f"Training stopped at epoch {epoch + 1} with error {epsilon}.")
                break

        # Plotting the error over epochs
        plt.plot(errors)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Error Decrease over Epochs")
        plt.show()

    def predict(self, X):
        """
        Predicts the output for a batch of inputs after forward propagation.

        Parameters:
            X (list(list of float)): List of input vectors to be propagated through the network.

        Returns:
            list(list of float): Network outputs for each input vector.
        """
        Y = []
        for x in X:
            Y.append(self.propagate(x))
        return Y

    def propagate(self, x):
        """
        Performs forward propagation for a single input vector.

        Parameters:
            x (list of float): Input vector for the network.

        Returns:
            list of float: Final output of the network after passing through all layers.
        """
        output = self.layers[0].propagate(x)

        for layer in self.layers[1:]:
            output = layer.propagate(output)

        return output
