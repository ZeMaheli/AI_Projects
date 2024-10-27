import numpy as np


class Neuron:
    """
    Represents a single neuron in a neural network with customizable activation function and learning methods.

    Attributes:
        input_size (int): The number of inputs to the neuron.
        activation_function (callable): The function to activate the neuron's output.
        activation_derivative (callable): The derivative of the activation function, used for backpropagation.
        weights (np.ndarray): The weights assigned to each input connection.
        bias (float): The neuron's bias term.
        delta_w (np.ndarray): The variation in weights for momentum-based updates.
        delta_b (float): The variation in bias for momentum-based updates.
        h (float): The weighted sum of inputs.
        y (float): The neuron's output after applying the activation function.
        y_prime (float): The derivative of the neuron's output.

    Methods:
        propagate(x):
            Calculates the neuron's output for a given input vector `x`.

        adapt(delta_n, y_n_minus_1, alpha, beta):
            Adjusts the neuron's weights and bias based on the error, previous layer's output, learning rate,
            and momentum factor.
    """

    def __init__(self, input_size, activation_function, activation_derivative):
        """
        Initializes a Neuron instance.

        Parameters:
            input_size (int): The number of inputs the neuron will process.
            activation_function (callable): The activation function applied to the weighted sum of inputs.
            activation_derivative (callable): The derivative of the activation function.
        """
        self.input_size = input_size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.weights = np.random.uniform(-1, 1, input_size)
        print(self.weights)
        self.bias = np.random.uniform(-1, 1)
        self.delta_w = np.zeros(input_size)  # weight variation
        self.delta_b = 0  # bias variation
        self.h = 0  # sum of inputs
        self.y = 0  # neuron output
        self.y_prime = 0  # Neuron output derivative

    def propagate(self, x):
        """
        Forward propagation step. Calculates the neuron's output for a given input.

        Parameters:
            x (np.ndarray): The input vector of size `input_size`.

        Returns:
            float: The neuron's output after applying the activation function.
        """
        self.h = np.dot(x, self.weights) + self.bias
        self.y = self.activation_function(self.h)
        self.y_prime = self.activation_derivative(self.h)  # Derivative
        return self.y

    def adapt(self, lowercase_delta_n, previous_layer_output, learning_rate, momentum):
        """
        Backpropagation and weight adjustment step. Updates the weights and bias based on the error signal,
        learning rate, and momentum.

        Parameters:
            lowercase_delta_n (float): The error signal from the next layer or loss gradient.
            previous_layer_output (np.ndarray): The output from the previous layer, of size `input_size`.
            learning_rate (float): The learning rate used to scale weight and bias adjustments.
            momentum (float): The momentum factor to stabilize training.

        Updates:
            self.weights: Adjusted by the calculated delta_w.
            self.bias: Adjusted by the calculated delta_b.
        """
        # Ensure previous_layer_output is a numpy array for element-wise operations
        previous_layer_output_array = np.array(previous_layer_output)

        # Momentum for weights
        momentum_w = momentum * self.delta_w

        # Update weights and bias based on the error and previous layer's outputs
        self.delta_w = -learning_rate * self.y_prime * lowercase_delta_n * previous_layer_output_array + momentum_w
        self.weights += self.delta_w

        # Momentum for bias
        momentum_b = momentum * self.delta_b

        self.delta_b = -learning_rate * self.y_prime * lowercase_delta_n + momentum_b
        self.bias += self.delta_b
