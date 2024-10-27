from .neuron import Neuron


class HiddenLayer:
    """
    Represents a hidden layer in a neural network, containing multiple neurons with a specified activation function.

    Attributes:
        input_size (int): The number of inputs each neuron in the layer receives.
        output_size (int): The number of neurons in the layer (defines the layer's output size)
        activation_function (callable): The function to activate each neuron's output.
        activation_derivative (callable): The derivative of the activation function, used for backpropagation.
        neurons (list of Neurons): A list of Neuron instances representing each neuron in the layer.
    Methods:
        y:
            Returns the current output value for each neuron in the layer

        propagate(x):
            Performs forward propagation by calculating the output of each neuron in the layer given the input `x`.

        adapt(delta_n, y_n_minus_1, alpha, beta):
            Adjusts each neuron's weights and bias based on the error, previous layer's output, learning rate,
            and momentum factor.
    """

    def __init__(self, input_size, output_size, activation_function, activation_derivative):
        """
        Initializes a HiddenLayer instance with a given number of neurons, each having a specific activation function.

        Parameters:
            input_size (int): The number of inputs each neuron in the layer will process.
            output_size (int): The number of neurons in the layer.
            activation_function (callable): The activation function for each neuron.
            activation_derivative (callable): The derivative of the activation function for backpropagation.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.neurons = [Neuron(input_size, activation_function, activation_derivative) for _ in range(output_size)]

    @property
    def y(self):
        """
        Retrieves the output of each neuron in the layer.

        Returns:
            list of float: The outputs of all neurons in the layer after propagation.
        """
        return [neuron.y for neuron in self.neurons]

    def propagate(self, x):
        """
        Performs forward propagation through the layer, calculating the output for each neuron.

        Parameters:
            x (list or np.ndarray): The input vector of size `input_size` for the layer.

        Returns:
            list of float: The output values from each neuron after activation.
        """
        output = [neuron.propagate(x) for neuron in self.neurons]
        return output

    def adapt(self, lowercase_delta_n, previous_layer_output, learning_rate, momentum):
        """
        Adjusts the weights and biases of all neurons in the layer based on the error.
        
        Parameters:
        lowercase_delta_n: list - Error vector for the current layer (layer n)
        previous_layer_output: list - Output vector of the previous layer (layer n-1)
        learning_rate: float - Learning rate
        momentum: float - Momentum factor
        """
        for j in range(self.output_size):
            lowercase_delta_jn = lowercase_delta_n[j]  # Error for neuron j in layer n
            self.neurons[j].adapt(lowercase_delta_jn, previous_layer_output, learning_rate, momentum)  # Adapt weights and bias for each neuron
