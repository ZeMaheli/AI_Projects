class EntryLayer:
    """
    Represents the input layer in a neural network, which holds and propagates the input data to the next layer.

    Attributes:
        output_size (int): The number of inputs the layer will process.
        y (list of float): The current input values stored in the layer.

    Methods:
        propagate(x):
            Stores and returns the input vector `x` as the layer's output.
    """

    def __init__(self, output_size):
        """
        Initializes an EntryLayer instance with a specified number of output nodes.

        Parameters:
            output_size (int): The number of output values expected by the layer.
        """
        self.output_size = output_size
        self.y = [0] * output_size  # Placeholder for output values

    def propagate(self, x):
        """
        Sets the input values `x` as the output of this layer to pass along to the next layer.

        Parameters:
            x (list or np.ndarray): The input vector of size `input_size`.

        Returns:
            list of float: The input vector `x`, representing the layer's output.
        """
        self.y = x
        return self.y
