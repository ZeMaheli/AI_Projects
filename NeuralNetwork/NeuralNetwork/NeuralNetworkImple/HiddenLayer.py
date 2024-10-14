import numpy as np

import Neuron
import BaseLayer

class HiddenLayer(BaseLayer.BaseLayer):
    def __init__(self, input_size, output_size, activation_fun, activation_derivative):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_fun = activation_fun
        self.activation_derivative = activation_derivative
        self.neurons = [Neuron.Neuron(input_size, activation_fun, activation_derivative) for _ in range(output_size)]

    @property
    def y(self):
        return [neuron.y for neuron in self.neurons]

    def propagate(self, x):
        output = [neuron.propagate(x) for neuron in self.neurons]
        return output
    
    def adapt(self, delta_n, y_prev, alpha, beta):
        """
        Adjusts the weights and biases of all neurons in the layer based on the error.
        
        Parameters:
        delta_n: list - Error vector for the current layer (layer n)
        y_prev: list - Output vector of the previous layer (layer n-1)
        alpha: float - Learning rate
        beta: float - Momentum factor
        """
        for j in range(self.output_size):
            delta_jn = delta_n[j]  # Error for neuron j in layer n
            self.neurons[j].adapt(delta_jn, y_prev, alpha, beta)  # Adapt weights and bias for each neuron