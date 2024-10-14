import numpy as np

class Neuron():
    def __init__(self, input_size, activation_function, activation_derivative):
        self.input_size = input_size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1,1)
        self.delta_w = np.zeros(input_size)  # Variação dos pesos
        self.delta_b = 0  # Variação do pendor
        self.h = 0
        self.y = 0
        self.y_prime = 0  # Derivada da saída do neurónio
    
    def propagate(self, x):
        self.h = np.dot(x, self.weights) + self.bias
        self.y = self.activation_function(self.h)
        self.y_prime = self.activation_derivative(self.h)  # Derivative
        return self.y
    
    def adapt(self, delta_n, y_n_minus_1, alpha, beta):
        # Update weights and bias based on the error and previous layer's outputs
        self.delta_w = -alpha * self.y_prime * delta_n * y_n_minus_1
        self.weights += self.delta_w
        
        self.delta_b = -alpha * self.y_prime * delta_n
        self.bias += self.delta_b