import numpy as np

import EntryLayer
import HiddenLayer
    
class NeuralNetwork():
    def __init__(self, forma_rede, activation_function, activation_derivative):
        self.camadas = []
        N = len(forma_rede)

        ds1 = forma_rede[0]
        camada1 = EntryLayer.EntryLayer(ds1)
        self.camadas.append(camada1)

        for n in range(1, N):
            de_n = forma_rede[n - 1] 
            ds_n = forma_rede[n]
            camada_n = HiddenLayer.HiddenLayer(de_n, ds_n, activation_function, activation_derivative)
            self.camadas.append(camada_n)

    def propagate(self, x):
        output = self.camadas[0].propagate(x)

        for camada in self.camadas[1:]:
            output = camada.propagate(output)
        
        return output

    @property
    def y(self):
        return self.camadas[-1].y
    
    def delta_saida(self, y_pred, y_true):
        """
        Computes the output loss delta for the final layer.
        
        Parameters:
        y_pred: list - Vector of the network's output (𝒚𝑁)
        y_true: list - Target output vector (𝒚)
        
        Returns:
        delta: list - Output loss vector
        """
        return [y_pred[k] - y_true[k] for k in range(len(y_true))]
    
    def retropropagar(self, delta_N, alpha, beta):
        """
        Performs backpropagation through the network, adjusting weights and biases.
        
        Parameters:
        delta_N: list - Error vector for the output layer (𝜹𝑁)
        alpha: float - Learning rate
        beta: float - Momentum factor
        """
        delta = delta_N  # Start with the error at the output layer

        # Loop through the layers from the last hidden layer to the first hidden layer (layer N to 2)
        for n in range(len(self.camadas) - 1, 0, -1):
            camada = self.camadas[n]
            camada_prev = self.camadas[n - 1]
            
            # Get the output of the previous layer (𝒚𝑛−1)
            y_prev = camada_prev.y

            # Adapt the current layer (adjust weights and biases using backpropagation)
            camada.adapt(delta, y_prev, alpha, beta)

            # Get dimensions of the current and previous layer
            #dn = len(camada.neurons)           # Dimension of current layer
            #dn_1 = len(camada_prev.neurons)    # Dimension of previous layer

            # Calculate the delta for the previous layer (𝜹𝑛−1)
            delta_new = []
            """ for i in range(dn_1):
                # Calculate the delta for each neuron in the previous layer
                sum_delta = 0
                for j in range(dn):
                    neuron_j = camada.neurons[j]
                    sum_delta += neuron_j.weights[i] * delta[j] * neuron_j.y_prime  # Weighted sum of deltas
                delta_new.append(sum_delta) """

            for i in range(len(y_prev)):
                sum_delta = sum(neuron.weights[i] * delta[j] * neuron.y_prime for j, neuron in enumerate(camada.neurons))
                delta_new.append(sum_delta)

            # Update delta for the next iteration (moving backward)
            delta = delta_new

    def adaptar(self, x, y_true, alpha, beta):
        """
        Adjusts the network parameters (weights and biases) based on the input x and target output y.
        
        Parameters:
        x: list - Input vector (𝒙)
        y_true: list - Target output vector (𝒚)
        alpha: float - Learning rate (𝛼)
        beta: float - Momentum factor (𝛽)
        
        Returns:
        epsilon: float - Average output error (𝜀)
        """
        # 2. Forward propagate to get the output of the network (𝒚𝑁)
        y_pred = self.propagate(x)

        # 3. Compute the output error (𝜹𝑁)
        delta_N = self.delta_saida(y_pred, y_true)

        # 4. Backpropagate to adjust weights and biases
        self.retropropagar(delta_N, alpha, beta)

        # 5. Compute K, the length of the error vector (number of output neurons)
        K = len(delta_N)

        # 6. Compute the average error (𝜀)
        epsilon = sum(delta_N) / K

        # 7. Return the average error
        return epsilon
    
    def treinar(self, X, Y, n_epocas, epsilon_max, alpha, beta):
        """
        Trains the neural network over multiple epochs.
        
        Parameters:
        X: list - Training input dataset (𝑿)
        Y: list - Training output dataset (𝒀)
        n_epocas: int - Number of epochs (𝑛é𝑝𝑜𝑐𝑎𝑠)
        epsilon_max: float - Maximum allowable error (𝜀𝑚𝑎𝑥)
        alpha: float - Learning rate (𝛼)
        beta: float - Momentum factor (𝛽)
        
        Returns:
        final_error: float - Final error after training.
        """
        for epoca in range(n_epocas):
            epsilon = 0  # Initialize error for the epoch

            # Iterate over the training examples (𝒙, 𝒚)
            for x, y in zip(X, Y):
                epsilon_x = self.adaptar(x, y, alpha, beta)  # Training error for input x
                epsilon = max(epsilon, epsilon_x)  # Track the maximum error in the epoch

            print(f"Epoch {epoca + 1}, Error: {epsilon}")

            # Check if the error is below the allowed threshold (𝜀𝑚𝑎𝑥)
            if epsilon <= epsilon_max:
                print(f"Training stopped at epoch {epoca + 1} with error {epsilon}.")
                break

        return epsilon  # Return the final error after training