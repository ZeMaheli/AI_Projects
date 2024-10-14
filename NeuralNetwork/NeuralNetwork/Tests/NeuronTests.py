import unittest
import numpy as np

from .. import Neuron

# Define a simple activation function for testing
def identity(x):
    return x

class TestNeuron(unittest.TestCase):
    
    def test_neuron_initialization(self):
        # Test if the neuron initializes correctly
        d = 3  # Number of input connections
        neuron = Neuron(d, identity)
        
        # Check if the input dimension is set correctly
        self.assertEqual(neuron.d, d)
        # Check if the weights are initialized to the correct length
        self.assertEqual(len(neuron.w), d)
        # Check if the bias is a float
        self.assertIsInstance(neuron.b, np.float64)
    
    def test_propagation_identity_activation(self):
        # Test propagation with identity activation function
        d = 3
        neuron = Neuron(d, identity)
        
        # Input vector (same length as neuron input dimension)
        x = np.array([0.5, -0.2, 0.1])
        
        # Propagate input through the neuron
        output = neuron.propagate(x)
        
        # Manually compute the expected output
        expected_h = np.dot(x, neuron.w) + neuron.b  # Weighted sum
        expected_output = expected_h  # Since activation is identity
        
        # Check if the output matches the expected output
        self.assertEqual(output, expected_output)
    
    def test_propagation_sigmoid_activation(self):
        # Test propagation with a sigmoid activation function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        d = 3
        neuron = Neuron(d, sigmoid)
        
        # Input vector
        x = np.array([0.5, -0.2, 0.1])
        
        # Propagate input through the neuron
        output = neuron.propagate(x)
        
        # Manually compute the expected output
        expected_h = np.dot(x, neuron.w) + neuron.b
        expected_output = sigmoid(expected_h)
        
        # Check if the output is within (0, 1) because it's sigmoid
        self.assertTrue(0 < output < 1)
    
    def test_different_inputs(self):
        # Test that different inputs produce different outputs
        d = 3
        neuron = Neuron(d, identity)
        
        # First input
        x1 = np.array([0.5, -0.2, 0.1])
        output1 = neuron.propagate(x1)
        
        # Second input (different from the first)
        x2 = np.array([-0.1, 0.2, 0.3])
        output2 = neuron.propagate(x2)
        
        # Ensure the outputs for different inputs are not the same
        self.assertNotEqual(output1, output2)

    def test_weight_and_bias_effect(self):
        # Test the effect of weights and bias on the output
        d = 3
        neuron = Neuron(d, identity)
        
        # Manually set weights and bias
        neuron.w = np.array([1, 0, -1])
        neuron.b = 1.0
        
        # Input vector
        x = np.array([1, 2, 3])  # This should create a specific output
        
        # Propagate input
        output = neuron.propagate(x)
        
        # Expected output: dot([1, 2, 3], [1, 0, -1]) + 1 = (1*1 + 2*0 + 3*(-1)) + 1 = -1 + 1 = 0
        expected_output = 0
        
        # Check if the output is as expected
        self.assertEqual(output, expected_output)

# Running the tests
if __name__ == '__main__':
    unittest.main()
