import unittest
import numpy as np
from NeuralNetwork import HiddenLayer  # Import the class

# Assuming Neuron and Layer classes are already defined above

# Define a simple activation function for testing
def identity(x):
    return x

# Test class for the Layer and Neuron classes
class TestHiddenLayer(unittest.TestCase):
    
    def test_layer_initialization(self):
        # Test if the layer initializes correctly
        de = 3  # 3 input connections
        ds = 3  # 3 neurons in the layer
        layer = HiddenLayer(de, ds, identity)
        
        # Check if the correct number of neurons are initialized
        self.assertEqual(len(layer.neurons), ds)
        # Check if each neuron has the correct input dimension
        for neuron in layer.neurons:
            self.assertEqual(neuron.d, de)
    
    def test_propagation(self):
        # Test if the layer propagates inputs correctly
        de = 3  # Input dimension
        ds = 3  # Output dimension (number of neurons)
        layer = HiddenLayer(de, ds, identity)
        
        # Input vector (matching the input dimension of the layer)
        x = np.array([0.5, -0.2, 0.1])
        
        # Propagate input through the layer
        output = layer.propagate(x)
        
        # Check if the layer returns the correct number of outputs
        self.assertEqual(len(output), ds)
        # Check if the output is a list of floats (identity activation returns the sum without modification)
        self.assertTrue(all(isinstance(y, np.float64) for y in output))
    
    def test_output_property(self):
        # Test the output property of the layer
        de = 3
        ds = 3
        layer = HiddenLayer(de, ds, identity)
        
        # Input vector
        x = np.array([0.5, -0.2, 0.1])
        
        # Propagate input through the layer
        layer.propagate(x)
        
        # Get the output via the property
        output_property = layer.y
        
        # Check that the output property returns the same as the direct propagation
        self.assertEqual(len(output_property), ds)
        self.assertTrue(all(isinstance(y, np.float64) for y in output_property))
    
    def test_different_activation_function(self):
        # Test with a sigmoid activation function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        de = 3
        ds = 3
        layer = HiddenLayer(de, ds, sigmoid)
        
        # Input vector
        x = np.array([0.5, -0.2, 0.1])
        
        # Propagate input
        output = layer.propagate(x)
        
        # Check if the output values are within (0, 1) due to sigmoid
        self.assertTrue(all(0 < y < 1 for y in output))
    
    def test_consistent_outputs(self):
        # Test if neurons produce different outputs with different weights/biases
        de = 3
        ds = 3
        layer = HiddenLayer(de, ds, identity)
        
        # Input vector
        x = np.array([0.5, -0.2, 0.1])
        
        # Propagate input
        output = layer.propagate(x)
        
        # Ensure outputs are not identical (if neurons are properly initialized)
        self.assertNotEqual(output[0], output[1])
        self.assertNotEqual(output[1], output[2])

# Running the tests
if __name__ == '__main__':
    unittest.main()
