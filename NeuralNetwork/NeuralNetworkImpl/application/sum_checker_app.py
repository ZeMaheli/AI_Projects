from NeuralNetworkImpl.activation_functions.sigmoid import sigmoid, sigmoid_derivative
from NeuralNetworkImpl.application.generate_sum_dataset import generate_sum_dataset
from NeuralNetworkImpl.neural_network import NeuralNetwork

# Create the neural network: 4 input neurons, 3 hidden neurons, 1 output neuron
nn = NeuralNetwork([4, 3, 1], sigmoid, sigmoid_derivative)

# Generate the dataset
X_train, Y_train = generate_sum_dataset(250)

# Train the neural network
nn.train(X_train, Y_train, n_epochs=1000, epsilon_max=0.01, learning_rate=0.01, momentum=0.9)

def get_user_input():
    """
    Prompts the user to enter two integers between 0 and 30 (representing two numbers to add)
    and a proposed sum. Validates that the input meets expected ranges.

    Returns:
        tuple of (int, int, int): Contains the two input numbers and the proposed sum
        if valid; otherwise, returns None if input is invalid.
    """
    user_input = input("Enter two numbers (0-30) and a proposed sum, separated by spaces (e.g., '15 15 30'): ")
    try:
        # Parse the input and convert to integers
        a, b, c = map(int, user_input.split())
        if all(0 <= x <= 30 for x in [a, b]) and 0 <= c <= 60:
            return a, b, c # Return valid input tuple
        print("Numbers must be within the range: 1-30 for the first two, and up to 60 for the sum.")
    except ValueError:
        # Handle cases where input cannot be parsed as integers
        print("Invalid input. Please enter three integers.")
    # Return None for invalid input
    return None

def main():
    """
    Main loop of the application. Continuously prompts the user for input,
    processes it through the neural network to predict if the sum is correct,
    and displays the result.
    """
    while True:
        user_input = get_user_input()

        # Proceed only if input is valid
        if user_input:
            a, b, c = user_input
            correct_sum = a + b
            abs_difference = abs(correct_sum - c)

            # Feed inputs into the neural network
            prediction = nn.propagate([a, b, c, abs_difference])

            # Interpret the neural network's output
            result = "Correct" if prediction[0] >= 0.5 else "Incorrect"
            print(f"The prediction is: {result} -> {a} + {b} {'==' if result == 'Correct' else '!='} {c}")


if __name__ == "__main__":
    main()
