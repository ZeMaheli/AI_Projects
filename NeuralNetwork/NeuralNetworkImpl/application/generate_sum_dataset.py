import random

def generate_sum_dataset(num_entries):
    """
    Generates a dataset of sums for training a neural network. Each entry in the dataset
    consists of two randomly generated numbers, their correct or incorrect sum, and
    the absolute difference between the actual and provided sums.

    Args:
        num_entries (int): The number of dataset entries to generate.

    Returns:
        tuple: A tuple containing two lists:
            - X_train (list of lists): Each inner list contains four values:
                [num1, num2, proposed_sum, abs_difference]
            - Y_train (list): Contains the labels (1 for correct sum, 0 for incorrect sum)
    """
    X_train = []
    Y_train = []

    for i in range(num_entries):
        # Generate two random integers between 1 and 30
        num1 = random.randint(1, 30)
        num2 = random.randint(1, 30)
        correct_sum = num1 + num2

        # Decide whether to use correct or incorrect sum
        if i % 2 == 0:
            # Use the correct sum
            c = correct_sum
            Y_train.append([1])  # Label for correct sum
        else:
            # Generate an incorrect sum by adding a random integer (1 to 10)
            c = correct_sum + random.randint(1, 10)
            Y_train.append([0])  # Label for incorrect sum

        # Calculate the absolute difference between correct and proposed sum
        abs_difference = abs(correct_sum - c)

        # Append values to the dataset
        X_train.append([num1, num2, c, abs_difference])

    return X_train, Y_train