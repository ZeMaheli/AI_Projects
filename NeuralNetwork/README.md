Sum Checker Neural Network

The Sum Checker app is designed to evaluate whether the proposed sum of two numbers is correct or incorrect based on user input. Using a neural network trained on a dataset of number pairs and their respective sums, this application determines if the given sum is accurate. This project demonstrates the power of neural networks in binary classification tasks, offering an interactive way for users to engage with machine learning concepts.

Purpose

The purpose of this app is to provide an educational tool where users input two numbers along with a proposed sum. The neural network then evaluates whether the proposed sum is correct, serving as a straightforward application of machine learning techniques to a simple arithmetic problem.

Dataset

The dataset contains 250 entries, each consisting of two numbers and a proposed sum. These numbers are randomly generated between 1 and 30, and each entry is labeled as correct or incorrect based on the sum’s accuracy. The dataset is balanced between correct and incorrect sums to enable effective training for the neural network.

Neural Network Architecture

The neural network consists of:

    Input Layer: 4 input neurons for two numbers, their proposed sum, and the absolute difference.
    Hidden Layer: 3 hidden neurons to capture the complexity of the problem.
    Output Layer: 1 output neuron to indicate whether the proposed sum is correct (1) or incorrect (0).

Shape of the Network

The choice of this network shape is based on the following considerations:

    Input Size: The input layer has 4 neurons because it takes into account two numbers, the proposed sum, and the absolute difference. This allows the network to process the necessary information to make predictions.

    Hidden Neurons: The use of 3 hidden neurons strikes a balance between complexity and simplicity. It provides sufficient capacity to capture patterns in the data without becoming overly complex, which can lead to overfitting.

    Output Size: A single output neuron is used to provide a binary classification: whether the proposed sum is correct or incorrect. This straightforward output is suitable for the nature of the problem.

Neural Network Training Parameters
1. Activation Function: Sigmoid

    Reason for Choice: Sigmoid was chosen for its suitability in binary classification tasks, outputting values between 0 and 1. This allows us to interpret the neural network’s output directly as the probability of the proposed sum being correct.

2. Number of Epochs: 1000

    Reason for Choice: With 1000 epochs, the neural network has adequate training time to learn from the dataset. This allows for effective convergence without overfitting, balancing training duration and learning effectiveness.

3. Learning Rate: 0.01

    Reason for Choice: A learning rate of 0.01 balances between convergence speed and stability. This value is small enough to prevent the model from diverging but large enough to ensure efficient progress.

4. Momentum: 0.9

    Reason for Choice: Momentum helps speed up training and reduce oscillations by accounting for past gradients. A momentum value of 0.9 provides stability and helps navigate local minima effectively.

5. Maximum Epsilon: 0.01

    Reason for Choice: The maximum epsilon value sets a threshold for the acceptable error margin during training, stopping early if error is reduced below this level, ensuring a reasonable accuracy.

Tutorial: Running the App

Step 1: Run the Application

    Execute the following command in your terminal, inside NeuralNetwork directory:

    python NeuralNetworkImpl\application\sum_checker_app.py

While the neural network is training, you will see the number of epochs increase with its error value being presented.
    
    ...
    > Epoch 104, Error: 0.022098366309300316
    > Epoch 105, Error: 0.021375706426725184
    > Epoch 106, Error: 0.020676963446158228
    > Epoch 107, Error: 0.020001666940626994
    > Epoch 108, Error: 0.019351247396971215
    > Epoch 109, Error: 0.018727459063384265
    > Epoch 104, Error: 0.022098366309300316
    > Epoch 105, Error: 0.021375706426725184
    > Epoch 106, Error: 0.020676963446158228
    > Epoch 107, Error: 0.020001666940626994
    ...

Step 2: Input Numbers

    After training completes, it will show the epoch where it reached the max epsilon condition and its error.
    The app will then prompt you to input two numbers between 1 and 30, followed by a proposed sum, formatted as:

    > Enter two numbers (1-30) and a proposed sum, separated by spaces (e.g., '15 15 30'):

Step 3: View Results

    The app will respond with whether the sum is predicted as correct or incorrect:

    > The prediction is: Correct -> 15 + 15 == 30

Conclusion

The Sum Checker app highlights the capability of neural networks to solve arithmetic classification problems. Through parameter tuning, the model can predict the correctness of sums based on user input, showcasing machine learning in a way that reinforces fundamental arithmetic skills.