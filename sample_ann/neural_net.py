"""
    AUTHOR: Owen Cupps
    DATE: 05/26/2025
    
    DESCRIPTION:
    This script defines and trains a simple artificial neural network using NumPy.

    The network consists of:
     - An input layer with 2 neurons
     - A single hidden layer with 4 neurons
     - An output layer with 1 neuron

    This implementation demonstrates how a multi-layer neural network can learn a 
    non-linear function - specifically, the XOR function - which cannot be solved
    by a linear model. The XOR truth table is provided as training data. It uses 
    the sigmoid activation function and is trained using gradient descent via 
    backpropogation.
"""
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        # Learning Rate: Controls how big of a step we take in the direction of gradient descent
        self.lr = lr

        # The weights are randomly initialized to prevent ambiguity between neurons, allowing them to pick up on
        # trends in the data.
        # .random.randn((rows, columns)) creates an array filled with random nums

        # Weights between input and hidden layer
        self.w1 = np.random.randn(input_size, hidden_size)
        # Weights between hidden layer and output layer
        self.w2 = np.random.randn(hidden_size, output_size)

        # Biases between each layer:
        # .zeros((rows, columns)) creates an array with all zeroes
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    # Seperates any real number into a value between 0 and 1
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Derivative of the sigmoid function
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    # Forward propogation
    def forward(self, x):
        # Multiply the inputs, x, by the first set of weights, then add the bias for the hidden layer
        self.z1 = x @ self.w1 + self.b1
        # Apply sigmoid activation function for z1
        self.a1 = self.sigmoid(self.z1)
        # Take the output from the hidden layer, a1, multiply by the weights and add the biases
        self.z2 = self.a1 @ self.w2 + self.b2
        # Apply sigmoid activation function for z2
        self.a2 = self.sigmoid(self.z2)
        # Return output from
        return self.a2

    # Backpropogation function
    def backward(self, x, y):
        # m is the number of training examples in this current batch
        # used to compute average gradients over the batch
        m = y.shape[0]
        # This is the gradient of the loss w.r.t. the output layer input z2
        dz2 = self.a2 - y
        # This is the gradient of the loss w.r.t. the weights in layer 2 (w2)
        dw2 = self.a1.T @ dz2 / m
        # This is the gradient of the loss w.r.t. the output layer bias
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # This backpropogates the error to the hidden layer
        dz1 = dz2 @ self.w2.T * self.sigmoid_derivative(self.z1)
        # Gradient of the loss w.r.t. weights in layer 1 (w1)
        dw1 = x.T @ dz1 / m
        # Gradient of the loss w.r.t. the hidden layer biases
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # These lines update the weights and biases used in the forward propogation step
        # It moves them in the direction of the gradient descent.
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2

    # One epoch is one full pass through the entire dataset
    def train(self, x, y, epochs=10000):
        for i in range(epochs):
            # Forward step for each epoch
            self.forward(x)
            # Backward step for each epoch
            self.backward(x, y)

# Input data
inputArray = np.array([[0,0], [0,1], [1,0], [1,1]])

# XOR training data
xorArray = np.array([[0], [1], [1], [0]])


# Initializes a new network with input_size, hidden_size, and output_size respectively
nn = NeuralNetwork(2, 4, 1)
# Train the NN
nn.train(inputArray, xorArray)
# Output the predictions
print("Predictions")
print(nn.forward(inputArray))
