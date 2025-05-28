"""
    AUTHOR: Owen Cupps
    DATE: 05/27/2025
    
    DESCRIPTION:
    This script defines and trains a simple artificial neural network using NumPy.

    The network consists of:
     - An input layer with 2 neurons
     - As many hidden layers as the user would like at any sizes
     - An output layer with 1 neuron

    This implementation is to demonstrate an ANN with multiple hidden layers. 
    This can be useful for other tasks but really for the XOR we don't need so many layers.
    To improve efficiency between hidden layers we use the ReLU activation 
    function, and on the output layer use sigmoid activation function.
"""
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.1):
        # Learning Rate: Controls how big of a step we take in the direction of gradient descent
        self.lr = lr
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # .random.randn((rows, columns)) 
        # creates a list of matrices of shape (size of layer i, size of layer i+1) 
        # filled with random nums representing the weights between pairs of adjacent layers,
        # Uses Xavier/Glorot initialization to prevent gradients from vanishing
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i+1]) 
            * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1]))
            for i in range(self.num_layers - 1)
        ]

        # .zeros((rows, columns)) 
        # creates a zero matrix of the shape (1, layer_sizes[i+1]) for each layer after
        # the input representing the biases between each layer.
        self.biases = [
            np.zeros((1, layer_sizes[i+1])) 
            for i in range(self.num_layers-1)
        ]


    # ACTIVATION FUNCTIONS
    # Seperates any real number into a value between 0 and 1
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Derivative of the sigmoid function
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    # Outputs the input if positive, and zero otherwise
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(Self, z):
        return (z > 0).astype(float)
    

    # Forward propogation
    def forward(self, x):
        self.zs = [] # List of linear combinations (z = a_prev @ w + b)
        self.activations = [x] # List of post-activations at each layer, initialized with input x

        # For each layer
        for i in range(self.num_layers - 1):
            # Compute the linear combination and add to list
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.zs.append(z)

            # Depending on which layer use activation function
            if i == self.num_layers - 2:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)

            # Add to list of post-activations
            self.activations.append(a)

        # Return most recent activation
        return self.activations[-1]


    # Backpropogation function
    def backward(self, y):
        # Number of training examples in the batch, used to average the gradients
        m = y.shape[0]
        # Initialize list for error at each layer
        deltas = [None] * (self.num_layers - 1)

        # Compute the error from the output at the forward step
        # Gives the gradient of the loss w.r.t. the weighted input
        deltas[-1] = (self.activations[-1] - y) * self.sigmoid_derivative(self.zs[-1])

        # Backpropogate through hidden layers
        # Iterates backwards from 2nd-to-last layer to the first hidden layer
        for l in reversed(range(self.num_layers - 2)):
            deltas[l] = (deltas[l + 1] @ self.weights[l + 1].T) * (self.relu_derivative(self.zs[l]))

        # Compute the gradients and update weights and biases
        for i in range(len(self.weights)):
            # Gradient of the loss w.r.t. the weights at layer i
            dw = self.activations[i].T @ deltas[i] / m
            # Gradient of the loss w.r.t. the biases
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            # Update the weights and biases by subtracting the gradients scaled
            # by the learning rate
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    # Compute loss using Mean Squared Error Formula
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # One epoch is one full pass through the entire dataset
    def train(self, x, y, epochs=50000, print_every = 10000):
        self.losses = [] # Track loss for visualization
        
        for i in range(epochs):
            # Output of forward step for each epoch
            output = self.forward(x)
            # Backward step for each epoch
            self.backward(y)

            # Compute loss and append to list
            loss = self.compute_loss(y, output)
            self.losses.append(loss)

            # Print the Epoch, Loss, and Current Predictions
            if i % print_every == 0 or i == epochs - 1:
                print(f"Epoch {i}, Loss: {loss:.6f}")
                print("Predictions")
                print(output)


    def make_plot(self):
        # Formating for Plot
        # figsize=(width, height), width and height for figure measured in inches
        plt.figure(figsize=(10, 6))
        # Loss data being measured
        plt.plot(self.losses)
        plt.title("Training Loss over Epochs", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plt.show()


# Input data
inputArray = np.array([[0,0], [0,1], [1,0], [1,1]])
# XOR training data
xorArray = np.array([[0], [1], [1], [0]])

input_size = 2
output_size = 1
hidden_layers = input("Number of Hidden Layers: ")
hidden_layer_sizes = []
for i in range(int(hidden_layers)):
    hidden_layer_sizes.append(int(input(f"Size of Hidden Layer {i+1}: ")))
layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

# Initializes a new network with input_size, hidden_size, and output_size respectively
nn = NeuralNetwork(layer_sizes)
# Train the NN
nn.train(inputArray, xorArray)
# Output the predictions
nn.make_plot()
