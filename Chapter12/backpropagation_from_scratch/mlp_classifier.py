# Author: Roi Yehoshua <roiyeho@gmail.com>
# August 2024
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Implementation of a Multi-Layer Perceptron (MLP) for binary classification using backpropagation."""
    def __init__(self, hidden_layer_sizes=[100], learning_rate=0.1, batch_size=32, epochs=1000, 
                 report_interval=100, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes  
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs  
        self.report_interval = report_interval  
        self.random_state = np.random.RandomState(random_state)

        self.weights = []  
        self.biases = [] 

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def initialize_weights(self):
        """Initialize weights with small random numbers and biases with zeros."""
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(self.random_state.randn(self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.biases.append(np.zeros(self.layer_sizes[i + 1]))

    def forward_pass(self, X):
        """Compute the output of the network for each layer."""
        activations = [X]  # List of activations for each layer
        net_inputs = []  # List of net inputs for each layer
        
        for i in range(len(self.weights)):
            Z = activations[-1] @ self.weights[i] + self.biases[i] 
            A = self.sigmoid(Z)
            net_inputs.append(Z)            
            activations.append(A)
        
        return net_inputs, activations

    def back_propagate(self, y, net_inputs, activations):
        """Backward propagation to compute the gradients of the loss function."""
        deltas = [None] * len(self.weights)  # List of deltas for each layer
        
        # Compute the deltas at the output layer
        delta = activations[-1] - y  
        deltas[-1] = delta

        # Propagate deltas backward
        for i in reversed(range(len(deltas) - 1)):
            net_input_deriv = self.sigmoid_derivative(net_inputs[i])
            delta = (deltas[i + 1] @ self.weights[i + 1].T) * net_input_deriv 
            deltas[i] = delta

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * activations[i].T @ deltas[i] / len(y)
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0)

    def compute_loss(self, y, y_pred):
        """Compute the loss using log loss."""
        return -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

    def fit(self, X, y):
        """Fit the neural network to the training data, using mini-batch gradient descent."""
        n_samples, n_features = X.shape
        y = y.reshape(n_samples, 1)  # Ensure y is a column vector

        # Include input and output layers in the layers list
        self.layer_sizes = [n_features] + self.hidden_layer_sizes + [1]

        # Initialize weights and biases
        self.initialize_weights()

        for epoch in range(self.epochs):
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # Process data in mini-batches
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # Forward pass
                net_inputs, activations = self.forward_pass(X_batch)

                # Backward pass
                self.back_propagate(y_batch, net_inputs, activations)

            # Report training loss at specified intervals
            if (epoch + 1) % self.report_interval == 0 or epoch == 0:
                y_pred = self.predict_proba(X)
                loss = self.compute_loss(y, y_pred)
                print(f'Epoch {epoch + 1}, Training loss: {loss:.6f}')
            
    def predict_proba(self, X):
        """Predict class probabilities for the given inputs."""
        _, activations = self.forward_pass(X)
        return activations[-1]

    def predict(self, X):
        """Predict class labels for the given inputs."""
        prob = self.predict_proba(X)
        y_pred = (prob > 0.5).astype(int)
        return y_pred