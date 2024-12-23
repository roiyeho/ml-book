# Author: Roi Yehoshua <roiyeho@gmail.com>
# August 2024
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Implementation of a Multi-Layer Perceptron (MLP) for binary classification using backpropagation."""
    def __init__(self, hidden_layer_sizes=[100], learning_rate=0.1, batch_size=32, 
                 n_epochs=1000, report_interval=100, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes 
        self.n_layers = len(self.hidden_layer_sizes) + 1  # adding the output layer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs  
        self.report_interval = report_interval  
        self.random_state = np.random.RandomState(random_state)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward_pass(self, X):
        """Compute the neuron activations for each layer."""
        net_inputs = []    # Net inputs of each hidden/output layer
        activations = []   # Activations of each hidden/output layer

        # Compute the activations for the first hidden layer
        Z = X @ self.weights[0] + self.biases[0]  
        A = self.sigmoid(Z)                       
        net_inputs.append(Z)
        activations.append(A)

        # Compute the activations for subsequent layers
        for i in range(1, len(self.weights)):
            Z = activations[-1] @ self.weights[i] + self.biases[i]  
            A = self.sigmoid(Z)                                     
            net_inputs.append(Z)
            activations.append(A)

        return net_inputs, activations
    
    def back_propagate(self, X, y, net_inputs, activations):
        """Perform backward propagation to compute the delta terms for each layer."""      

        # Initialize deltas for all layers
        deltas = [None] * self.n_layers  
        
        # Compute the delta for the output layer
        deltas[-1] = activations[-1] - y  

        # Propagate the deltas backward
        for i in reversed(range(self.n_layers - 1)):  # skip the output layer
            net_input_deriv = self.sigmoid_derivative(net_inputs[i])
            deltas[i] = (deltas[i + 1] @ self.weights[i + 1].T) * net_input_deriv 
            
        return deltas
    
    def update_weights(self, X, activations, deltas):
        """Update the weights and biases."""    

        # Size of the current mini-batch; may be smaller than self.batch_size for the last batch
        batch_size = len(X)
                            
        # Update weights and biases for each layer
        for i in range(self.n_layers):            
            if i == 0:  # Use X for the first layer's activations
                self.weights[i] -= self.learning_rate / batch_size * X.T @ deltas[i]
            else:
                self.weights[i] -= self.learning_rate / batch_size * activations[i - 1].T @ deltas[i]
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0)

    def fit(self, X, y):
        """
        Fit the neural network to the training data using mini-batch gradient descent.

        Arguments:
        - X: Input feature matrix of shape (n_samples, n_features).
        - y: Target values of shape (n_samples, 1).
        """
        n_samples, n_features = X.shape
        y = y.reshape(n_samples, 1)  # Ensure y is a column vector

        # Include the input and output layers in the architecture
        self.layer_sizes = [n_features] + self.hidden_layer_sizes + [1]

        # Initialize weights and biases
        self.initialize_weights()

        # Run the training loop
        for epoch in range(self.n_epochs):
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
                deltas = self.back_propagate(X_batch, y_batch, net_inputs, activations)

                # Update weights and biases
                self.update_weights(X_batch, activations, deltas)

            # Report training loss at specified intervals
            if (epoch + 1) % self.report_interval == 0 or epoch == 0:
                y_pred = self.predict_proba(X)
                loss = self.compute_loss(y, y_pred)
                print(f'Epoch {epoch + 1}, Train loss: {loss:.6f}')

    def initialize_weights(self):
        """Initialize weights with small random numbers and biases with zeros."""
        self.weights = []  
        self.biases = [] 

        for i in range(self.n_layers):   
            # Initialize weights for the connections between layer i and i + 1         
            self.weights.append(self.random_state.randn(self.layer_sizes[i], self.layer_sizes[i + 1]))

            # Initialize biases for layer i + 1 (skipping the input layer)
            self.biases.append(np.zeros(self.layer_sizes[i + 1]))

    def compute_loss(self, y, y_pred):
        """Compute the average log loss."""
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
    
    def predict_proba(self, X):
        """Predict class probabilities for the given inputs."""
        _, activations = self.forward_pass(X)
        return activations[-1]

    def predict(self, X):
        """Predict class labels for the given inputs."""
        prob = self.predict_proba(X)
        y_pred = (prob > 0.5).astype(int)
        return y_pred