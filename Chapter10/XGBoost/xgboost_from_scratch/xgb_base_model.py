import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import List
from xgb_tree import XGBTree

class XGBBaseModel(ABC, BaseEstimator):
    """Base class for XGBoost estimators, compatible with the Scikit-Learn API."""
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3,    
                 reg_lambda=1, gamma=0, verbose=0):
        self.n_estimators = n_estimators    # Number of trees      
        self.max_depth = max_depth          # Maximum tree depth
        self.learning_rate = learning_rate  # Shrinkage factor
        self.reg_lambda = reg_lambda        # L2 regularization term
        self.gamma = gamma                  # Minimum loss reduction to split a node
        self.verbose = verbose              # Verbosity level for logging
        
    def fit(self, X, y):
        """Build an ensemble of decision trees from the input data."""
        # Initialize predictions with a constant base value 
        self.base_pred = self.get_base_prediction(y)
        
        self.estimators: List[XGBTree] = []
        for i in range(self.n_estimators):
            # Get predictions of the current ensemble
            output = self.get_output_values(X)

            # Compute gradients and Hessians of the loss function  
            grads = self.calc_gradients(y, output)
            hessians = self.calc_hessians(y, output)

            # Build a new tree to correct the residual errors 
            tree = XGBTree()
            tree.build(X, grads, hessians, self.max_depth, self.reg_lambda, self.gamma)
            self.estimators.append(tree)
            
            if self.verbose and i % 10 == 0:
                print(f'Boosting iteration {i}')
        return self

    def get_output_values(self, X):
        """Compute the ensemble predictions for the given dataset."""
        # Initialize the output with the base prediction
        output = np.full(X.shape[0], self.base_pred)

        # Add the predictions from each tree, scaled by the learning rate
        for estimator in self.estimators:
            output += self.learning_rate * np.array([estimator.predict(x) for x in X])
        return output
    
    @abstractmethod
    def get_base_prediction(self, y):
        """Return the initial prediction value for the model."""
        pass

    @abstractmethod
    def calc_gradients(self, y, output):
        """Compute the first-order derivatives (gradients) of the loss function.""" 
        pass

    @abstractmethod
    def calc_hessians(self, y, output):
        """Compute the second-order derivatives (Hessians) of the loss function."""
        pass

    @abstractmethod
    def predict(self, X):
        """Return the final predictions for the input samples."""
        pass