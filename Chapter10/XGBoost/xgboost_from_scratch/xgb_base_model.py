import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import List
from xgb_tree import XGBTree

class XGBBaseModel(ABC, BaseEstimator):
    """Base class for the XGBoost estimators, compatible with Scikit-Learn API."""
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3,    
                 reg_lambda=1, gamma=0, verbose=0):
        self.n_estimators = n_estimators    # Number of boosting rounds      
        self.max_depth = max_depth          # Maximum depth of each tree 
        self.learning_rate = learning_rate  # Shrinkage of leaf weights
        self.reg_lambda = reg_lambda        # L2 regularization on weights
        self.gamma = gamma                  # Minimum loss reduction for splits
        self.verbose = verbose              # Output log information
        
    def fit(self, X, y):
        """Build an ensemble of decision trees on the training data."""
        # Initialize the prediction with a base value 
        self.base_pred = self.get_base_prediction(y)
        
        self.estimators: List[XGBTree] = []
        for i in range(self.n_estimators):
            # Calculate the current predictions of the ensemble and compute the gradients
            # and Hessians of the loss function with repsect to them
            out = self.get_output_values(X)
            grads = self.calc_gradients(y, out)
            hessians = self.calc_hessians(y, out)

            # Build a new tree aimed at correcting the predicion errors of the ensemble 
            tree = XGBTree()
            tree.build(X, grads, hessians, self.max_depth, self.reg_lambda, self.gamma)
            self.estimators.append(tree)
            
            if self.verbose and i % 10 == 0:
                print(f'Boosting iteration {i}')
        return self

    def get_output_values(self, X):
        """Calculate the current predictions of the ensemble for the given dataset."""
        # Initialize the outputs with the base prediction
        output = np.full(X.shape[0], self.base_pred)

        # Sum up the predictions from all trees, adjusted by the learning rate
        if len(self.estimators) > 0:
            for i in range(len(X)):            
                output[i] += np.sum(self.learning_rate * estimator.predict(X[i]) 
                                    for estimator in self.estimators)
        return output
    
    @abstractmethod
    def get_base_prediction(self, y):
        """Define initial model prediction."""
        pass

    @abstractmethod
    def calc_gradients(self, y, out):
        """Compute first-order gradients.""" 
        pass

    @abstractmethod
    def calc_hessians(self, y, out):
        """Compute second-order gradients (Hessians)."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate final predictions for the input samples."""
        pass