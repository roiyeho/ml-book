import numpy as np

from sklearn.base import BaseEstimator
from xgb_tree import XGBTree
from typing import List
from abc import ABC, abstractmethod

class XGBBaseModel(ABC, BaseEstimator):
    """Base class for the XGBoost estimators
    """
    def __init__(
        self,
        n_estimators=100,     # The number of trees (boosting rounds)
        max_depth=6,          # Maximum depth of a tree        
        learning_rate=0.3,    # Step size shrinkage applied to the leaf weights
        reg_lambda=1,         # L2 regularization term on the leaf weights
        gamma=0,              # Minimum loss reduction required to split a node
        verbose=0             # Verbosity of the log messages (change to 1 for debug mode)
    ):
        self.n_estimators = n_estimators        
        self.max_depth = max_depth       
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.verbose = verbose
        
    def fit(self, X, y):
        """Build an ensemble of trees for the given training set
        """
        # Get the initial prediction of the ensemble
        self.base_pred = self.get_base_prediction(y)
        
        self.estimators: List[XGBTree] = []
        for i in range(self.n_estimators):
            # Compute the first and second order gradients of the loss function with respect
            # to the predictions of the current ensemble
            out = self.get_output_values(X)
            grads = self.calc_gradients(y, out)
            hessians = self.calc_hessians(y, out)

            # Add a new tree to the ensemble
            tree = XGBTree()
            tree.build(X, grads, hessians, self.max_depth, self.reg_lambda, self.gamma)
            self.estimators.append(tree)
            
            if self.verbose and i % 10 == 0:
                print(f'Boosting iteration {i}')
        return self
    
    def get_output_values(self, X):
        """Return the predicted values of the ensemble for the given data set
        """
        # Initialize the output values with the base prediction
        output = np.full(X.shape[0], self.base_pred)

        # Add the predictions of the base trees scaled by the learning rate
        if len(self.estimators) > 0:
            for i in range(len(X)):            
                output[i] += np.sum(self.learning_rate * estimator.predict(X[i]) 
                                        for estimator in self.estimators)
        return output
    
    @abstractmethod
    def get_base_prediction(self, y):
        """Return the initial prediction of the model"""
        pass

    @abstractmethod
    def calc_gradients(self, y, out):
        """Calculate the first order gradients""" 
        pass

    @abstractmethod
    def calc_hessians(self, y, out):
        """Calculate the second order gradients"""
        pass

    @abstractmethod
    def predict(self, X):
        """Return the final predicted labels for the given samples"""
        pass