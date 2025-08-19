import numpy as np
from xgb_base_model import XGBBaseModel
from sklearn.base import ClassifierMixin

class XGBClassifier(XGBBaseModel, ClassifierMixin):
    """An XGBoost estimator for binary classification tasks."""
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, 
                 reg_lambda=1, gamma=0, verbose=0):
        super().__init__(n_estimators, max_depth, learning_rate, reg_lambda, gamma, verbose)

    def get_base_prediction(self, y):
        """Return the log-odds of the positive class as the initial prediction."""
        prob = np.sum(y == 1) / len(y)
        return np.log(prob / (1 - prob))
    
    def sigmoid(self, x):
        """Apply the sigmoid function."""
        return 1 / (1 + np.exp(-x))
    
    def calc_gradients(self, y, output):
        """Compute the gradients of the log loss function."""
        prob = self.sigmoid(output)    
        grads = prob - y
        return grads

    def calc_hessians(self, y, output):
        """Compute the Hessians of the log loss function."""
        prob = self.sigmoid(output)
        hessians = prob * (1 - prob)
        return hessians
    
    def predict_proba(self, X):
        """Return the predicted probabilities of the positive class."""
        log_odds = self.get_output_values(X)
        prob = self.sigmoid(log_odds)
        return prob
    
    def predict(self, X):
        """Return binary class labels based on the predicted probabilities."""
        prob = self.predict_proba(X)
        y_pred = np.where(prob > 0.5, 1, 0)
        return y_pred