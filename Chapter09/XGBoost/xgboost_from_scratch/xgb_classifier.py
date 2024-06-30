import numpy as np

from xgb_base_model import XGBBaseModel
from sklearn.metrics import accuracy_score

class XGBClassifier(XGBBaseModel):
    """An XGBoost estimator for binary classification tasks
    """
    def __init__(
        self, 
        n_estimators=100, 
        max_depth=6,         
        learning_rate=0.3, 
        reg_lambda=1, 
        gamma=0,
        verbose=0
    ):
        super().__init__(n_estimators, max_depth, learning_rate, reg_lambda, gamma, verbose)

    def get_base_prediction(self, y):
        # The initial prediction is the log odds of the positive class
        prob = np.sum(y == 1) / len(y)
        return np.log(prob / (1 - prob))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def calc_gradients(self, y, out):
        # The first order gradients are the residuals between the predicted probabilities 
        # (sigmoid of the log odds) and the true labels
        prob = self.sigmoid(out)    
        grads = prob - y
        return grads

    def calc_hessians(self, y, out):
        # The second order gradients are p(1 - p) where p is the predicted probability
        prob = self.sigmoid(out)
        hessians = prob * (1 - prob)
        return hessians
    
    def predict_proba(self, X):
        # The predicted probability is the sigmoid of the log odds
        log_odds = self.get_output_values(X)
        prob = self.sigmoid(log_odds)
        return prob
    
    def predict(self, X):
        # Assign a label of 1 if the probability of the positive class is > 0.5,
        # otherwise assign a label of 0
        prob = self.predict_proba(X)
        y_pred = np.where(prob > 0.5, 1, 0)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)