import numpy as np
from xgb_base_model import XGBBaseModel
from sklearn.base import RegressorMixin

class XGBRegressor(XGBBaseModel, RegressorMixin):
    """An XGBoost estimator for regression tasks"""
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, 
                 reg_lambda=1, gamma=0, verbose=0):
        super().__init__(n_estimators, max_depth, learning_rate, reg_lambda, gamma, verbose)

    def get_base_prediction(self, y):
        """Initialize the prediction to the mean of the target variable."""
        return np.mean(y)

    def calc_gradients(self, y, output):
        """Compute the first-order gradients of the squared loss function."""
        grads = 2 * (output - y)
        return grads

    def calc_hessians(self, y, output):
        """Compute the second-order gradients of the squared loss function."""
        hessians = np.full(len(y), 2)
        return hessians
    
    def predict(self, X):
        """Return the ensemble's raw outputs as the final predictions."""
        y_pred = self.get_output_values(X)
        return y_pred