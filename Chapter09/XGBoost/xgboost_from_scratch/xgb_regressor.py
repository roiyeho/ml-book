import numpy as np

from xgb_base_model import XGBBaseModel
from sklearn.metrics import r2_score

class XGBRegressor(XGBBaseModel):
    """An XGBoost estimator for regression tasks
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
        # The initial prediction is the mean of the targets
        return np.mean(y)

    def calc_gradients(self, y, out):
        # The first order gradients are twice the residuals
        grads = 2 * (out - y)
        return grads

    def calc_hessians(self, y, out):
        # The second order gradients are equal to the constant 2
        hessians = np.full(len(y), 2)
        return hessians
    
    def predict(self, X):
        # The predicted labels are the same as the output values
        y_pred = self.get_output_values(X)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)