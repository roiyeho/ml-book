from xgb_node import XGBNode

class XGBTree:
    """Represents a regression tree in the XGBoost ensemble."""
    def __init__(self):
        self.root: XGBNode = None

    def build(self, X, grads, hessians, max_depth, reg_lambda, gamma):
        """Initiate the recursive construction of the tree starting from the root."""
        self.root = XGBNode()
        curr_depth = 0
        self.root.build(X, grads, hessians, curr_depth, max_depth, reg_lambda, gamma)

    def predict(self, x):
        """Predict the output for a given sample by traversing the tree."""
        if self.root is not None:
            return self.root.predict(x)
        else:
            raise Exception("The tree has not been built yet.")