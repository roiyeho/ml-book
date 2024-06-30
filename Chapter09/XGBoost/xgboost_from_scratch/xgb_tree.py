from xgb_node import XGBNode

class XGBTree:
    """A single tree object that will be used for gradient boosting
    """
    def __init__(self):
        self.root: XGBNode = None

    def build(self, X, grads, hessians, max_depth, reg_lambda, gamma):
        """Recursively build the root node of the tree 
        """
        self.root = XGBNode()
        curr_depth = 0
        self.root.build(X, grads, hessians, curr_depth, max_depth, reg_lambda, gamma)

    def predict(self, x):
        """Return the score of a given sample x
        """
        return self.root.predict(x)