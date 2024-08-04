import numpy as np

class XGBNode:
    """A node within the XGBoost regression tree"""
    def __init__(self):
        self.is_leaf: bool = False
        self.left_child: XGBNode = None
        self.right_child: XGBNode = None

    def build(self, X, grads, hessians, curr_depth, max_depth, reg_lambda, gamma):
        """Recursively build the node until a stopping criterion is reached."""
        # Check if only one sample is left or the tree's max depth has been reached
        if len(X) == 1 or curr_depth >= max_depth:
            self.is_leaf = True
            self.weight = self.calc_leaf_weight(grads, hessians, reg_lambda)
            return
        
        best_gain, best_split = self.find_best_split(X, grads, hessians, reg_lambda, gamma)
        
        # Check if the maximum possible gain is less than 0
        if best_gain < 0:
            self.is_leaf = True
            self.weight = self.calc_leaf_weight(grads, hessians, reg_lambda)
            return        
        else:
            # Split the node into two child nodes
            feature_idx, threshold, left_samples_idx, right_samples_idx = best_split
            self.split_feature_idx = feature_idx
            self.split_threshold = threshold
            self.left_child = XGBNode()
            self.left_child.build(X[left_samples_idx], grads[left_samples_idx],
                                    hessians[left_samples_idx], curr_depth + 1,
                                    max_depth, reg_lambda, gamma)            
            self.right_child = XGBNode()
            self.right_child.build(X[right_samples_idx], grads[right_samples_idx],
                                    hessians[right_samples_idx], curr_depth + 1,
                                    max_depth, reg_lambda, gamma)
            
    def calc_leaf_weight(self, grads, hessians, reg_lambda):
        """Calculate the weight for a leaf node."""
        return -np.sum(grads) / (np.sum(hessians) + reg_lambda)
       
    def find_best_split(self, X, grads, hessians, reg_lambda, gamma):
        """Find the optimal split point by evaluating each possible split of the data."""
        G = np.sum(grads)
        H = np.sum(hessians)
        best_gain = float('-inf')   
        best_split = None
        
        # Iterate over all the possible features
        for j in range(X.shape[1]):
            G_left, H_left = 0, 0

            # Sort the samples according to their value in the current feature
            sorted_samples_idx = np.argsort(X[:, j])

            # Calculate the gain of every possible split point
            for i in range(0, X.shape[0] - 1):   
                G_left += grads[sorted_samples_idx[i]]
                H_left += hessians[sorted_samples_idx[i]]
                G_right = G - G_left
                H_right = H - H_left
                curr_gain = self.calc_split_gain(G, H, G_left, H_left, G_right, H_right, 
                                                    reg_lambda, gamma)

                if curr_gain > best_gain:
                    # Update the best split
                    best_gain = curr_gain     
                    threshold = (X[sorted_samples_idx[i], j] + X[sorted_samples_idx[i + 1], j]) / 2
                    left_samples_idx = sorted_samples_idx[:i + 1]
                    right_samples_idx = sorted_samples_idx[i + 1:]
                    best_split = (j, threshold, left_samples_idx, right_samples_idx)

        return best_gain, best_split
    
    def calc_split_gain(self, G, H, G_left, H_left, G_right, H_right, reg_lambda, gamma):
        """Calculates the gain from a split."""
        def calc_term(g, h):
            return g**2 / (h + reg_lambda)

        gain = 0.5 * (calc_term(G_left, H_left) + calc_term(G_right, H_right) 
                        - calc_term(G, H)) - gamma
        return gain
        
    def predict(self, x):
        """Predict the outcome for a sample by traversing the tree to the appropriate leaf."""
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature_idx] <= self.split_threshold:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)  