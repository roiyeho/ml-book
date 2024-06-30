import numpy as np

class XGBNode:
    """A node object that recursively builds itself to construct a regression tree
    """
    def __init__(self):
        self.is_leaf: bool = False
        self.left_child: XGBNode = None
        self.right_child: XGBNode = None

    def build(self, X, grads, hessians, curr_depth, max_depth, reg_lambda, gamma):
        """Recursively build the node until a stopping criterion is reached
        """
        if len(X) == 1 or curr_depth >= max_depth:
            # Stopping criterion (1): there is only one sample left at this node
            # Stopping criterion (2): max depth of the tree has been reached 
            self.is_leaf = True
            self.weight = self.calc_leaf_weight(grads, hessians, reg_lambda)
            return
        
        best_gain, best_split = self.find_best_split(X, grads, hessians, reg_lambda)
        
        if best_gain < gamma:
            # Stopping criterion (3): the best gain is less than the minimum split gain 
            self.is_leaf = True
            self.weight = self.calc_leaf_weight(grads, hessians, reg_lambda)
            return        
        else:
            # Split the node according to the best split found
            feature_idx, threshold, left_samples_idx, right_samples_idx = best_split

            self.split_feature_idx = feature_idx
            self.split_threshold = threshold

            self.left_child = XGBNode()
            self.left_child.build(X[left_samples_idx],
                                grads[left_samples_idx],
                                hessians[left_samples_idx],
                                curr_depth + 1,
                                max_depth, reg_lambda, gamma)
            
            self.right_child = XGBNode()
            self.right_child.build(X[right_samples_idx],
                                grads[right_samples_idx],
                                hessians[right_samples_idx],
                                curr_depth + 1,
                                max_depth, reg_lambda, gamma)
            
    def calc_leaf_weight(self, grads, hessians, reg_lambda):
        """Calculate the optimal weight of this leaf node (eq.(5) in [1])
        """
        return -np.sum(grads) / (np.sum(hessians) + reg_lambda)
       
    def find_best_split(self, X, grads, hessians, reg_lambda):
        """Scan through every feature and find the best split point (Algorithm 1 in [1])
        """
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
            for i in range(X.shape[0] - 1):   
                G_left += grads[sorted_samples_idx[i]]
                H_left += hessians[sorted_samples_idx[i]]

                G_right = G - G_left
                H_right = H - H_left
                curr_gain = self.calc_split_gain(G, H, G_left, H_left, G_right, H_right, reg_lambda)

                if curr_gain > best_gain:
                    # Update the properties of the best split
                    best_gain = curr_gain     
                    feature_idx = j 
                    threshold = X[sorted_samples_idx[i]][j]
                    left_samples_idx = sorted_samples_idx[:i + 1]
                    right_samples_idx = sorted_samples_idx[i + 1:]
                    best_split = (feature_idx, threshold, left_samples_idx, right_samples_idx)

        return best_gain, best_split
    
    def calc_split_gain(self, G, H, G_left, H_left, G_right, H_right, reg_lambda):
        """Compute the loss reduction (eq. (7) in [1])
        """
        def calc_term(g, h):
            return g**2 / (h + reg_lambda)

        gain = calc_term(G_left, H_left) + calc_term(G_right, H_right) - calc_term(G, H)
        return 0.5 * gain
        
    def predict(self, x):
        """Return the score of a given sample x
        """
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature_idx] <= self.split_threshold:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)  