import numpy as np
import pandas as pd
import math
# We authorized oursleves to use pandas only for data preprocessing and loading. The XGBoost implementation must be done from scratch.

class Node:
    """
    Decision tree node for XGBoost. Recursively splits data using gradient and Hessian information.
    Becomes a leaf when no split improves gain. Based on XGBoost paper [1].
    
    Parameters :
    
    X : np.ndarray
        Feature matrix of the training data.
    grad : np.ndarray
        Negative gradient (first-order derivative of the loss) for the samples.
    row_indices : np.ndarray
        Indices of training samples for this node.
    hess : np.ndarray
        Hessian (second-order derivative of the loss) for the samples.
    max_depth : int
        Maximum tree depth.
    min_leaf : int
        Minimum samples required for a node.
    lambda_ : float
        L2 regularization term on leaf weights.
    gamma : float
        Regularization term on the number of leaves.
    min_child_weight : float
        Minimum sum of Hessians in each child node.
    solver : str
        Split strategy: 'greedy', 'global', or 'local'.
    eps : float
        Approximation factor for weighted quantile sketch (default=0.1).
    global_candidates : dict or None
        Pre-computed candidates for global variant.
    """
    def __init__(self, X, grad, row_indices, hess, max_depth, min_leaf, lambda_, gamma, min_child_weight, solver, eps=0.1, global_candidates=None):
        self.X = X
        self.grad = grad
        self.hess = hess
        self.row_indices = row_indices

        self.n_samples, self.n_features = X.shape
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.solver = solver
        self.eps = eps
        self.global_candidates = global_candidates  
        self.column_subsample = self.n_features  # No column subsampling 
                                                 

        # Tree structure
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.depth = 0
        self.best_gain = -np.inf

    @property
    def split_col(self):
        """Splits a colum based on row_indices and split_feature."""
        if self.split_feature is None:
            return None
        if self.row_indices.size == 0:
            return None
        return self.X[self.row_indices, self.split_feature]
    
    @property
    def is_leaf(self):
        """Determines if the node is a leaf."""
        return self.leaf_value is not None

    def compute_omega(self):
        """Computes the optimal leaf value (omega^{star} as the equation (5) in [1]) for this node."""
        G = np.sum(self.grad[self.row_indices])
        H = np.sum(self.hess[self.row_indices])
        return -G / (H + self.lambda_)

    def gain(self, G_left, H_left, G_right, H_right):
        """
        Compute the gain for a potential split given sums of gradients and Hessians
        in left and right nodes.
        """
        return 0.5 * ((G_left**2 / (H_left + self.lambda_)) +
                    (G_right**2 / (H_right + self.lambda_)) -
                    ((G_left + G_right)**2 / (H_left + H_right + self.lambda_))) - self.gamma

    def find_best_split_greedy(self, feature_index):
        """Exact greedy split: evaluates all possible split points for a feature."""
        X_column = self.X[self.row_indices, feature_index]
        sorted_indices = np.argsort(X_column)
        X_sorted = X_column[sorted_indices]
        grad_sorted = self.grad[self.row_indices][sorted_indices]
        hess_sorted = self.hess[self.row_indices][sorted_indices]

        G_total = np.sum(grad_sorted)
        H_total = np.sum(hess_sorted)

        G_left, H_left = 0.0, 0.0
        best_gain = -np.inf
        best_split_value = None

        for i in range(1, len(X_sorted)):
            G_left += grad_sorted[i - 1]
            H_left += hess_sorted[i - 1]
            G_right = G_total - G_left
            H_right = H_total - H_left

            if X_sorted[i] == X_sorted[i - 1]:
                continue

            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue

            gain = self.gain(G_left, H_left, G_right, H_right)

            if gain > best_gain:
                best_gain = gain
                best_split_value = (X_sorted[i] + X_sorted[i - 1]) / 2

        return best_gain, best_split_value
     
    def weighted_quantile_sketch(self, feature_index):
        """Generate candidate split points using Hessian-weighted quantiles."""
        X_column = self.X[self.row_indices, feature_index]
        hess_subset = self.hess[self.row_indices]
        
        sorted_indices = np.argsort(X_column)
        X_sorted = X_column[sorted_indices]
        hess_sorted = hess_subset[sorted_indices]
        
        # Compute cumulative Hessian weights
        cumsum_hess = np.cumsum(hess_sorted)
        total_hess = cumsum_hess[-1]
        normalized_cumsum = cumsum_hess / total_hess
        
        # Find quantile points
        quantile_indices = []
        for q in np.arange(0, 1 + self.eps, self.eps):
            idx = np.searchsorted(normalized_cumsum, q, side='left')
            if idx < len(X_sorted):
                quantile_indices.append(idx)
        
        # Convert to candidate split values
        quantile_indices = sorted(set(quantile_indices))
        candidates = [X_sorted[i] for i in quantile_indices if i > 0 and i < len(X_sorted) - 1]
        
        return candidates if candidates else [np.median(X_sorted)]
    
    def _evaluate_candidates(self, X_column, grad_subset, hess_subset, candidates):
        """Evaluate a list of candidate split points, return best gain and split value."""
        G_total = np.sum(grad_subset)
        H_total = np.sum(hess_subset)
        
        best_gain = -np.inf
        best_split_value = None
        
        for split_val in candidates:
            mask_left = X_column <= split_val
            
            G_left = np.sum(grad_subset[mask_left])
            H_left = np.sum(hess_subset[mask_left])
            G_right = G_total - G_left
            H_right = H_total - H_left
            
            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue
            
            gain = self.gain(G_left, H_left, G_right, H_right)
            
            if gain > best_gain:
                best_gain = gain
                best_split_value = split_val
        
        return best_gain, best_split_value

    def find_best_split_local(self, feature_index):
        """Local variant: re-compute candidates at each node for refinement."""
        X_column = self.X[self.row_indices, feature_index]
        grad_subset = self.grad[self.row_indices]
        hess_subset = self.hess[self.row_indices]
        
        candidates = self.weighted_quantile_sketch(feature_index)
        
        return self._evaluate_candidates(X_column, grad_subset, hess_subset, candidates)

    def find_best_split_global(self, feature_index):
        """Global variant: use pre-computed candidates reused across all tree levels."""
        X_column = self.X[self.row_indices, feature_index]
        grad_subset = self.grad[self.row_indices]
        hess_subset = self.hess[self.row_indices]

        if self.global_candidates is None or feature_index not in self.global_candidates:
            candidates = self.weighted_quantile_sketch(feature_index)
        else:
            candidates = self.global_candidates[feature_index]
        
        return self._evaluate_candidates(X_column, grad_subset, hess_subset, candidates)

    def find_split(self):
        """Recursively find best split across all features using selected solver."""
        if self.depth >= self.max_depth or len(self.row_indices) <= self.min_leaf:
            self.leaf_value = self.compute_omega()
            return

        best_gain = -np.inf
        best_feature = None
        best_value = None

        # Loop over all features
        for col in range(self.n_features):
            if self.solver == 'greedy':
                gain, split_value = self.find_best_split_greedy(col)
            elif self.solver == 'global':
                gain, split_value = self.find_best_split_global(col)
            elif self.solver == 'local':
                gain, split_value = self.find_best_split_local(col)
            if gain > best_gain:
                best_gain = gain
                best_feature = col
                best_value = split_value

        # No good split found -> leaf
        if best_gain <= 0:
            self.leaf_value = self.compute_omega()
            return

        # Otherwise, store the best split
        self.split_feature = best_feature
        self.split_value = best_value

        # Split data
        left_indices = self.row_indices[self.X[self.row_indices, self.split_feature] <= self.split_value]
        right_indices = self.row_indices[self.X[self.row_indices, self.split_feature] > self.split_value]

        if len(left_indices) == 0 or len(right_indices) == 0:
            self.leaf_value = self.compute_omega()
            return

        self.left = Node(self.X, self.grad, left_indices, self.hess,
                        self.max_depth, self.min_leaf, self.lambda_, self.gamma,
                        self.min_child_weight, self.solver, self.eps, self.global_candidates)
        self.right = Node(self.X, self.grad, right_indices, self.hess,
                        self.max_depth, self.min_leaf, self.lambda_, self.gamma,
                        self.min_child_weight, self.solver, self.eps, self.global_candidates)

        self.left.depth = self.depth + 1
        self.right.depth = self.depth + 1

        self.left.find_split()
        self.right.find_split()
    
    def predict(self, X):
        """Vectorized prediction for all rows."""
        # If node is a leaf, return the same value for all rows
        if self.is_leaf:
            return np.full(X.shape[0], self.leaf_value)

        # Otherwise, split the rows based on the split feature
        mask_left = X[:, self.split_feature] <= self.split_value
        mask_right = ~mask_left

        preds = np.empty(X.shape[0], dtype=float)

        # Predict recursively for left and right children
        if self.left is not None:
            preds[mask_left] = self.left.predict(X[mask_left])
        if self.right is not None:
            preds[mask_right] = self.right.predict(X[mask_right])

        return preds
        

class XGBoostTree:
    '''
    Single XGBoost tree using gradient boosting with configurable split strategy.
    
    '''
    def _compute_global_candidates(self, X, hess, eps):
        """Pre-compute global candidate split points for all features once at tree initialization."""
        n_samples, n_features = X.shape
        global_candidates = {}
        
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            sorted_indices = np.argsort(X_column)
            X_sorted = X_column[sorted_indices]
            hess_sorted = hess[sorted_indices]
            
            cumsum_hess = np.cumsum(hess_sorted)
            total_hess = cumsum_hess[-1]
            normalized_cumsum = cumsum_hess / total_hess
            
            # Find quantile points
            quantile_indices = []
            for q in np.arange(0, 1 + eps, eps):
                idx = np.searchsorted(normalized_cumsum, q, side='left')
                if idx < len(X_sorted):
                    quantile_indices.append(idx)
            
            quantile_indices = sorted(set(quantile_indices))
            candidates = [X_sorted[i] for i in quantile_indices if i > 0 and i < len(X_sorted) - 1]
            
            global_candidates[feature_idx] = candidates if candidates else [np.median(X_sorted)]
        
        return global_candidates
    
    def fit(self, X, grad, hess, min_leaf = 5, min_child_weight = 1 ,max_depth = 10, lambda_ = 1, gamma = 1, solver = 'greedy', eps = 0.1):
        """ Fit the XGBoost tree to the data.

        Parameters:

        X : np.ndarray
            Feature matrix.
        grad : np.ndarray
            First-order gradients.
        hess : np.ndarray
            Second-order gradients (Hessians).
        min_leaf : int (default=5)
            Minimum samples per node.
        min_child_weight : float (default=1)
            Minimum sum of Hessians in each child.
        max_depth : int (default=10)
            Maximum tree depth.
        lambda_ : float (default=1)
            L2 regularization on leaf weights.
        gamma : float (default=1)
            Regularization on number of leaves.
        solver : str (default='greedy')
            Split strategy: 'greedy', 'global', or 'local'.
        eps : float (default=0.1)
            Approximation factor for weighted quantile sketch.
        """
        global_candidates = None
        if solver == 'global':
            global_candidates = self._compute_global_candidates(X, hess, eps)
        
        self.tree = Node(X, grad, np.array(np.arange(len(X))), hess, max_depth, min_leaf, lambda_, gamma, min_child_weight, solver, eps, global_candidates)
        self.tree.find_split()
        return self
    
    def predict(self, X):
        return self.tree.predict(X)
   
   
class XGBoostClassifier:
    '''
    XGBoost classifier using gradient boosting with logistic loss.
    '''
    def __init__(self):
        self.trees = []
    
    @staticmethod
    def sigmoid(x):
        """Numerically stable sigmoid (avoids overflow for large abs(x))."""
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        result = np.zeros_like(x, dtype=np.float64)

        # For positive x
        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

        # For negative x, avoid large exp(-x)
        result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

        return result

    
    @staticmethod
    def log_odds(column):
        binary_yes = np.count_nonzero(column == 1)
        binary_no  = np.count_nonzero(column == 0)
        return(np.log(binary_yes/binary_no))

    def grad(self, preds, labels):
        """First-order gradient of logistic loss."""
        preds = self.sigmoid(preds)
        return(preds - labels)
    def hess(self, preds):
        """Second-order gradient (Hessian) of logistic loss."""
        preds = self.sigmoid(preds)
        return(preds * (1 - preds))
    
    
    def fit(self, X, y, min_child_weight = 1, max_depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, solver = 'greedy', eps = 0.1):
        """    
        Fit the XGBoost classifier to the data.
        
        Parameters:

        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Binary labels (0 or 1).
        min_child_weight : float (default=1)
            Minimum sum of Hessians in each child node.
        max_depth : int (default=5)
            Maximum tree depth.
        min_leaf : int (default=5)
            Minimum samples per node.
        learning_rate : float (default=0.4)
            Shrinkage parameter for tree predictions.
        boosting_rounds : int (default=5)
            Number of boosting iterations.
        lambda_ : float (default=1.5)
            L2 regularization on leaf weights.
        gamma : float (default=1)
            Regularization on number of leaves.
        solver : str (default='greedy')
            Split strategy: 'greedy', 'global', or 'local'.
        eps : float (default=0.1)
            Approximation factor for weighted quantile sketch.
        """
        self.X, self.y = X, y
        self.max_depth = max_depth
        self.eps = eps
        self.solver = solver
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma
    
        self.base_pred = np.full((X.shape[0],), self.log_odds(y)).astype('float64')
    
        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, max_depth = self.max_depth, min_leaf = self.min_leaf, lambda_ = self.lambda_, gamma = self.gamma, solver = self.solver, eps = self.eps, min_child_weight = self.min_child_weight)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.trees.append(boosting_tree)
          
    def predict_proba(self, X):
        pred = np.full((X.shape[0],), self.log_odds(self.y)).astype('float64')
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
          
        return self.sigmoid(pred)
    
    def predict(self, X):
        pred = np.full((X.shape[0],), self.log_odds(self.y)).astype('float64')
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        
        predicted_probas = self.sigmoid(pred)
        return np.where(predicted_probas > 0.5, 1, 0)


    




