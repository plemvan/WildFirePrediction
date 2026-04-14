"""
Decision tree node for the XGBoost from-scratch implementation.

Based on the XGBoost paper:
    Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
    KDD '16. https://doi.org/10.1145/2939672.2939785
"""

import numpy as np


class Node:
    """
    A single decision tree node using gradient and Hessian split logic.

    Recursively splits data until a stopping criterion is met (max depth,
    minimum samples, or no gain-improving split), at which point it becomes
    a leaf and stores the optimal leaf weight omega*.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    grad : np.ndarray
        First-order derivatives of the loss, shape (n_samples,).
    row_indices : np.ndarray
        Indices of samples assigned to this node.
    hess : np.ndarray
        Second-order derivatives of the loss (Hessians), shape (n_samples,).
    max_depth : int
        Maximum allowed tree depth.
    min_leaf : int
        Minimum number of samples required to attempt a split.
    lambda_ : float
        L2 regularization on leaf weights.
    gamma : float
        Minimum gain required to accept a split.
    min_child_weight : float
        Minimum sum of Hessians in each child node.
    solver : str
        Split strategy: 'greedy', 'global', or 'local'.
    eps : float, optional
        Approximation factor for weighted quantile sketch (default=0.1).
    global_candidates : dict or None, optional
        Pre-computed candidate split points keyed by feature index.
        Only used when solver='global'.
    """

    def __init__(
        self,
        X,
        grad,
        row_indices,
        hess,
        max_depth,
        min_leaf,
        lambda_,
        gamma,
        min_child_weight,
        solver,
        eps=0.1,
        global_candidates=None,
    ):
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

        # Tree structure
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.depth = 0

    @property
    def is_leaf(self):
        """Return True if this node is a leaf."""
        return self.leaf_value is not None

    def compute_omega(self):
        """
        Compute the optimal leaf weight omega* (equation (5) in [1]).

        Returns
        -------
        float
            -G / (H + lambda_)
        """
        G = np.sum(self.grad[self.row_indices])
        H = np.sum(self.hess[self.row_indices])
        return -G / (H + self.lambda_)

    def gain(self, G_left, H_left, G_right, H_right):
        """
        Compute the gain for a candidate split.

        Parameters
        ----------
        G_left, H_left : float
            Sum of gradients and Hessians in the left child.
        G_right, H_right : float
            Sum of gradients and Hessians in the right child.

        Returns
        -------
        float
            Split gain; negative means the split is not beneficial.
        """
        return (
            0.5
            * (
                G_left**2 / (H_left + self.lambda_)
                + G_right**2 / (H_right + self.lambda_)
                - (G_left + G_right) ** 2 / (H_left + H_right + self.lambda_)
            )
            - self.gamma
        )

    def find_best_split_greedy(self, feature_index):
        """
        Exact greedy split: evaluate all unique thresholds for one feature.

        Parameters
        ----------
        feature_index : int

        Returns
        -------
        best_gain : float
        best_split_value : float or None
        """
        X_col = self.X[self.row_indices, feature_index]
        order = np.argsort(X_col)
        X_sorted = X_col[order]
        g_sorted = self.grad[self.row_indices][order]
        h_sorted = self.hess[self.row_indices][order]

        G_total = g_sorted.sum()
        H_total = h_sorted.sum()
        G_left = H_left = 0.0
        best_gain = -np.inf
        best_value = None

        for i in range(1, len(X_sorted)):
            G_left += g_sorted[i - 1]
            H_left += h_sorted[i - 1]
            G_right = G_total - G_left
            H_right = H_total - H_left

            if X_sorted[i] == X_sorted[i - 1]:
                continue
            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue

            g = self.gain(G_left, H_left, G_right, H_right)
            if g > best_gain:
                best_gain = g
                best_value = (X_sorted[i] + X_sorted[i - 1]) / 2

        return best_gain, best_value

    def weighted_quantile_sketch(self, feature_index):
        """
        Generate candidate split points using Hessian-weighted quantiles.

        Parameters
        ----------
        feature_index : int

        Returns
        -------
        list of float
        """
        X_col = self.X[self.row_indices, feature_index]
        h_sub = self.hess[self.row_indices]

        order = np.argsort(X_col)
        X_sorted = X_col[order]
        h_sorted = h_sub[order]

        cumsum = np.cumsum(h_sorted)
        norm = cumsum / cumsum[-1]

        indices = []
        for q in np.arange(0, 1 + self.eps, self.eps):
            idx = np.searchsorted(norm, q, side="left")
            if idx < len(X_sorted):
                indices.append(idx)

        indices = sorted(set(indices))
        candidates = [X_sorted[i] for i in indices if 0 < i < len(X_sorted) - 1]
        return candidates if candidates else [np.median(X_sorted)]

    def _evaluate_candidates(self, X_col, g_sub, h_sub, candidates):
        """
        Evaluate a list of candidate thresholds and return the best split.

        Parameters
        ----------
        X_col : np.ndarray
        g_sub : np.ndarray
        h_sub : np.ndarray
        candidates : list of float

        Returns
        -------
        best_gain : float
        best_value : float or None
        """
        G_total = g_sub.sum()
        H_total = h_sub.sum()
        best_gain = -np.inf
        best_value = None

        for split_val in candidates:
            mask = X_col <= split_val
            G_left = g_sub[mask].sum()
            H_left = h_sub[mask].sum()
            G_right = G_total - G_left
            H_right = H_total - H_left

            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue

            g = self.gain(G_left, H_left, G_right, H_right)
            if g > best_gain:
                best_gain = g
                best_value = split_val

        return best_gain, best_value

    def find_best_split_local(self, feature_index):
        """Local variant: re-compute quantile candidates at each node."""
        X_col = self.X[self.row_indices, feature_index]
        g_sub = self.grad[self.row_indices]
        h_sub = self.hess[self.row_indices]
        candidates = self.weighted_quantile_sketch(feature_index)
        return self._evaluate_candidates(X_col, g_sub, h_sub, candidates)

    def find_best_split_global(self, feature_index):
        """Global variant: use pre-computed candidates shared across all nodes."""
        X_col = self.X[self.row_indices, feature_index]
        g_sub = self.grad[self.row_indices]
        h_sub = self.hess[self.row_indices]

        if self.global_candidates and feature_index in self.global_candidates:
            candidates = self.global_candidates[feature_index]
        else:
            candidates = self.weighted_quantile_sketch(feature_index)

        return self._evaluate_candidates(X_col, g_sub, h_sub, candidates)

    def find_split(self):
        """
        Recursively find and apply the best split across all features.

        Sets leaf_value if this node becomes a leaf, otherwise builds
        left/right children and recurses.
        """
        if self.depth >= self.max_depth or len(self.row_indices) <= self.min_leaf:
            self.leaf_value = self.compute_omega()
            return

        best_gain = -np.inf
        best_feature = best_value = None

        for col in range(self.n_features):
            if self.solver == "greedy":
                g, v = self.find_best_split_greedy(col)
            elif self.solver == "local":
                g, v = self.find_best_split_local(col)
            elif self.solver == "global":
                g, v = self.find_best_split_global(col)
            else:
                raise ValueError(
                    f"Unknown solver {self.solver!r}. "
                    "Choose 'greedy', 'local', or 'global'."
                )
            if g > best_gain:
                best_gain, best_feature, best_value = g, col, v

        if best_gain <= 0 or best_value is None:
            self.leaf_value = self.compute_omega()
            return

        self.split_feature = best_feature
        self.split_value = best_value

        mask = self.X[self.row_indices, self.split_feature] <= self.split_value
        left_idx = self.row_indices[mask]
        right_idx = self.row_indices[~mask]

        if len(left_idx) == 0 or len(right_idx) == 0:
            self.leaf_value = self.compute_omega()
            return

        kwargs = dict(
            max_depth=self.max_depth,
            min_leaf=self.min_leaf,
            lambda_=self.lambda_,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            solver=self.solver,
            eps=self.eps,
            global_candidates=self.global_candidates,
        )
        self.left = Node(self.X, self.grad, left_idx, self.hess, **kwargs)
        self.right = Node(self.X, self.grad, right_idx, self.hess, **kwargs)
        self.left.depth = self.depth + 1
        self.right.depth = self.depth + 1
        self.left.find_split()
        self.right.find_split()

    def predict(self, X):
        """
        Vectorized prediction for a batch of samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.is_leaf:
            return np.full(X.shape[0], self.leaf_value)

        mask = X[:, self.split_feature] <= self.split_value
        preds = np.empty(X.shape[0], dtype=float)
        if self.left is not None:
            preds[mask] = self.left.predict(X[mask])
        if self.right is not None:
            preds[~mask] = self.right.predict(X[~mask])
        return preds
