"""
Single XGBoost tree (one boosting step).

Wraps Node with global-candidate pre-computation and exposes a
sklearn-style fit/predict interface.
"""

import numpy as np

from src.models.node import Node


class XGBoostTree:
    """
    Single XGBoost tree fitted on gradient and Hessian residuals.

    Parameters are passed at fit time, mirroring the XGBoost paper's
    algorithm for a single tree.
    """

    def _compute_global_candidates(self, X, hess, eps):
        """
        Pre-compute Hessian-weighted quantile candidates for all features.

        Called once at tree initialisation when solver='global', so that
        every node reuses the same set of split candidates.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        hess : np.ndarray, shape (n_samples,)
        eps : float
            Quantile approximation factor.

        Returns
        -------
        dict
            {feature_index: [candidate_split_values]}
        """
        _, n_features = X.shape
        global_candidates = {}

        for feat in range(n_features):
            X_col = X[:, feat]
            order = np.argsort(X_col)
            X_sorted = X_col[order]
            h_sorted = hess[order]

            cumsum = np.cumsum(h_sorted)
            norm = cumsum / cumsum[-1]

            indices = []
            for q in np.arange(0, 1 + eps, eps):
                idx = np.searchsorted(norm, q, side="left")
                if idx < len(X_sorted):
                    indices.append(idx)

            indices = sorted(set(indices))
            candidates = [X_sorted[i] for i in indices if 0 < i < len(X_sorted) - 1]
            global_candidates[feat] = (
                candidates if candidates else [np.median(X_sorted)]
            )

        return global_candidates

    def fit(
        self,
        X,
        grad,
        hess,
        min_leaf=5,
        min_child_weight=1,
        max_depth=10,
        lambda_=1,
        gamma=1,
        solver="greedy",
        eps=0.1,
    ):
        """
        Fit a single boosting tree.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        grad : np.ndarray, shape (n_samples,)
            First-order gradients.
        hess : np.ndarray, shape (n_samples,)
            Second-order gradients (Hessians).
        min_leaf : int
            Minimum samples per node (default=5).
        min_child_weight : float
            Minimum sum of Hessians per child (default=1).
        max_depth : int
            Maximum tree depth (default=10).
        lambda_ : float
            L2 regularization on leaf weights (default=1).
        gamma : float
            Minimum gain to allow a split (default=1).
        solver : str
            'greedy', 'local', or 'global' (default='greedy').
        eps : float
            Quantile sketch approximation factor (default=0.1).

        Returns
        -------
        self
        """
        global_candidates = None
        if solver == "global":
            global_candidates = self._compute_global_candidates(X, hess, eps)

        self.tree = Node(
            X,
            grad,
            np.arange(len(X)),
            hess,
            max_depth,
            min_leaf,
            lambda_,
            gamma,
            min_child_weight,
            solver,
            eps,
            global_candidates,
        )
        self.tree.find_split()
        return self

    def predict(self, X):
        """
        Return raw leaf scores for X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        return self.tree.predict(X)
