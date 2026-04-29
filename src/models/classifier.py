"""
XGBoost binary classifier using gradient boosting with logistic loss.

Composes XGBoostTree objects iteratively, shrinking each tree's
contribution by the learning rate.
"""

import numpy as np

from src.models.tree import XGBoostTree


class XGBoostClassifier:
    """
    Binary XGBoost classifier built from scratch using logistic loss.

    Each boosting round fits one XGBoostTree on the current gradient
    and Hessian residuals, then adds its predictions (scaled by
    learning_rate) to the running score.

    Attributes
    ----------
    trees : list of XGBoostTree
        Accumulated boosting trees after calling fit().

    Examples
    --------
    >>> import numpy as np
    >>> from src.models.classifier import XGBoostClassifier
    >>> X = np.random.rand(200, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> clf = XGBoostClassifier()
    >>> clf.fit(X, y, boosting_rounds=3, max_depth=3)
    >>> clf.predict(X).shape
    (200,)
    """

    def __init__(self):
        self.trees = []

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sigmoid(x):
        """
        Numerically stable sigmoid.

        Uses separate branches for positive and negative inputs to avoid
        floating-point overflow.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray
        """
        result = np.zeros_like(x, dtype=np.float64)
        pos = x >= 0
        result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        result[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
        return result

    @staticmethod
    def log_odds(y):
        """
        Compute the log-odds of a binary target as the base prediction.

        Parameters
        ----------
        y : np.ndarray
            Binary array of 0s and 1s.

        Returns
        -------
        float
        """
        n_pos = np.count_nonzero(y == 1)
        n_neg = np.count_nonzero(y == 0)
        return np.log(n_pos / n_neg)

    # ------------------------------------------------------------------
    # Gradient / Hessian for logistic loss
    # ------------------------------------------------------------------

    def _grad(self, raw_preds, labels):
        """
        First-order gradient of the logistic loss: sigmoid(raw) - y.

        Parameters
        ----------
        raw_preds : np.ndarray
        labels : np.ndarray

        Returns
        -------
        np.ndarray
        """
        return self.sigmoid(raw_preds) - labels

    def _hess(self, raw_preds):
        """
        Second-order gradient (Hessian) of the logistic loss: p * (1 - p).

        Parameters
        ----------
        raw_preds : np.ndarray

        Returns
        -------
        np.ndarray
        """
        p = self.sigmoid(raw_preds)
        return p * (1.0 - p)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        min_child_weight=1,
        max_depth=5,
        min_leaf=5,
        learning_rate=0.4,
        boosting_rounds=5,
        lambda_=1.5,
        gamma=1,
        solver="greedy",
        eps=0.1,
    ):
        """
        Fit the classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Binary labels (0 or 1).
        min_child_weight : float
            Minimum Hessian sum per child node (default=1).
        max_depth : int
            Maximum tree depth (default=5).
        min_leaf : int
            Minimum samples per leaf (default=5).
        learning_rate : float
            Shrinkage factor per tree (default=0.4).
        boosting_rounds : int
            Number of boosting iterations (default=5).
        lambda_ : float
            L2 regularization on leaf weights (default=1.5).
        gamma : float
            Minimum gain to allow a split (default=1).
        solver : str
            Split strategy: 'greedy', 'local', or 'global' (default='greedy').
        eps : float
            Quantile sketch approximation factor (default=0.1).

        Returns
        -------
        self
        """
        self.X = X
        self.y = y
        self.learning_rate = learning_rate

        raw = np.full(X.shape[0], self.log_odds(y), dtype=np.float64)

        for _ in range(boosting_rounds):
            grad = self._grad(raw, y)
            hess = self._hess(raw)
            tree = XGBoostTree().fit(
                X,
                grad,
                hess,
                max_depth=max_depth,
                min_leaf=min_leaf,
                lambda_=lambda_,
                gamma=gamma,
                solver=solver,
                eps=eps,
                min_child_weight=min_child_weight,
            )
            raw += learning_rate * tree.predict(X)
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for class 1.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            P(y=1) for each sample.
        """
        raw = np.full(X.shape[0], self.log_odds(self.y), dtype=np.float64)
        for tree in self.trees:
            raw += self.learning_rate * tree.predict(X)
        return self.sigmoid(raw)

    def predict(self, X):
        """
        Return binary class predictions (threshold at 0.5).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Binary predictions (0 or 1).
        """
        return np.where(self.predict_proba(X) > 0.5, 1, 0)
