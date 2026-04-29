"""
Backward-compatibility shim.

The implementation has been split into:
    src/models/node.py         — Node class
    src/models/tree.py         — XGBoostTree class
    src/models/classifier.py   — XGBoostClassifier class

Importing from this file still works so that existing notebook cells
(``from src.models.xgboost_from_scratch import XGBoostClassifier``)
do not break after the refactor.
"""

from src.models.classifier import XGBoostClassifier  # noqa: F401
from src.models.node import Node  # noqa: F401
from src.models.tree import XGBoostTree  # noqa: F401

__all__ = ["Node", "XGBoostTree", "XGBoostClassifier"]
