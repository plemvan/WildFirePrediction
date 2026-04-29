"""
src.models — XGBoost from-scratch model package.

Public API
----------
from src.models import XGBoostClassifier
from src.models import XGBoostTree
from src.models import Node
"""

from src.models.classifier import XGBoostClassifier
from src.models.node import Node
from src.models.tree import XGBoostTree

__all__ = ["Node", "XGBoostTree", "XGBoostClassifier"]
