import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Node:
    """
    Parameters
    ----------
    n_items : int :
        number of items in leaf node
    ATE : float :
        average treatment effect as DeltaDeltaP
    split_feat: int :
        feature index for split
    split_threshold : float :
        threshold for split
    left : List["Node"] :
        left leaf Node after split
    right : List["Node"] :
        right leaf Node after split
   """
    n_items: int
    ATE: float
    split_feat: int = None
    split_threshold: float = 0
    left: List["Node"] = None
    right: List["Node"] = None


@dataclass
class UpliftTreeRegressor:
    """
    Parameters
    ----------
    max_depth : int :
        maximum depth of tree
    min_samples_leaf : int :
        minimum count of samples in leaf
    min_samples_leaf_treated : int :
        minimum count of treated samples in leaf
    min_samples_leaf_control : int :
        minimum count of control samples in leaf
    """

    max_depth: int = 3
    min_samples_leaf: int = 1000
    min_samples_leaf_treated: int = 300
    min_samples_leaf_control: int = 300

    @staticmethod
    def threshold_options(column_values: np.ndarray) -> np.ndarray:
        """
        column_values - one-dimensional array of feature.

        thresholds - the resulting threshold options.
        """

        unique_values = np.unique(column_values)

        if len(unique_values) > 10:
            percentiles = np.percentile(
                column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]
            )
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])

        thresholds = np.unique(percentiles)

        return thresholds

    def _leaf_condition(self, left_mask: np.ndarray, treatment: np.ndarray) -> bool:
        """Check if the split satisfies the condition of the minimum number of objects."""

        right_mask = ~left_mask
        min_leaf_cond = min(left_mask.sum(), right_mask.sum()) >= self.min_samples_leaf
        min_treated_cond = min(
            treatment[left_mask].sum(),
            treatment[right_mask].sum()
        ) >= self.min_samples_leaf_treated
        min_control_cond = min(
            (treatment[left_mask] == 0).sum(),
            (treatment[right_mask] == 0).sum()
        ) >= self.min_samples_leaf_control

        return min_leaf_cond and min_treated_cond and min_control_cond

    def _split(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, ate) -> Tuple[int, float]:
        """Return the feature index and the threshold for split that maximizes ATE."""

        split_feat = None
        split_thr = None

        for i in range(self.n_features):
            column_values = X[:, i]
            thresholds = self.threshold_options(column_values)
            for threshold in thresholds:
                left_mask = column_values <= threshold
                if self._leaf_condition(left_mask, treatment):
                    ate_left = y[left_mask & (treatment == 1)].mean() - y[left_mask & (treatment == 0)].mean()
                    ate_right = y[~left_mask & (treatment == 1)].mean() - y[~left_mask & (treatment == 0)].mean()
                    cur_ate = abs(ate_left - ate_right)
                    if cur_ate > ate:
                        ate = cur_ate
                        split_feat = i
                        split_thr = threshold
                else:
                    continue

        return split_feat, split_thr

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model.

        Parameters
        ----------
        X : np.ndarray :
            users features
        treatment : np.ndarray :
            ndarray of treatment flags
        y : np.ndarray :
            ndarray of targets
        """

        self.n_features = X.shape[1]
        self.tree_ = self._build_tree(X, treatment, y)

    def _build_tree(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Build a decision tree by recursively finding the best split."""

        ate = y[treatment == 1].mean() - y[treatment == 0].mean()

        node = Node(
            n_items=len(y),
            ATE=ate,
        )

        if depth < self.max_depth:
            split_feat, split_thr = self._split(X, treatment, y, ate)
            if split_feat is not None:
                left_mask = X[:, split_feat] <= split_thr
                node.split_feat = split_feat
                node.split_threshold = split_thr
                node.left = self._build_tree(X[left_mask], treatment[left_mask], y[left_mask], depth + 1)
                node.right = self._build_tree(X[~left_mask], treatment[~left_mask], y[~left_mask], depth + 1)

        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicts for X."""
        predictions = np.apply_along_axis(self._predict, 1, X)

        return predictions

    def _predict(self, inputs) -> float:
        """Predict ATE for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.split_feat] <= node.split_threshold:
                node = node.left
            else:
                node = node.right

        return node.ATE
