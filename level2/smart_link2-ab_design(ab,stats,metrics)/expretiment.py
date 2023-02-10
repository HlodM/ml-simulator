from typing import Tuple, List
import numpy as np


class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
            self,
            experiment_id: int,
            groups: Tuple[str] = ("A", "B"),
            group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights

        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        self.salt = str(self.experiment_id)

        # Define the group weights if they are not provided equaly distributed
        # Check input group weights. They must be non-negative and sum to 1.

        if self.group_weights:
            if sum(self.group_weights) != 1 and np.all(np.array(self.group_weights) >= 0):
                raise NotImplementedError("Group weights must be non-negative and sum to 1!")
        else:
            group_count = len(self.groups)
            self.group_weights = [1 / group_count for _ in range(group_count)]

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        # Assign the click to a group randomly based on the group weights
        # Return the group id and group name

        acc = max(map(lambda x: len(str(x)) - 2, self.group_weights))
        remainder = hash(str(click_id) + self.salt) % 10**acc / 10**acc
        group_id = np.searchsorted(np.cumsum(self.group_weights), remainder, side='right')

        return group_id, self.groups[group_id]
