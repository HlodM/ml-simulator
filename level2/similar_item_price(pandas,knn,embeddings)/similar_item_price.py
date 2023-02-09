"""Solution for Similar Items task"""
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """

        combs = combinations(embeddings.keys(), 2)
        pair_sims = {
            pair: round(1 - cosine(embeddings[pair[0]], embeddings[pair[1]]), 8) for pair in combs
        }

        return pair_sims

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """

        keys, values = list(sim.keys()), list(sim.values())
        sorted_value_idx = np.argsort(values)
        sorted_sim = {keys[i]: values[i] for i in sorted_value_idx[::-1]}

        knn_dict = dict()

        for (i, j), value in sorted_sim.items():
            if len(knn_dict.setdefault(i, [])) < top:
                knn_dict[i] += [(j, value)]
            if len(knn_dict.setdefault(j, [])) < top:
                knn_dict[j] += [(i, value)]

        return knn_dict

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """

        knn_price_dict = dict.fromkeys(prices.keys(), 0)

        for key, value in knn_dict.items():

            tot_sim = 0

            for j, sim in value:
                tot_sim += sim + 1
                knn_price_dict[key] += prices[j] * (sim + 1)

            knn_price_dict[key] = round(knn_price_dict[key] / tot_sim, 2)

        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """

        pair_sims = SimilarItems.similarity(embeddings)
        knn_dict = SimilarItems.knn(pair_sims, top)
        knn_price_dict = SimilarItems.knn_price(knn_dict, prices)

        return knn_price_dict
