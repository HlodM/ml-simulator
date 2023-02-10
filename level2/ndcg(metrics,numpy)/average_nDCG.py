from typing import List
import numpy as np


def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = "standard") -> float:
    """average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """

    score = np.mean(np.apply_along_axis(
        lambda relevance: normalized_dcg(relevance, k, method), 1, list_relevances)
    )

    return score


def dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain"""

    if method == "standard":
        dcg_score = np.sum(np.array(relevance[:k]) / np.log2(np.arange(2, k + 2)))
    elif method == "industry":
        dcg_score = np.sum((2 ** np.array(relevance[:k]) - 1) / np.log2(np.arange(2, k + 2)))
    else:
        raise ValueError

    return dcg_score


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain."""

    ndcg_score = dcg(relevance, k, method) / dcg(np.sort(relevance)[::-1], k, method)

    return ndcg_score
