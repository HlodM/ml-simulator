"""Solution's template for user."""
from sklearn.neighbors import NearestNeighbors
import numpy as np


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness    

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    nn = NearestNeighbors(n_neighbors=num_neighbors).fit(embeddings)
    uniqueness = nn.kneighbors()[0].mean(axis=1)

    return uniqueness
