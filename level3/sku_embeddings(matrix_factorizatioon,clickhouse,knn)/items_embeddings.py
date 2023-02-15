import numpy as np
from scipy.sparse import csr_matrix
import implicit


def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using factorization model.
    The order of items should be the same in the output matrix.

    Args:
        ui_matrix (pd.DataFrame): User-Item matrix of size (N, M)
        dim (int): Dimension of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    model = implicit.als.AlternatingLeastSquares(factors=dim, iterations=8)
    model.fit(ui_matrix)
    items_vec = model.item_factors
    return items_vec
