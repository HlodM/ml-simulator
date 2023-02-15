from scipy.sparse import csr_matrix, diags
import numpy as np


class Normalization:
    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        """Normalization by column

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        diagonal = (1 / matrix.sum(axis=0)).flat
        norm_matrix = matrix @ diags(diagonal)
        return norm_matrix

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        diagonal = (1 / matrix.sum(axis=1)).flat
        norm_matrix = diags(diagonal) @ matrix
        return norm_matrix

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """Normalization using tf-idf

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        tf = Normalization.by_row(matrix)
        user_count = matrix.shape[0]
        idf = np.log(user_count / np.bincount(matrix.nonzero()[1]))
        norm_matrix = tf @ diags(idf)
        return norm_matrix

    @staticmethod
    def bm_25(
        matrix: csr_matrix, k1: float = 2.0, b: float = 0.75
    ) -> csr_matrix:
        """Normalization based on BM-25

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        tf = Normalization.by_row(matrix)

        rows_length = matrix.sum(axis=1)
        vector = k1 * (1 - b + b * rows_length / rows_length.mean())
        denominator = diags((1 / vector).flat) @ tf
        denominator.data += 1
        denominator = diags(vector.flat) @ denominator
        denominator.sort_indices()

        norm_matrix = Normalization.tf_idf(matrix)
        norm_matrix.data *= (k1 + 1) / denominator.data
        return norm_matrix
