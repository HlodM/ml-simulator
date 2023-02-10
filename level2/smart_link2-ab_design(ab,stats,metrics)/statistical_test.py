from typing import List, Tuple
import numpy as np
from scipy import stats


def cpc_sample(
    n_samples: int, conversion_rate: float, reward_avg: float, reward_std: float
) -> np.ndarray:
    """Sample data."""
    cvr = np.random.binomial(1, conversion_rate, n_samples)
    cpa = np.random.normal(reward_avg, reward_std, n_samples)
    cpc = cvr * cpa

    return cpc


def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05) -> Tuple[bool, float]:
    """Perform t-test.

    Parameters
    ----------
    cpc_a: np.ndarray :
        first samples
    cpc_b: np.ndarray :
        second samples
    alpha :
         (Default value = 0.05)

    Returns
    -------
    Tuple[bool, float] :
        True if difference is significant, False otherwise
        p-value
    """
    p_value = float(stats.ttest_ind(cpc_a, cpc_b).pvalue)

    return p_value <= alpha, p_value


def aa_test(
        n_simulations: int,
        n_samples: int,
        cvr: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
) -> float:
    """Do the A/A test (simulation)."""

    type_1_errors = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Generate two cpc samples with the same cvr, reward_avg, and reward_std
        # Check t-test and save type 1 error
        cpc1 = cpc_sample(n_samples, cvr, reward_avg, reward_std)
        cpc2 = cpc_sample(n_samples, cvr, reward_avg, reward_std)
        type_1_errors[i] = t_test(cpc1, cpc2, alpha)[0]

    # Calculate the type 1 errors rate
    type_1_errors_rate = type_1_errors.mean()

    return type_1_errors_rate


def ab_test(
    n_simulations: int,
    n_samples: int,
    cvr: float,
    mde: float,
    reward_avg: float,
    reward_std: float,
    alpha: float = 0.05,
) -> float:
    """Do the A/B test (simulation)."""

    type_2_errors = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Generate one cpc sample with the given conversion_rate, reward_avg, and reward_std
        # Generate another cpc sample with the given conversion_rate * mde, reward_avg, and reward_std
        # Check t-test and save type 2 error
        cpc1 = cpc_sample(n_samples, cvr, reward_avg, reward_std)
        cpc2 = cpc_sample(n_samples, cvr * (1 + mde), reward_avg, reward_std)
        type_2_errors[i] = not t_test(cpc1, cpc2, alpha)[0]

    # Calculate the type 2 errors rate
    type_2_errors_rate = type_2_errors.mean()

    return type_2_errors_rate


def select_sample_size(
        n_samples_grid: List[int],
        n_simulations: int,
        cvr: float,
        mde: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[int, float, float]:
    """Select sample size."""
    for n_samples in n_samples_grid:
        # Implement your solution here
        type_1_error = aa_test(n_simulations, n_samples, cvr, reward_avg, reward_std, alpha)
        type_2_error = ab_test(n_simulations, n_samples, cvr, mde, reward_avg, reward_std, alpha)
        if type_1_error <= alpha and type_2_error <= beta:
            return n_samples, type_1_error, type_2_error

    raise RuntimeError(
        "Can't find sample size. "
        f"Last sample size: {n_samples}, "
        f"last type 1 error: {type_1_error}, "
        f"last type 2 error: {type_2_error}"
        "Make sure that the grid is big enough."
    )


def select_mde(
        n_samples: int,
        n_simulations: int,
        cvr: float,
        mde_grid: List[float],
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[float, float]:
    """Select MDE."""
    for mde in mde_grid:
        # Implement your solution here
        type_1_error = aa_test(n_simulations, n_samples, cvr, reward_avg, reward_std, alpha)
        type_2_error = ab_test(n_simulations, n_samples, cvr, mde, reward_avg, reward_std, alpha)
        if type_1_error <= alpha and type_2_error <= beta:
            return mde, type_2_error

    raise RuntimeError(
        "Can't find MDE. "
        f"Last MDE: {mde}, "
        f"last type 2 error: {type_2_error}. "
        "Make sure that the grid is big enough."
    )
