from numba import njit
import numpy as np


@njit
def calc_weights(L, alpha, beta, kappa):
    lambda_ = (alpha ** 2) * (L + kappa) - L
    n_weights = 2 * L + 1
    wm = np.full(int(n_weights), 1 / (2 * (L + lambda_)))
    wc = np.full(int(n_weights), 1 / (2 * (L + lambda_)))
    wm[0] = lambda_ / (L + lambda_)
    wc[0] = wm[0] + (1 - alpha ** 2 + beta)
    return wm, wc, lambda_


@njit
def generate_sp(X_mean, P, lambda_):
    """
    Generate sigma points for the update step using covariance matrix P and mean X_mean.

    P: Covariance matrix (assumed to be already Cholesky-decomposed or sqrt of covariance matrix)
    L: Dimension of the state vector (here assumed to be 1, scalar case)
    lambda_: Scaling parameter for sigma points
    X_mean: Predicted state mean (float scalar)
    """
    L = X_mean.size

    # Compute the custom Cholesky decomposition of the augmented covariance matrix
    P_sqrt = custom_cholesky(P)

    # Compute phi as sqrt(L + lambda_)
    phi = np.sqrt(L + lambda_)

    # Initialize the sigma points array
    X2 = np.zeros((2 * X_mean.size + 1, X_mean.size))

    # First sigma point is the mean (aug_x)
    X2[0] = X_mean

    # Generate the rest of the sigma points
    for i in range(X_mean.size):
        X2[i + 1] = X_mean + phi * P_sqrt[:, i]
        X2[i + X_mean.size + 1] = X_mean - phi * P_sqrt[:, i]

    return X2


@njit
def custom_cholesky(A, epsilon=1e-8):
    """
    Custom Cholesky decomposition that works with Numba.
    Adds small regularization (epsilon) to the diagonal elements if decomposition fails.
    Handles only 2D positive-definite matrices.
    """
    # Ensure A is a 2D array (matrix case)
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_val = 0.0
            for k in range(j):
                sum_val += L[i, k] * L[j, k]
            if i == j:
                diff = A[i, i] - sum_val
                # Regularization: If diff is non-positive, add epsilon to the diagonal
                if diff <= 0:
                    diff += epsilon
                L[i, j] = np.sqrt(diff)
            else:
                L[i, j] = (A[i, j] - sum_val) / L[j, j]

    return L


@njit
def f_vectorized(x1, x2, a, l, q):
    """
    Vectorized transition function for state evolution, compatible with Numba.
    Applies mean reversion and clipping to the state variables.

    x1: array of previous state variables
    x2: array of noise or random process variables
    a: mean reversion speed
    l: long-term mean (or target value)
    """
    # Apply mean-reverting process element-wise
    return a * (x1 - l) + l + q * x2
