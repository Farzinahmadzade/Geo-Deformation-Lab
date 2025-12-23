# analysis/tikhonov_1d.py
"""
Simple 1‑D Tikhonov smoothing (second‑derivative regularization).

min_x ||x - y||^2 + alpha ||L x||^2
where L is the discrete 2nd‑difference operator.[web:90][web:11]
"""

import numpy as np


def second_diff_matrix(n: int) -> np.ndarray:
    """
    Build (n-2) x n finite‑difference matrix for the 2nd derivative.

    L x ~= x[i-1] - 2 x[i] + x[i+1]  for interior points.[web:11][web:97]
    """
    if n < 3:
        raise ValueError("Need at least 3 points for 2nd‑order differences.")

    L = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def tikhonov_1d(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply 1‑D Tikhonov regularization with a 2nd‑derivative penalty.

    Parameters
    ----------
    y : 1‑D array
        Noisy input time series (length N).[web:90]
    alpha : float
        Regularization parameter (>=0). Larger -> smoother result.

    Returns
    -------
    x : 1‑D array
        Smoothed time series of length N.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    if alpha < 0:
        raise ValueError("alpha must be non‑negative.")

    # Identity and curvature operator
    I = np.eye(n, dtype=float)
    L = second_diff_matrix(n)

    # Solve (I + alpha * L^T L) x = y
    A = I + alpha * (L.T @ L)
    x = np.linalg.solve(A, y)  # symmetric positive‑definite system[web:8][web:97]
    return x