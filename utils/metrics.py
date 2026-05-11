"""
Core metric functions for evaluating synthetic expression data.

Functions:
    mmd(X, Y, kernel='rbf', gamma=None)   : Maximum Mean Discrepancy² with RBF or linear kernel.
    mmd_rbf(X, Y, gamma=None)             : Backward‑compatible alias for MMD with RBF kernel.
    mmd_linear(X, Y)                      : MMD² with linear kernel (squared mean difference).
    frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) : Gaussian Fréchet distance.
"""

import numpy as np
from scipy.linalg import sqrtm


def mmd(X, Y, kernel='rbf', gamma=None):
    """
    Unbiased estimate of the squared Maximum Mean Discrepancy.

    Supports two kernel types:
      - 'linear' : K(x,y) = x·y.  Returns ||mean(X) - mean(Y)||² (exact, no gamma).
      - 'rbf'    : K(x,y) = exp(-gamma * ||x-y||²). Median heuristic if gamma is None.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features)
    kernel : str, default 'rbf'
        Kernel type: 'linear' or 'rbf'.
    gamma : float or None, default None
        RBF kernel coefficient. Ignored for linear kernel.

    Returns
    -------
    mmd2 : float
        Non‑negative MMD² value (guaranteed >= 0).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if kernel == 'linear':
        # Linear kernel MMD² = ||μ_X - μ_Y||²
        diff = X.mean(axis=0) - Y.mean(axis=0)
        return np.dot(diff, diff)

    elif kernel == 'rbf':
        return _mmd_rbf(X, Y, gamma)

    else:
        raise ValueError(f"Unsupported kernel: {kernel}. Use 'linear' or 'rbf'.")


def _mmd_rbf(X, Y, gamma):
    """Internal RBF MMD² implementation (unbiased V‑statistic)."""
    n, m = X.shape[0], Y.shape[0]

    # Gram matrices (linear part)
    XX = X @ X.T
    XY = X @ Y.T
    YY = Y @ Y.T

    # Squared Euclidean distances
    X_diag = np.diag(XX)
    Y_diag = np.diag(YY)
    dist_XX = X_diag[:, None] + X_diag[None, :] - 2 * XX
    dist_XY = X_diag[:, None] + Y_diag[None, :] - 2 * XY
    dist_YY = Y_diag[:, None] + Y_diag[None, :] - 2 * YY

    # Median heuristic for gamma
    if gamma is None:
        triu_idx = np.triu_indices_from(dist_XX, k=1)
        all_dists = np.concatenate([dist_XX[triu_idx], dist_XY.ravel(), dist_YY[triu_idx]])
        median_dist = np.median(all_dists)
        gamma = 1.0 / (2.0 * median_dist) if median_dist > 0 else 1.0

    # RBF kernel matrices
    K_XX = np.exp(-gamma * dist_XX)
    K_XY = np.exp(-gamma * dist_XY)
    K_YY = np.exp(-gamma * dist_YY)

    mmd2 = ((K_XX.sum() - n) / (n * (n - 1)) +
            (K_YY.sum() - m) / (m * (m - 1)) -
            2.0 * K_XY.mean())
    return max(0.0, mmd2)


def mmd_rbf(X, Y, gamma=None):
    """Backward‑compatible alias for mmd(..., kernel='rbf')."""
    return mmd(X, Y, kernel='rbf', gamma=gamma)


def mmd_linear(X, Y):
    """Convenience function for linear MMD²."""
    return mmd(X, Y, kernel='linear')


# ----------------------------------------------------------------------
# Fréchet distance (unchanged)
# ----------------------------------------------------------------------
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Fréchet distance between two multivariate Gaussian distributions.

    Computes:
        ||mu1 - mu2||² + Tr(Σ1 + Σ2 - 2*(Σ1 Σ2)^{1/2})

    Parameters
    ----------
    mu1, mu2 : array-like of shape (n_features,)
    sigma1, sigma2 : array-like of shape (n_features, n_features)
    eps : float, default 1e-6
        Small diagonal regularisation for numerical stability.

    Returns
    -------
    distance : float
    """
    diff = np.asarray(mu1) - np.asarray(mu2)
    sigma1 = np.asarray(sigma1) + eps * np.eye(sigma1.shape[0])
    sigma2 = np.asarray(sigma2) + eps * np.eye(sigma2.shape[0])

    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
