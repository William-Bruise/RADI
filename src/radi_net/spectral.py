import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh


def spectral_data_diagonal(lambdas, B):
    lambdas = np.asarray(lambdas, dtype=float)
    weights = np.sum(np.asarray(B, dtype=float) ** 2, axis=1)
    weights = np.maximum(weights, 0)
    s = weights.sum()
    weights = weights / s if s > 0 else np.ones_like(weights) / len(weights)
    return lambdas, weights


def spectral_data_laplacian_2d(m, B):
    d = 2.0 * (m + 1) ** 2 * np.ones(m)
    e = -1.0 * (m + 1) ** 2 * np.ones(m - 1)
    lam1, Q = eigh_tridiagonal(d, e)
    r = B.shape[1]
    W2 = np.zeros((m, m))
    for j in range(r):
        G = B[:, j].reshape(m, m)
        C = Q.T @ G @ Q
        W2 += np.abs(C) ** 2
    lambdas2 = (lam1[:, None] + lam1[None, :]).reshape(-1)
    weights = W2.reshape(-1)
    weights = np.maximum(weights, 0)
    weights /= weights.sum() + 1e-15
    return lambdas2, weights


def spectral_data_eigsh(S, B, k=100):
    vals, vecs = eigsh(S, k=min(k, S.shape[0] - 2), which="SA")
    proj = vecs.T @ B
    weights = np.sum(proj**2, axis=1)
    weights /= weights.sum() + 1e-15
    return vals, weights
