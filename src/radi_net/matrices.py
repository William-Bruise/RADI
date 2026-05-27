import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh


def make_1d_dirichlet_laplacian(m: int):
    if m <= 0:
        raise ValueError("m must be positive")
    main = 2.0 * np.ones(m)
    off = -1.0 * np.ones(m - 1)
    T = diags([off, main, off], offsets=[-1, 0, 1], format="csc")
    return T * (m + 1) ** 2


def make_2d_dirichlet_laplacian(m: int):
    T = make_1d_dirichlet_laplacian(m)
    I = eye(m, format="csc")
    return kron(I, T, format="csc") + kron(T, I, format="csc")


def make_diagonal_spd(lambdas):
    lambdas = np.asarray(lambdas, dtype=float)
    if np.any(lambdas <= 0):
        raise ValueError("all lambdas must be positive")
    return diags(lambdas, 0, format="csc")


def estimate_spectral_interval(S, known_lambdas=None):
    if known_lambdas is not None:
        vals = np.asarray(known_lambdas, dtype=float)
        return float(vals.min()), float(vals.max())
    lmin = eigsh(csc_matrix(S), k=1, which="SA", return_eigenvectors=False)[0]
    lmax = eigsh(csc_matrix(S), k=1, which="LA", return_eigenvectors=False)[0]
    return float(lmin), float(lmax)
