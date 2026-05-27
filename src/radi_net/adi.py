import time
import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu

from .metrics import relative_residual_from_W


def lr_adi_spd(S, B, alphas, verbose=False, return_history=True):
    t0 = time.time()
    S = csc_matrix(S)
    n = S.shape[0]
    W = np.array(B, dtype=float, copy=True)
    Z_blocks = []
    hist = []
    success = True
    I = eye(n, format="csc")
    for a in alphas:
        try:
            lu = splu((S + float(a) * I).tocsc())
            V = lu.solve(W)
        except Exception:
            success = False
            break
        Z_blocks.append(np.sqrt(2.0 * a) * V)
        W = W - 2.0 * a * V
        if return_history:
            hist.append(relative_residual_from_W(B, W))
        if verbose:
            print(f"alpha={a:.3e}, relres={hist[-1]:.3e}")
    Z = np.hstack(Z_blocks) if Z_blocks else np.zeros((n, 0))
    return {
        "Z": Z,
        "W": W,
        "relres_history": hist,
        "alphas": np.asarray(alphas, dtype=float),
        "rank": Z.shape[1],
        "time_seconds": time.time() - t0,
        "success": success,
    }


def dense_residual_check(S, B, Z):
    X = Z @ Z.T
    R = S @ X + X @ S - B @ B.T
    return np.linalg.norm(R, ord="fro") / (np.linalg.norm(B @ B.T, ord="fro") + 1e-15)
