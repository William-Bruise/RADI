import numpy as np


def _normalize_cols(B):
    nrm = np.linalg.norm(B, axis=0) + 1e-15
    return B / nrm


def rhs_random(n, r, seed=0):
    rng = np.random.default_rng(seed)
    return _normalize_cols(rng.standard_normal((n, r)))


def rhs_localized_2d(m, points, sigma=1.5):
    x = np.arange(m)
    X, Y = np.meshgrid(x, x, indexing="ij")
    cols = []
    for px, py in points:
        g = np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * sigma**2))
        cols.append(g.reshape(-1))
    return _normalize_cols(np.column_stack(cols))


def rhs_low_frequency_2d(m, r):
    x = np.linspace(0, 1, m)
    X, Y = np.meshgrid(x, x, indexing="ij")
    cols = []
    for k in range(1, r + 1):
        cols.append((np.sin(np.pi * k * X) * np.sin(np.pi * k * Y)).reshape(-1))
    return _normalize_cols(np.column_stack(cols))


def rhs_high_frequency_2d(m, r):
    x = np.linspace(0, 1, m)
    X, Y = np.meshgrid(x, x, indexing="ij")
    cols = []
    base = max(1, m // 3)
    for k in range(r):
        f = base + k
        cols.append((np.sin(np.pi * f * X) * np.sin(np.pi * f * Y)).reshape(-1))
    return _normalize_cols(np.column_stack(cols))


def rhs_diagonal_weighted(n, mode, r, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, r))
    idx = np.arange(n) / max(1, n - 1)
    if mode == "low":
        w = np.exp(-6 * idx)
    elif mode == "high":
        w = np.exp(-6 * (1 - idx))
    elif mode == "bimodal":
        w = np.exp(-80 * (idx - 0.2) ** 2) + np.exp(-80 * (idx - 0.8) ** 2)
    elif mode == "random":
        w = 0.1 + rng.random(n)
    else:
        raise ValueError(f"unknown mode {mode}")
    return _normalize_cols(B * w[:, None])
