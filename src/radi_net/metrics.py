import numpy as np


def relative_residual_from_W(B, W):
    num = np.linalg.norm(W.T @ W, ord="fro")
    den = np.linalg.norm(B.T @ B, ord="fro") + 1e-15
    return float(num / den)


def rational_residual_values(lambdas, alphas):
    lambdas = np.asarray(lambdas, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    logabs = np.zeros_like(lambdas)
    sign = np.ones_like(lambdas)
    for a in alphas:
        frac = (lambdas - a) / (lambdas + a)
        sign *= np.sign(frac)
        logabs += np.log(np.abs(frac) + 1e-300)
    return sign * np.exp(logabs)


def weighted_rational_loss(lambdas, weights, alphas, gamma=0.0, reg=0.0):
    r = rational_residual_values(lambdas, alphas)
    vals = r**2
    theta = np.log(np.asarray(alphas))
    reg_term = reg * np.mean((theta - theta.mean()) ** 2) if len(theta) else 0.0
    return float(np.dot(weights, vals) + gamma * np.max(vals) + reg_term)
