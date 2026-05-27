import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .metrics import weighted_rational_loss


def shifts_logspace(lmin, lmax, K):
    return np.logspace(np.log10(lmin), np.log10(lmax), K)


def shifts_geomean(lmin, lmax, K):
    return np.full(K, np.sqrt(lmin * lmax))


def shifts_random_loguniform(lmin, lmax, K, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(lmin), np.log(lmax), size=K))


def optimize_weighted_shifts(lambdas, weights, K, gamma=0.0, reg=1e-4, n_restarts=8, seed=0, maxiter=300):
    lmin, lmax = float(np.min(lambdas)), float(np.max(lambdas))
    lo, hi = np.log(lmin) - 4.0, np.log(lmax) + 4.0
    bounds = [(lo, hi)] * K
    rng = np.random.default_rng(seed)

    inits = [np.log(shifts_logspace(lmin, lmax, K)), np.log(shifts_geomean(lmin, lmax, K)) + 0.1 * rng.standard_normal(K)]
    for i in range(max(0, n_restarts - len(inits))):
        inits.append(np.log(shifts_random_loguniform(lmin, lmax, K, seed + i + 1)))

    best = None
    for x0 in inits:
        f = lambda th: weighted_rational_loss(lambdas, weights, np.exp(th), gamma=gamma, reg=reg)
        res = minimize(f, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    alphas = np.sort(np.exp(best.x))
    return {"alphas": alphas, "objective": float(best.fun), "success": bool(best.success), "message": str(best.message)}


def greedy_weighted_shifts(lambdas, weights, K, gamma=0.0, grid_size=200, refine=True):
    lmin, lmax = float(np.min(lambdas)), float(np.max(lambdas))
    chosen = []
    grid = np.exp(np.linspace(np.log(lmin), np.log(lmax), grid_size))
    for _ in range(K):
        vals = [weighted_rational_loss(lambdas, weights, chosen + [a], gamma=gamma, reg=0.0) for a in grid]
        a0 = grid[int(np.argmin(vals))]
        if refine:
            res = minimize_scalar(
                lambda x: weighted_rational_loss(lambdas, weights, chosen + [np.exp(x)], gamma=gamma, reg=0.0),
                bounds=(np.log(lmin), np.log(lmax)),
                method="bounded",
            )
            a0 = np.exp(res.x) if res.success else a0
        chosen.append(float(a0))
    return np.asarray(chosen)
