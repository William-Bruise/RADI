import numpy as np
from scipy.optimize import minimize

from .adi import lr_adi_spd
from .metrics import relative_residual_from_W
from .shifts import optimize_weighted_shifts


class RADINet:
    def __init__(self, K, init_alphas=None):
        self.K = int(K)
        if init_alphas is None:
            init_alphas = np.ones(self.K)
        self.theta = np.log(np.asarray(init_alphas, dtype=float))

    def alphas(self):
        return np.exp(self.theta)

    def forward(self, S, B):
        return lr_adi_spd(S, B, self.alphas(), return_history=True)

    def loss(self, S, B):
        out = self.forward(S, B)
        return relative_residual_from_W(B, out["W"])

    def fit_spectral(self, lambdas, weights, gamma=0.0, reg=1e-4, n_restarts=8, seed=0, maxiter=300):
        res = optimize_weighted_shifts(lambdas, weights, self.K, gamma=gamma, reg=reg, n_restarts=n_restarts, seed=seed, maxiter=maxiter)
        self.theta = np.log(res["alphas"])
        return res

    def fit_direct(self, S, B, lmin, lmax, maxiter=80, seed=0):
        rng = np.random.default_rng(seed)
        x0 = np.log(np.exp(np.linspace(np.log(lmin), np.log(lmax), self.K)) * np.exp(0.05 * rng.standard_normal(self.K)))
        bounds = [(np.log(lmin) - 4, np.log(lmax) + 4)] * self.K

        def obj(th):
            out = lr_adi_spd(S, B, np.exp(th), return_history=False)
            return relative_residual_from_W(B, out["W"])

        res = minimize(obj, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        self.theta = res.x
        return {"alphas": np.exp(res.x), "objective": float(res.fun), "success": bool(res.success), "message": str(res.message)}
