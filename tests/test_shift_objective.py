import numpy as np
from radi_net.metrics import weighted_rational_loss
from radi_net.shifts import optimize_weighted_shifts, shifts_logspace


def test_shift_optimizer_beats_logspace():
    lam = np.logspace(-2, 2, 100)
    w = np.exp(-((np.log10(lam) + 1.5) ** 2) / 0.2)
    w /= w.sum()
    K = 5
    fit = optimize_weighted_shifts(lam, w, K, seed=0)
    base = shifts_logspace(lam.min(), lam.max(), K)
    assert np.all(fit["alphas"] > 0)
    assert weighted_rational_loss(lam, w, fit["alphas"]) < weighted_rational_loss(lam, w, base)
