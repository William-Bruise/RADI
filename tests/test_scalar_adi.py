import numpy as np
from radi_net.adi import lr_adi_spd
from radi_net.metrics import relative_residual_from_W


def test_scalar_adi_formula():
    lam = 3.0
    b = 2.0
    alphas = np.array([0.5, 1.5, 2.5])
    S = np.array([[lam]])
    B = np.array([[b]])
    out = lr_adi_spd(S, B, alphas)
    r = np.prod((lam - alphas) / (lam + alphas))
    assert np.allclose(out["W"], [[b * r]], atol=1e-12)
    assert np.isclose(relative_residual_from_W(B, out["W"]), abs(r) ** 2, atol=1e-12)
