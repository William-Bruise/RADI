import numpy as np
from radi_net.adi import dense_residual_check, lr_adi_spd
from radi_net.matrices import make_diagonal_spd
from radi_net.metrics import relative_residual_from_W


def test_residual_identity():
    n = 20
    rng = np.random.default_rng(0)
    lambdas = np.linspace(1, 20, n)
    S = make_diagonal_spd(lambdas)
    B = rng.standard_normal((n, 2))
    out = lr_adi_spd(S, B, np.array([1.0, 2.0, 4.0]))
    cheap = relative_residual_from_W(B, out["W"])
    dense = dense_residual_check(S.toarray(), B, out["Z"])
    assert abs(cheap - dense) < 1e-8
