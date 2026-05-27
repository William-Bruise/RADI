import numpy as np
from radi_net.matrices import make_diagonal_spd
from radi_net.radi_net import RADINet
from radi_net.rhs import rhs_diagonal_weighted
from radi_net.shifts import shifts_geomean
from radi_net.spectral import spectral_data_diagonal
from radi_net.adi import lr_adi_spd


def test_radi_net_spectral_improves():
    n, K = 50, 5
    lambdas = np.logspace(-2, 2, n)
    S = make_diagonal_spd(lambdas)
    B = rhs_diagonal_weighted(n, "low", 2, seed=0)
    lam, w = spectral_data_diagonal(lambdas, B)
    net = RADINet(K)
    net.fit_spectral(lam, w, seed=0)
    out = net.forward(S, B)
    base = lr_adi_spd(S, B, shifts_geomean(lambdas.min(), lambdas.max(), K))
    assert out["relres_history"][-1] <= base["relres_history"][-1]
