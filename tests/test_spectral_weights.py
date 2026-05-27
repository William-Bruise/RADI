import numpy as np
from radi_net.rhs import rhs_low_frequency_2d
from radi_net.spectral import spectral_data_laplacian_2d


def test_spectral_weights_2d():
    m = 8
    B = rhs_low_frequency_2d(m, 2)
    lam, w = spectral_data_laplacian_2d(m, B)
    assert len(lam) == m * m
    assert len(w) == m * m
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0, atol=1e-10)
