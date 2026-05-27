import json
import numpy as np
import pandas as pd

from .adi import lr_adi_spd
from .matrices import make_2d_dirichlet_laplacian, make_diagonal_spd, estimate_spectral_interval
from .metrics import weighted_rational_loss
from .radi_net import RADINet
from .rhs import rhs_diagonal_weighted, rhs_high_frequency_2d, rhs_localized_2d, rhs_low_frequency_2d
from .shifts import greedy_weighted_shifts, shifts_geomean, shifts_logspace, shifts_random_loguniform
from .spectral import spectral_data_diagonal, spectral_data_laplacian_2d


def _steps(hist, thr):
    for i, v in enumerate(hist, 1):
        if v <= thr:
            return i
    return np.nan


def _record(rows, exp, n, m, matrix_type, rhs_mode, rank_B, K, method, gamma, seed, out, shifts, obj=np.nan, success=True):
    h = out.get("relres_history", [])
    rows.append(
        {
            "experiment": exp,
            "n": n,
            "m": m,
            "matrix_type": matrix_type,
            "rhs_mode": rhs_mode,
            "rank_B": rank_B,
            "K": K,
            "method": method,
            "gamma": gamma,
            "seed": seed,
            "final_relres": h[-1] if h else np.nan,
            "steps_to_1e-4": _steps(h, 1e-4),
            "steps_to_1e-6": _steps(h, 1e-6),
            "steps_to_1e-8": _steps(h, 1e-8),
            "time_seconds": out.get("time_seconds", np.nan),
            "shifts_json": json.dumps(list(map(float, shifts))),
            "relres_history_json": json.dumps(list(map(float, h))),
            "objective_value": float(obj),
            "success": bool(success),
        }
    )


def run_diagonal_experiment(out_csv, large=False, seed=0):
    n_list = [256, 512, 1024, 2048] if large else [128, 256]
    K_list = [5, 10, 20] if large else [5, 10]
    rows = []
    for n in n_list:
        lambdas = np.logspace(-2, 2, n)
        S = make_diagonal_spd(lambdas)
        lmin, lmax = estimate_spectral_interval(S, known_lambdas=lambdas)
        for rhs_mode in ["low", "high", "bimodal"]:
            B = rhs_diagonal_weighted(n, rhs_mode, r=3, seed=seed)
            spec_lam, spec_w = spectral_data_diagonal(lambdas, B)
            for K in K_list:
                methods = {
                    "LR-ADI-logspace": shifts_logspace(lmin, lmax, K),
                    "LR-ADI-geomean": shifts_geomean(lmin, lmax, K),
                    "LR-ADI-random": shifts_random_loguniform(lmin, lmax, K, seed=seed),
                    "LR-ADI-greedy-weighted": greedy_weighted_shifts(spec_lam, spec_w, K),
                }
                for name, shifts in methods.items():
                    out = lr_adi_spd(S, B, shifts)
                    _record(
                        rows,
                        "diagonal",
                        n,
                        np.nan,
                        "diagonal",
                        rhs_mode,
                        B.shape[1],
                        K,
                        name,
                        0.0,
                        seed,
                        out,
                        shifts,
                        obj=weighted_rational_loss(spec_lam, spec_w, shifts),
                        success=out["success"],
                    )
                for gamma in [0.0, 1e-2]:
                    net = RADINet(K)
                    fit = net.fit_spectral(spec_lam, spec_w, gamma=gamma, seed=seed)
                    out = net.forward(S, B)
                    _record(rows, "diagonal", n, np.nan, "diagonal", rhs_mode, B.shape[1], K, "RADI-Net-spectral", gamma, seed, out, net.alphas(), fit["objective"], fit["success"])
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run_laplacian_experiment(out_csv, large=False, seed=0):
    m_list = [16, 32, 45] if large else [8, 16]
    K_list = [5, 10, 20, 30] if large else [5, 10]
    rows = []
    for m in m_list:
        S = make_2d_dirichlet_laplacian(m)
        lmin, lmax = estimate_spectral_interval(S)
        rhs_modes = {
            "localized_center": rhs_localized_2d(m, [(m // 2, m // 2), (m // 2 + 1, m // 2)]),
            "low_frequency": rhs_low_frequency_2d(m, 2),
            "high_frequency": rhs_high_frequency_2d(m, 2),
        }
        for rhs_mode, B in rhs_modes.items():
            spec_lam, spec_w = spectral_data_laplacian_2d(m, B)
            for K in K_list:
                shifts = shifts_logspace(lmin, lmax, K)
                _record(rows, "laplacian", m * m, m, "laplacian2d", rhs_mode, B.shape[1], K, "LR-ADI-logspace", 0.0, seed, lr_adi_spd(S, B, shifts), shifts)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run_direct_training_sanity(out_csv, seed=0):
    n, K = 64, 5
    lambdas = np.logspace(-2, 2, n)
    S = make_diagonal_spd(lambdas)
    B = rhs_diagonal_weighted(n, "bimodal", r=2, seed=seed)
    spec_lam, spec_w = spectral_data_diagonal(lambdas, B)
    rows = []
    net_s = RADINet(K)
    fit_s = net_s.fit_spectral(spec_lam, spec_w, seed=seed)
    out_s = net_s.forward(S, B)
    _record(rows, "direct_sanity", n, np.nan, "diagonal", "bimodal", 2, K, "RADI-Net-spectral", 0.0, seed, out_s, net_s.alphas(), fit_s["objective"], fit_s["success"])
    net_d = RADINet(K)
    fit_d = net_d.fit_direct(S, B, lambdas.min(), lambdas.max())
    out_d = net_d.forward(S, B)
    _record(rows, "direct_sanity", n, np.nan, "diagonal", "bimodal", 2, K, "RADI-Net-direct", 0.0, seed, out_d, net_d.alphas(), fit_d["objective"], fit_d["success"])
    shifts = shifts_logspace(lambdas.min(), lambdas.max(), K)
    out = lr_adi_spd(S, B, shifts)
    _record(rows, "direct_sanity", n, np.nan, "diagonal", "bimodal", 2, K, "LR-ADI-logspace", 0.0, seed, out, shifts)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run_storyline_demo(out_csv, small=True, seed=0):
    n, K = (128, 10) if small else (512, 20)
    lambdas = np.logspace(-2, 2, n)
    S = make_diagonal_spd(lambdas)
    rows = []
    log_shifts = shifts_logspace(lambdas.min(), lambdas.max(), K)
    for mode in ["low", "high", "bimodal"]:
        B = rhs_diagonal_weighted(n, mode, 3, seed)
        out0 = lr_adi_spd(S, B, log_shifts)
        _record(rows, "storyline", n, np.nan, "diagonal", mode, 3, K, "LR-ADI-logspace", 0.0, seed, out0, log_shifts)
        lam, w = spectral_data_diagonal(lambdas, B)
        net = RADINet(K)
        fit = net.fit_spectral(lam, w, seed=seed)
        out1 = net.forward(S, B)
        _record(rows, "storyline", n, np.nan, "diagonal", mode, 3, K, "RADI-Net-spectral", 0.0, seed, out1, net.alphas(), fit["objective"], fit["success"])
    pd.DataFrame(rows).to_csv(out_csv, index=False)
