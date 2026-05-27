# RADI-Net

RADI-Net is a reproducible Python codebase for low-rank Lyapunov matrix equations
\( S X + X S = B B^T \) with SPD \(S\), comparing classical LR-ADI against a finite-depth unrolled LR-ADI network.

## Core idea
- **Classical LR-ADI:** iterate with fixed positive shifts \(\alpha_j\):
  \( (S+\alpha_j I)V_j = W_{j-1},\; W_j = W_{j-1}-2\alpha_j V_j \), append \(\sqrt{2\alpha_j}V_j\) to low-rank factor \(Z\).
- **Residual identity:** \(S X_K + X_K S - B B^T = -W_KW_K^T\), with \(X_K=Z_KZ_K^T\).
- **RADI-Net:** unfold K LR-ADI steps as K network layers, with trainable \(\theta_j=\log(\alpha_j)\).

## Training
- **Spectral surrogate (default):** optimize weighted rational loss over eigenvalues and RHS-induced weights.
- **Direct unrolled (small only):** optimize final relative residual by explicitly running LR-ADI layers.

## Fair baselines
All methods solve the same equation and differ only in shift choices:
- LR-ADI-logspace
- LR-ADI-geomean
- LR-ADI-random
- LR-ADI-greedy-weighted
- RADI-Net-spectral
- RADI-Net-direct (small sanity)

## Storyline demo
The storyline demo uses one matrix and different RHS patterns (`low`, `high`, `bimodal`) to show:
1. logspace LR-ADI keeps the same shifts across RHS,
2. RADI-Net learns RHS-dependent shifts,
3. RADI-Net often reduces residual faster.

## Installation
```bash
python -m pip install -e .
pytest
```

## Small run
```bash
python scripts/run_smoke.py
```

## Scope
Laptop-friendly defaults. Large mode is optional and intended for bigger runs (up to roughly n≈2048 diagonal cases).
