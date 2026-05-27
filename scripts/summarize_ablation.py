import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x, ddof=0))


def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x))


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted(glob.glob(str(root / "results" / "ablation_n*_seed*.csv")))
    if not files:
        raise SystemExit("No ablation files found. Run scripts/run_ablation.py first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    key_cols = ["experiment", "n", "matrix_type", "rhs_mode", "rank_B", "K", "method", "gamma", "seed"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    after = len(df)

    df["family"] = np.where(df["method"].str.startswith("RADI-Net"), "network", "classical")

    out_dir = root / "results"
    df.to_csv(out_dir / "ablation_all_dedup.csv", index=False)

    # Main mean±std table per n/K/rhs/method
    grouped = df.groupby(["n", "rhs_mode", "K", "method", "gamma"], as_index=False)
    summary = grouped.agg(
        seeds=("seed", "nunique"),
        final_relres_mean=("final_relres", _safe_mean),
        final_relres_std=("final_relres", _safe_std),
        time_mean=("time_seconds", _safe_mean),
        time_std=("time_seconds", _safe_std),
        steps_1e4_mean=("steps_to_1e-4", _safe_mean),
        steps_1e6_mean=("steps_to_1e-6", _safe_mean),
        steps_1e8_mean=("steps_to_1e-8", _safe_mean),
    ).sort_values(["n", "rhs_mode", "K", "final_relres_mean"])
    summary.to_csv(out_dir / "ablation_mean_std_table.csv", index=False)

    # Stability table: best/worst across seeds
    stab = grouped.agg(
        final_relres_best=("final_relres", "min"),
        final_relres_worst=("final_relres", "max"),
        final_relres_std=("final_relres", _safe_std),
        time_best=("time_seconds", "min"),
        time_worst=("time_seconds", "max"),
        time_std=("time_seconds", _safe_std),
    ).sort_values(["n", "rhs_mode", "K", "final_relres_best"])
    stab.to_csv(out_dir / "ablation_stability_table.csv", index=False)

    # Simple comparison: best classical vs best network using mean performance
    rows = []
    for (n, rhs_mode, K), g in summary.groupby(["n", "rhs_mode", "K"]):
        gc = g[~g["method"].str.startswith("RADI-Net")].sort_values("final_relres_mean")
        gn = g[g["method"].str.startswith("RADI-Net")].sort_values("final_relres_mean")
        if len(gc) == 0 or len(gn) == 0:
            continue
        c = gc.iloc[0]
        u = gn.iloc[0]
        rows.append({
            "n": n,
            "rhs_mode": rhs_mode,
            "K": K,
            "best_classical": c["method"],
            "best_classical_mean_relres": c["final_relres_mean"],
            "best_classical_mean_time": c["time_mean"],
            "best_network": u["method"],
            "best_network_gamma": u["gamma"],
            "best_network_mean_relres": u["final_relres_mean"],
            "best_network_mean_time": u["time_mean"],
            "relres_gain_classical_over_network": c["final_relres_mean"] / max(u["final_relres_mean"], 1e-300),
            "time_ratio_classical_over_network": c["time_mean"] / max(u["time_mean"], 1e-300),
        })
    pd.DataFrame(rows).sort_values(["n", "rhs_mode", "K"]).to_csv(out_dir / "ablation_classical_vs_network.csv", index=False)

    report = {
        "input_files": files,
        "rows_before_dedup": int(before),
        "rows_after_dedup": int(after),
        "outputs": [
            "results/ablation_all_dedup.csv",
            "results/ablation_mean_std_table.csv",
            "results/ablation_stability_table.csv",
            "results/ablation_classical_vs_network.csv",
        ],
    }
    (out_dir / "ablation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
