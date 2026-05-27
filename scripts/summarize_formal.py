import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_ratio(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a / b)


def _method_group(method: str) -> str:
    if method.startswith("RADI-Net"):
        return "unrolled_network"
    return "classical_lr_adi"


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted(glob.glob(str(root / "results" / "formal_diagonal_n*.csv")))
    if not files:
        raise SystemExit("No formal result files found: results/formal_diagonal_n*.csv")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    key_cols = ["experiment", "n", "matrix_type", "rhs_mode", "rank_B", "K", "method", "gamma", "seed"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    after = len(df)

    # Add simple readability columns
    df["family"] = df["method"].map(_method_group)
    df["iter_steps_used"] = df["K"]

    out_dir = root / "results"
    dedup_path = out_dir / "formal_all_dedup.csv"
    df.to_csv(dedup_path, index=False)

    # Simple method summary: users can directly read "steps / precision / time"
    simple_cols = [
        "n",
        "rhs_mode",
        "K",
        "family",
        "method",
        "gamma",
        "iter_steps_used",
        "final_relres",
        "steps_to_1e-4",
        "steps_to_1e-6",
        "steps_to_1e-8",
        "time_seconds",
    ]
    simple = (
        df[simple_cols]
        .sort_values(["n", "rhs_mode", "K", "family", "method", "gamma"], ascending=[True, True, True, True, True, True])
        .reset_index(drop=True)
    )
    simple_path = out_dir / "formal_simple_table.csv"
    simple.to_csv(simple_path, index=False)

    # Best classical vs best network for each (n, rhs_mode, K)
    rows = []
    for (n, rhs_mode, K), g in df.groupby(["n", "rhs_mode", "K"]):
        g_classical = g[g["family"] == "classical_lr_adi"].sort_values("final_relres")
        g_network = g[g["family"] == "unrolled_network"].sort_values("final_relres")
        if len(g_classical) == 0 or len(g_network) == 0:
            continue
        c = g_classical.iloc[0]
        u = g_network.iloc[0]
        rows.append(
            {
                "n": n,
                "rhs_mode": rhs_mode,
                "K": K,
                "classical_best_method": c["method"],
                "classical_best_relres": c["final_relres"],
                "classical_best_time": c["time_seconds"],
                "network_best_method": u["method"],
                "network_best_gamma": u["gamma"],
                "network_best_relres": u["final_relres"],
                "network_best_time": u["time_seconds"],
                "relres_improvement_factor(classical/network)": _safe_ratio(c["final_relres"], u["final_relres"]),
                "time_speedup_factor(classical/network)": _safe_ratio(c["time_seconds"], u["time_seconds"]),
            }
        )
    compare_simple = pd.DataFrame(rows).sort_values(["n", "rhs_mode", "K"]) if rows else pd.DataFrame()
    compare_simple_path = out_dir / "formal_classical_vs_network.csv"
    compare_simple.to_csv(compare_simple_path, index=False)

    # Keep previous detailed outputs for advanced analysis
    ranking = (
        df.groupby(["n", "method"], as_index=False)["final_relres"]
        .mean()
        .sort_values(["n", "final_relres"], ascending=[True, True])
    )
    ranking_path = out_dir / "formal_method_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    report = {
        "input_files": files,
        "rows_before_dedup": int(before),
        "rows_after_dedup": int(after),
        "outputs": {
            "dedup": str(dedup_path),
            "simple_table": str(simple_path),
            "classical_vs_network": str(compare_simple_path),
            "ranking": str(ranking_path),
        },
        "readme": {
            "simple_table": "Each row directly reports: iteration steps used(K), final precision(final_relres), threshold-reaching steps, and runtime.",
            "classical_vs_network": "Best classical LR-ADI method vs best unrolled network method per (n, rhs_mode, K).",
        },
    }
    report_path = out_dir / "formal_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
