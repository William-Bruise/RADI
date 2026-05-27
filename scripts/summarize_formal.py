import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_ratio(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a / b)


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted(glob.glob(str(root / "results" / "formal_diagonal_n*.csv")))
    if not files:
        raise SystemExit("No formal result files found: results/formal_diagonal_n*.csv")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    key_cols = ["experiment", "n", "matrix_type", "rhs_mode", "rank_B", "K", "method", "gamma", "seed"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    after = len(df)

    # best method ranking by mean final residual for each n
    ranking = (
        df.groupby(["n", "method"], as_index=False)["final_relres"]
        .mean()
        .sort_values(["n", "final_relres"], ascending=[True, True])
    )

    # threshold summaries
    thresh_cols = ["steps_to_1e-4", "steps_to_1e-6", "steps_to_1e-8", "time_seconds", "final_relres"]
    summary = (
        df.groupby(["n", "rhs_mode", "K", "method"], as_index=False)[thresh_cols]
        .mean(numeric_only=True)
        .sort_values(["n", "rhs_mode", "K", "final_relres"], ascending=[True, True, True, True])
    )

    # compare RADI-Net-spectral (gamma=0) vs LR-ADI-logspace
    rows = []
    for (n, rhs_mode, K), g in df.groupby(["n", "rhs_mode", "K"]):
        base = g[(g["method"] == "LR-ADI-logspace") & (g["gamma"] == 0.0)]
        radi = g[(g["method"] == "RADI-Net-spectral") & (g["gamma"] == 0.0)]
        if len(base) == 0 or len(radi) == 0:
            continue
        b = base.iloc[0]
        r = radi.iloc[0]
        rows.append(
            {
                "n": n,
                "rhs_mode": rhs_mode,
                "K": K,
                "baseline_final_relres": b["final_relres"],
                "radinet_final_relres": r["final_relres"],
                "residual_improvement_factor": _safe_ratio(b["final_relres"], r["final_relres"]),
                "baseline_time": b["time_seconds"],
                "radinet_time": r["time_seconds"],
                "time_speedup_factor": _safe_ratio(b["time_seconds"], r["time_seconds"]),
                "baseline_steps_1e6": b["steps_to_1e-6"],
                "radinet_steps_1e6": r["steps_to_1e-6"],
            }
        )
    compare = pd.DataFrame(rows)

    out_dir = root / "results"
    dedup_path = out_dir / "formal_all_dedup.csv"
    ranking_path = out_dir / "formal_method_ranking.csv"
    summary_path = out_dir / "formal_threshold_summary.csv"
    compare_path = out_dir / "formal_radinet_vs_logspace.csv"
    report_path = out_dir / "formal_report.json"

    df.to_csv(dedup_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    summary.to_csv(summary_path, index=False)
    compare.to_csv(compare_path, index=False)

    report = {
        "input_files": files,
        "rows_before_dedup": int(before),
        "rows_after_dedup": int(after),
        "outputs": {
            "dedup": str(dedup_path),
            "ranking": str(ranking_path),
            "threshold_summary": str(summary_path),
            "radinet_vs_logspace": str(compare_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
