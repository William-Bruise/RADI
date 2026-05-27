import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _load_formal(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "results" / "formal_diagonal_n*.csv")))
    if not files:
        raise SystemExit("No formal files found. Run scripts/run_formal_scales.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["experiment", "n", "matrix_type", "rhs_mode", "rank_B", "K", "method", "gamma", "seed"]
    return df.drop_duplicates(subset=key, keep="first")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results"
    df = _load_formal(root)

    # Main paper table: for each n/rhs/K compare best classical vs best network
    df["family"] = np.where(df["method"].str.startswith("RADI-Net"), "network", "classical")
    rows = []
    for (n, rhs, K), g in df.groupby(["n", "rhs_mode", "K"]):
        cands_c = g[g["family"] == "classical"].sort_values("final_relres")
        cands_n = g[g["family"] == "network"].sort_values("final_relres")
        if len(cands_c) == 0 or len(cands_n) == 0:
            continue
        c = cands_c.iloc[0]
        u = cands_n.iloc[0]
        rows.append({
            "n": n,
            "rhs_mode": rhs,
            "K": K,
            "best_classical": c["method"],
            "classical_relres": c["final_relres"],
            "classical_time_s": c["time_seconds"],
            "best_network": u["method"],
            "network_gamma": u["gamma"],
            "network_relres": u["final_relres"],
            "network_time_s": u["time_seconds"],
            "relres_gain_classical_over_network": c["final_relres"] / max(u["final_relres"], 1e-300),
            "time_ratio_classical_over_network": c["time_seconds"] / max(u["time_seconds"], 1e-300),
        })
    paper_main = pd.DataFrame(rows).sort_values(["n", "rhs_mode", "K"])
    paper_main.to_csv(out_dir / "paper_main_table.csv", index=False)

    # Aggregate mean±std by method for reporting paragraphs
    agg = (
        df.groupby(["n", "K", "method", "gamma"], as_index=False)
        .agg(final_relres_mean=("final_relres", "mean"),
             final_relres_std=("final_relres", "std"),
             time_mean=("time_seconds", "mean"),
             time_std=("time_seconds", "std"))
        .sort_values(["n", "K", "final_relres_mean"])
    )
    agg.to_csv(out_dir / "paper_method_stats.csv", index=False)

    print("wrote:", out_dir / "paper_main_table.csv")
    print("wrote:", out_dir / "paper_method_stats.csv")


if __name__ == "__main__":
    main()
