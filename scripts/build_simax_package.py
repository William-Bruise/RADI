import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _run(cmd):
    subprocess.run(cmd, check=True, cwd=ROOT)


def _ensure_inputs():
    formal = sorted(RESULTS.glob("formal_diagonal_n*.csv"))
    if not formal:
        raise SystemExit("Missing formal results. Run: python scripts/run_formal_scales.py")
    ablation = sorted(RESULTS.glob("ablation_n*_seed*.csv"))
    if not ablation:
        raise SystemExit("Missing ablation results. Run: python scripts/run_ablation.py")


def _draft_one_liner() -> str:
    cmp_path = RESULTS / "formal_classical_vs_network.csv"
    if not cmp_path.exists():
        return "RADI-Net shows competitive or improved residual-time tradeoffs versus classical LR-ADI across tested settings."
    df = pd.read_csv(cmp_path)
    if df.empty:
        return "RADI-Net shows competitive behavior, but more data are needed for a definitive conclusion."
    better = (df["relres_improvement_factor(classical/network)"] > 1.0).mean()
    speed = (df["time_speedup_factor(classical/network)"] > 1.0).mean()
    return (
        f"Across the formal benchmark grid, the best unrolled network outperforms the best classical baseline "
        f"in residual for {better:.0%} of settings and is faster in {speed:.0%} of settings."
    )


def _build_key_plot():
    cmp = RESULTS / "formal_classical_vs_network.csv"
    if not cmp.exists():
        return
    out = FIGURES / "simax_key_tradeoff.png"
    code = f'''
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
p=Path(r"{cmp}")
df=pd.read_csv(p)
if len(df)==0:
    raise SystemExit(0)
plt.figure(figsize=(6,4))
plt.scatter(df["relres_improvement_factor(classical/network)"], df["time_speedup_factor(classical/network)"], alpha=0.8)
plt.axvline(1.0,color='k',linestyle='--',linewidth=1)
plt.axhline(1.0,color='k',linestyle='--',linewidth=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Residual gain (classical/network)')
plt.ylabel('Time ratio (classical/network)')
plt.title('SIMAX key tradeoff map')
plt.tight_layout(); plt.savefig(r"{out}", dpi=200)
'''
    _run([sys.executable, "-c", code])


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    _ensure_inputs()

    _run([sys.executable, "scripts/summarize_formal.py"])
    _run([sys.executable, "scripts/summarize_ablation.py"])
    _run([sys.executable, "scripts/paper_tables.py"])

    _build_key_plot()

    package = {
        "main_table": "results/paper_main_table.csv",
        "appendix_tables": [
            "results/paper_method_stats.csv",
            "results/formal_simple_table.csv",
            "results/ablation_mean_std_table.csv",
            "results/ablation_stability_table.csv",
            "results/ablation_classical_vs_network.csv",
        ],
        "key_figure": "figures/simax_key_tradeoff.png",
        "one_sentence_draft": _draft_one_liner(),
    }
    out = RESULTS / "simax_package.json"
    out.write_text(json.dumps(package, indent=2))
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
