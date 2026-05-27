import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")

# Ablation over seeds + K + gamma at two representative scales
scales = [256, 1024]
Ks = [5, 10, 20]
seeds = [0, 1, 2]

for n in scales:
    for seed in seeds:
        out_csv = f"results/ablation_n{n}_seed{seed}.csv"
        code = (
            "from radi_net.experiments import run_diagonal_experiment;"
            f"run_diagonal_experiment('{out_csv}', seed={seed}, n_list=[{n}], K_list={Ks})"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env=ENV)

print("ablation runs complete")
