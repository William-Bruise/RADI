import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")

# Formal diagonal-scale experiments requested by user.
# If you really need 515x512 specifically, add a rectangular benchmark separately;
# Lyapunov SPD matrix S is square, so we run n in {256, 512, 1024, 2048}.
for n in [256, 512, 1024, 2048]:
    out_csv = f"results/formal_diagonal_n{n}.csv"
    cmd = [
        sys.executable,
        "-c",
        (
            "from radi_net.experiments import run_diagonal_experiment;"
            f"run_diagonal_experiment('{out_csv}', seed=0, n_list=[{n}], K_list=[5,10,20])"
        ),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, env=ENV)

print("formal scale experiments complete")
