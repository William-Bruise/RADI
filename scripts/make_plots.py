import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")

subprocess.run(
    [sys.executable, "-m", "radi_net.cli", "plot", "--csv", "results/storyline.csv", "--out-dir", "figures"],
    check=True,
    cwd=ROOT,
    env=ENV,
)
