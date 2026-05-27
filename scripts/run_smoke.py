import subprocess, sys

subprocess.run([sys.executable, "-m", "radi_net.cli", "diagonal", "--out", "results/smoke_diagonal.csv", "--small"], check=True)
subprocess.run([sys.executable, "-m", "radi_net.cli", "storyline", "--out", "results/smoke_storyline.csv", "--small"], check=True)
print("smoke complete")
