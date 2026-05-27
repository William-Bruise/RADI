import subprocess, sys
subprocess.run([sys.executable, "-m", "radi_net.cli", "diagonal", "--out", "results/diagonal.csv", "--small"], check=True)
