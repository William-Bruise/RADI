import subprocess, sys
subprocess.run([sys.executable, "-m", "radi_net.cli", "laplacian", "--out", "results/laplacian.csv", "--small"], check=True)
