import subprocess, sys
subprocess.run([sys.executable, "-m", "radi_net.cli", "plot", "--csv", "results/storyline.csv", "--out-dir", "figures"], check=True)
