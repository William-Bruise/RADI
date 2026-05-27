import subprocess, sys

cmds = [
    ["-m", "radi_net.cli", "diagonal", "--out", "results/diagonal.csv", "--small"],
    ["-m", "radi_net.cli", "laplacian", "--out", "results/laplacian.csv", "--small"],
    ["-m", "radi_net.cli", "direct-sanity", "--out", "results/direct_sanity.csv"],
    ["-m", "radi_net.cli", "storyline", "--out", "results/storyline.csv", "--small"],
]
for c in cmds:
    subprocess.run([sys.executable] + c, check=True)
print("all small experiments complete")
