import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_residual_curves(csv_path, out_path, filter_dict=None):
    df = pd.read_csv(csv_path)
    if filter_dict:
        for k, v in filter_dict.items():
            df = df[df[k] == v]
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        y = json.loads(row["relres_history_json"])
        plt.semilogy(range(1, len(y) + 1), y, label=f"{row['method']}:{row['rhs_mode']}")
    plt.legend(fontsize=8)
    plt.xlabel("Iteration")
    plt.ylabel("Relative residual")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_shift_locations(lambdas, weights, methods_to_shifts, out_path):
    plt.figure(figsize=(8, 4))
    plt.scatter(lambdas, weights, s=5, alpha=0.3, label="spectral weights")
    for name, shifts in methods_to_shifts.items():
        plt.scatter(shifts, np.interp(shifts, np.sort(lambdas), np.sort(weights)), marker="x", label=name)
    plt.xscale("log")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_final_residual_bar(csv_path, out_path):
    df = pd.read_csv(csv_path)
    g = df.groupby("method")["final_relres"].mean().sort_values()
    plt.figure(figsize=(7, 4)); plt.bar(g.index, g.values)
    plt.yscale("log"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_storyline_demo(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_residual_curves(csv_path, os.path.join(out_dir, "storyline_residuals.png"))
    plot_final_residual_bar(csv_path, os.path.join(out_dir, "storyline_final_residual_bar.png"))
