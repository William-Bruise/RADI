import argparse
from .experiments import run_diagonal_experiment, run_direct_training_sanity, run_laplacian_experiment, run_storyline_demo
from .plotting import plot_storyline_demo


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    for name in ["diagonal", "laplacian", "direct-sanity", "storyline"]:
        q = sp.add_parser(name)
        q.add_argument("--out", required=True)
        q.add_argument("--small", action="store_true")
        q.add_argument("--large", action="store_true")
        q.add_argument("--seed", type=int, default=0)

    q = sp.add_parser("plot")
    q.add_argument("--csv", required=True)
    q.add_argument("--out-dir", required=True)

    a = p.parse_args()
    if a.cmd == "diagonal":
        run_diagonal_experiment(a.out, large=a.large and not a.small, seed=a.seed)
    elif a.cmd == "laplacian":
        run_laplacian_experiment(a.out, large=a.large and not a.small, seed=a.seed)
    elif a.cmd == "direct-sanity":
        run_direct_training_sanity(a.out, seed=a.seed)
    elif a.cmd == "storyline":
        run_storyline_demo(a.out, small=not a.large, seed=a.seed)
    elif a.cmd == "plot":
        plot_storyline_demo(a.csv, a.out_dir)


if __name__ == "__main__":
    main()
