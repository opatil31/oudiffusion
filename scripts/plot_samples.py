from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ou_diffusion import PROCESSES, get_process

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=str, required=True, help=".npy from run_ou --save-samples")
    p.add_argument("--out", type=str, default="fig_samples.png")
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--process", type=str, default="ou", choices=sorted(PROCESSES))
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required for plotting: pip install matplotlib")

    gen = np.load(args.samples)
    if gen.ndim == 3:
        gen = gen[:, 0, :]
    L = gen.shape[1]
    if args.process == "ou":
        proc = get_process("ou", theta=args.theta, sigma=args.sigma, dt=args.dt)
        band = 2.0 * np.sqrt(proc.c.s2)
    else:
        proc = get_process("bm", mu=args.mu, sigma=args.sigma, dt=args.dt, x0=args.x0)
        band = 2.0 * np.sqrt(args.sigma**2 * args.dt * (L - 1))
    real = proc.exact_sample(args.n_show, L, seed=args.seed)[:, 0, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, data, title in (
        (axes[0], real, f"{proc.name} data (exact sampler)"),
        (axes[1], gen[: args.n_show], "diffusion samples (trained U-Net)"),
    ):
        ax.axhspan(-band, band, alpha=0.08, color="gray")
        for row in data:
            ax.plot(row, lw=1.0)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("physical time step")
    axes[0].set_ylabel("x")
    fig.suptitle("Shaded band: +/- 2 stationary std", fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()