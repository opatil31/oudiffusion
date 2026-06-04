"""
Usage:
-----
python -m scripts.run --process ou --K 1000 --steps 6000
python -m scripts.run --process bm --K 1000 --steps 6000 --mu 0.3
python -m scripts.run --process ou --kalman-demo --R 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ou_diffusion import (
    PROCESSES,
    TrainConfig,
    add_measurement_noise,
    ddpm_sample,
    gaussian_loss_floor,
    get_process,
    kalman_filter,
    make_linear_schedule,
    steady_state_variance,
    train_denoiser,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="exact-toy diffusion pipeline")
    p.add_argument("--process", type=str, default="ou", choices=sorted(PROCESSES))
    # shared process parameters (interpretation depends on the process)
    p.add_argument("--theta", type=float, default=1.0, help="OU mean-reversion rate")
    p.add_argument("--sigma", type=float, default=1.0, help="diffusion coefficient")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--mu", type=float, default=0.0, help="BM drift")
    p.add_argument("--x0", type=float, default=0.0, help="BM start value")
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--N", type=int, default=10_000)
    p.add_argument("--K", type=int, default=1000, help="number of diffusion steps")
    p.add_argument("--steps", type=int, default=6_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=4_000)
    p.add_argument("--save-samples", type=str, default=None,
                   help="optional .npy path to save generated trajectories")
    p.add_argument("--kalman-demo", action="store_true", help="(OU only)")
    p.add_argument("--R", type=float, default=0.5, help="measurement variance for Kalman demo")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config for a fast end-to-end sanity check")
    return p.parse_args(argv)


def build_process(args):
    if args.process == "ou":
        return get_process("ou", theta=args.theta, sigma=args.sigma, dt=args.dt)
    if args.process == "bm":
        return get_process("bm", mu=args.mu, sigma=args.sigma, dt=args.dt, x0=args.x0)
    raise KeyError(args.process)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args.N = min(args.N, 2_000)
        args.L = 32
        args.K = 50
        args.steps = min(args.steps, 2_000)
        args.base_channels = 16
        args.n_samples = 1_000

    proc = build_process(args)
    print(f"process: {proc.name} (d={proc.d}, gaussian={proc.is_gaussian})")

    # stage 1
    x0 = proc.exact_sample(args.N, args.L, seed=args.seed)
    print(f"generated dataset: {x0.shape}  (N x d x L)")
    print("[exact-sampler self-check]")
    print(proc.validate(x0))

    # stage 3
    if args.kalman_demo:
        if args.process != "ou":
            raise SystemExit("--kalman-demo is defined for the OU process only")
        clean = x0[:, 0, :]
        y = add_measurement_noise(clean, args.R, seed=args.seed + 1)
        res = kalman_filter(y, proc.c, args.R)
        raw = float(np.sqrt(((y - clean) ** 2).mean()))
        filt = float(np.sqrt(((res.x_filt - clean) ** 2).mean()))
        print(f"[Kalman demo] R={args.R}  raw RMSE={raw:.4f}  filtered RMSE={filt:.4f}  "
              f"steady-state P={res.P_filt[-1]:.4f} "
              f"(Riccati={steady_state_variance(proc.c, args.R):.4f})")
    
    # stage 2
    schedule = make_linear_schedule(T=args.K)
    if proc.is_gaussian:
        floor = gaussian_loss_floor(proc.covariance(args.L),
                                    schedule.alphas_cumprod.numpy())
        print(f"irreducible eps-MSE loss floor for this config: {floor:.4f}")

    # stage 4
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                      base_channels=args.base_channels, seed=args.seed)
    model, ema, _ = train_denoiser(x0, schedule, cfg, device=args.device)
    ema_model = ema.clone_into(model)
    samples = ddpm_sample(ema_model, schedule, n=args.n_samples, L=args.L,
                          d=proc.d, device=args.device, seed=args.seed + 2)
    samples = samples.numpy()
    print("\n[generated samples]")
    print(proc.validate(samples))
    if args.save_samples:
        np.save(args.save_samples, samples)
        print(f"saved generated samples to {args.save_samples}")
    return samples


if __name__ == "__main__":
    main()