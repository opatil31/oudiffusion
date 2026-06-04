"""
Usage:
python -m scripts.runner --K 1000 --steps 6000 (full run)
python -m scripts.runner --K 1000 --steps 6000 --kalman-demo --R 0.5       # filtering
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ou_diffusion import (
    ou_constants,
    generate_ou_dataset,
    add_measurement_noise,
    make_linear_schedule,
    kalman_filter,
    steady_state_variance,
    TrainConfig,
    train_denoiser,
    ddpm_sample,
    validate,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OU diffusion pipeline")
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--N", type=int, default=10_000)
    # diffusion
    p.add_argument("--K", type=int, default=200, help="number of diffusion steps")
    # training
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    # sampling / eval
    p.add_argument("--n-samples", type=int, default=4_000)
    # extras
    p.add_argument("--kalman-demo", action="store_true")
    p.add_argument("--R", type=float, default=0.5, help="measurement variance for Kalman demo")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny config for a fast end-to-end sanity check",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args.N = min(args.N, 2_000)
        args.L = 32
        args.K = 50
        args.steps = min(args.steps, 2_000)
        args.base_channels = 16
        args.n_samples = 1_000

    c = ou_constants(args.theta, args.sigma, args.dt)
    print(f"OU constants: a={c.a:.4f}  q={c.q:.4f}  s2 (stationary var)={c.s2:.4f}")
    # stae 1:
    x0 = generate_ou_dataset(args.theta, args.sigma, args.dt, args.L, args.N, seed=args.seed)
    print(f"generated dataset: {x0.shape}  (N x L)")

    # stage 3 - include measurement noise for kalman showcase
    if args.kalman_demo:
        y = add_measurement_noise(x0, args.R, seed=args.seed + 1)
        res = kalman_filter(y, c, args.R)
        raw_rmse = float(np.sqrt(((y - x0) ** 2).mean()))
        filt_rmse = float(np.sqrt(((res.x_filt - x0) ** 2).mean()))
        print(
            f"[Kalman demo] R={args.R}  raw RMSE={raw_rmse:.4f}  "
            f"filtered RMSE={filt_rmse:.4f}  "
            f"steady-state P={res.P_filt[-1]:.4f} (Riccati={steady_state_variance(c, args.R):.4f})"
        )

    # stage 2
    schedule = make_linear_schedule(T=args.K)

    # stage 4 - training
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        seed=args.seed,
    )
    model, ema, _ = train_denoiser(x0, schedule, cfg, device=args.device)
    ema_model = ema.clone_into(model)
    samples = ddpm_sample(ema_model, schedule, n=args.n_samples, L=args.L,
                          device=args.device, seed=args.seed + 2).numpy()
    print()
    print(validate(samples, c))
    return samples


if __name__ == "__main__":
    main()
