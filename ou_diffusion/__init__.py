"""
1. ou_process.generate_ou_dataset     -- Stage 1 (Algorithm 1)
2. schedule.make_linear_schedule      -- Stage 2 (Eq. 6-8)
   schedule.q_sample                  -- forward sample (Algorithm 2)
3. kalman.kalman_filter               -- Stage 3 (Algorithm 3)
4. train.train_denoiser               -- Stage 4 training (Algorithm 4)
    sample.ddpm_sample                -- Stage 4 sampling (Algorithm 5)
    validate.validate                 -- marginal variance + lag-1 autocorr
"""
from .ou_process import (
    OUConstants,
    ou_constants,
    generate_ou_dataset,
    add_measurement_noise,
)
from .schedule import NoiseSchedule, make_linear_schedule, q_sample
from .kalman import KalmanResult, kalman_filter, steady_state_variance
from .unet1d import UNet1D, SinusoidalTimeEmbedding
from .ema import EMA
from .train import TrainConfig, train_denoiser
from .sample import ddpm_sample
from .validate import validate, ValidationReport, marginal_variance, lag1_autocorrelation

__all__ = [
    "OUConstants",
    "ou_constants",
    "generate_ou_dataset",
    "add_measurement_noise",
    "NoiseSchedule",
    "make_linear_schedule",
    "q_sample",
    "KalmanResult",
    "kalman_filter",
    "steady_state_variance",
    "UNet1D",
    "SinusoidalTimeEmbedding",
    "EMA",
    "TrainConfig",
    "train_denoiser",
    "ddpm_sample",
    "validate",
    "ValidationReport",
    "marginal_variance",
    "lag1_autocorrelation",
]
