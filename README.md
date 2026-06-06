# Space-Time Diffusion Ongoing Project
Author: Oankar Patil

**Five stochastic processes with closed-form ground truth: scalar OU, Brownian motion, geometric Brownian motion, a vector OU / stochastic oscillator, and a 1D stochastic heat equation**

These are used to train and, more importantly, analyze denoising diffusion models. Every stage of the pipeline (data, forward process, filter, training loss, sampler, output statistics) is verified against a value it provably must produce.

Main Findings: First, in five separately documented instances the training loss sat exactly on its analytic optimum while calibrated sample statistics measured real distributional biases between $z=+3$ and $z=+30$: the $\varepsilon$-MSE loss is a necessary diagnostic and a blind one. Second, the residual error of a conv-U-Net diffusion model is consistently located in *temporal dynamics* (decay rates, phases, long-horizon coherence) rather than marginal power: generated fine structure decorrelates a measurable few percent too fast. These should translate to audio/video cases.

The pipeline follows the reading-group note *"A Fully Worked 1D Example: From a Scalar Wiener Process to a U-Net Diffusion Model"* (Algorithms 1–5); the process roster, the verification stack, and the findings below are this repository's additions. This is part of a larger effort on autoregressive / space–time diffusion project.

---

## 1. Design principle:

This project focuses on the data sources are chosen so that the dataset has exact statistics.

## 2. The processes:

| process | dynamics | new difficulty | exact verification anchors |
|---|---|---|---|
| `ou` | $dx=-\theta x\,dt+\sigma\,dW$ | baseline | stationary $\sigma^2/2\theta$, lag-1 $e^{-\theta\Delta t}$, Kalman, oracle + loss floor |
| `bm` | $dx=\mu\,dt+\sigma\,dW$ | nonstationary, singular cov | variance-growth slope, increment statistics |
| `gbm` | $dx=\mu x\,dt+\sigma x\,dW$ | **non-Gaussian**, positive | log-space reduction, lognormal moments, linear-denoiser bound |
| `vou` / `osc` | $dx=-\Theta x\,dt+B\,dW$ | matrix dynamics, $d{=}2$ channels | Van Loan $A,Q$; Lyapunov $S$; lag-$h$ laws $A^hS$ |
| `heat` | $du=[-\lambda u+\kappa\,\partial_x^2u]\,dt+\sigma\,dW$ | spatial field, stiff spectrum | per-mode OU targets, $\theta_j=\lambda+4\kappa\sin^2(\pi j/d)$ |

All five sit behind one abstraction (`ou_diffusion/processes/`): a `Process` is an exact trajectory sampler **plus its own ground truth** — validation targets, the exact one-step transition kernel $x_{\ell+1}\mid x_\ell \sim \mathcal N(Ax_\ell+b,\,Q)$ (exposed deliberately: it is what autoregressive validation will condition on), and, for Gaussian processes, the mean and covariance of the flattened trajectory, which feed the oracle and loss floor of Section 4. Vector-valued systems map onto the existing U-Net with **channels = state dimensions**, no architecture change.

**Scalar OU** is the continuous-time AR(1): exact transition $x_\ell = a x_{\ell-1} + \sqrt q\,\xi$ with $a=e^{-\theta\Delta t}$, $q=\frac{\sigma^2}{2\theta}(1-e^{-2\theta\Delta t})$, stationary $\mathcal N(0,\sigma^2/2\theta)$, and because the model is linear-Gaussian, an exact Kalman filter under measurement noise, kept deliberately separate from diffusion (designed noise on the diffusion clock vs given noise on the physical clock).

**Brownian motion** ($\pm$ drift) is the $a=1$ boundary case and the first *nonstationary* dataset: $\mathrm{Var}(x_\ell)=\sigma^2\ell\Delta t$ grows, the covariance $\sigma^2\Delta t\min(i,j)$ is non-Toeplitz and singular at $\ell=0$, handled exactly by the oracle, since a deterministic coordinate has irreducible loss 0. It doubles as an architectural probe: translation-equivariant convolutions sense absolute position only through boundary padding, which proved sufficient at $L=64$ (all targets pass at the 1–2% level with no coordinate channel).

**Geometric Brownian motion** is the first **non-Gaussian** case: lognormal marginals, strictly positive, right-skewed. By design there is no Gaussian oracle; verification goes through the exact log-space reduction ($\log x$ *is* Brownian motion with drift $\nu=\mu-\sigma^2/2$) plus lognormal moment targets and positivity. It is paired with two instruments: the **fitted-Gaussian baseline** (moment-matched control, passes every check for every Gaussian process, the honest statement that diffusion is overkill there) and the **linear-denoiser bound** (the Gaussian loss floor of the empirical covariance). Measured: the trained network's loss plateaus ≈18% *below* the linear bound (0.063 vs 0.0765), its nonlinearity provably necessary, while the baseline fails the shape checks exactly as designed (3.8% impossible negative values; terminal skewness 0.03 against a target of 2.96).

**Vector OU / stochastic oscillator** introduces matrix dynamics with the same closed forms the group's state-space forward process uses, under exact tests: the Van Loan block trick yields $A=e^{-\Theta\Delta t}$ and the Lyapunov-integral $Q$ from one matrix exponential (diagonal-$\Theta$ reproduces scalar OU to $10^{-12}$; Lyapunov residuals $<10^{-10}$). Validation exercises the full matrix $A$ through lag-$h$ cross-covariances $A^hS$, necessary because the rotation×contraction default has an isotropic stationary covariance carrying no rotation information. The oscillator configuration (position/velocity state, rank-deficient $B$ yet full-rank $Q$ by controllability) adds textbook targets: $\mathrm{Var}(\mathrm{pos})=\sigma^2/4\zeta\omega^3$, $\mathrm{Var}(\mathrm{vel})=\sigma^2/4\zeta\omega$, zero position–velocity covariance, sign-flipping autocovariance at half period.

**The 1D stochastic heat equation** on a periodic ring closes the roster: discretized as a vector OU with $\Theta=\lambda I+\kappa\,\mathrm{Lap}$, it inherits the entire Van Loan / Lyapunov / oracle machinery, and its $(N,d,L)$ output is structurally a low-resolution video. The new content is spectral: $\Theta$ diagonalizes in the Fourier basis into independent scalar OU modes whose stationary variances $\sigma^2/2\theta_j$ span ≈33:1 at the defaults. Because translation invariance equalizes all *site* variances, channel normalization reduces to a global rescale here, the heat equation isolates *within-channel spectral stiffness*, the toy form of audio's spectral decay, and validation therefore adds three mode-space statistics (max-over-modes relative variance error, max-over-modes lag-1 error, max cross-mode correlation), calibrated by the same replication machinery. A prediction was recorded in `processes/heat.py` before the runs: uniform-$k$ $\varepsilon$-MSE should under-serve the low-power high-frequency modes. Outcome in Section 5: confirmed for dynamics, refuted for variance.

## 3. The pipeline

| Stage | Math | Module |
|---|---|---|
| 1. Data | exact process sampler; $x^{(0)}\in\mathbb R^{d\times L}$, $N$ trajectories | `ou_diffusion/processes/` |
| 2. Forward diffusion | $x^{(k)}=\sqrt{\bar\alpha_k}\,x^{(0)}+\sqrt{1-\bar\alpha_k}\,\epsilon$, linear $\beta$ schedule | `schedule.py` |
| 3. Filtering (OU demo) | scalar Kalman predict/update on $y_\ell=x_\ell+v_\ell$ | `kalman.py` |
| 4. Reverse process | 1D U-Net $\epsilon_\theta(x^{(k)},k)$, MSE objective, ancestral sampling | `unet1d.py`, `train.py`, `sample.py` |
| Validation | per-process targets, oracle, floors, calibrated z-statistics | `processes/`, `oracle.py` |

The U-Net is the note's 3-resolution design ($C \to 2C\times\frac L2 \to 4C\times\frac L4$ and back with skips), the diffusion step injected into every residual block through a sinusoidal embedding. Training supports EMA and optional cosine LR decay (`--lr-final`); `--normalize` adds exactly-invertible per-channel standardization with the printed loss floor rescaled consistently.

**Two clocks.** Physical time $\ell$ (indexing the trajectory) and diffusion time $k$ (indexing the noising ladder) are distinct axes. The generative model treats each trajectory as a fixed data point and never sees the physical SDE again. Keeping the clocks separate is the conceptual backbone of the autoregressive extension, where they form a frames × noise-levels lattice.

## 4. Verification:

**Loss floor:** For Gaussian data $x^{(0)}\sim\mathcal N(\mu,\Sigma)$ the posterior covariance of $\epsilon$ given $x^{(k)}$ is closed-form, so the irreducible $\epsilon$-MSE averaged over uniform $k$ is a number, e.g. $\mathcal L^\star=\mathbf{0.1071}$ for OU at $(\theta,\sigma,\Delta t,L,K)=(1,1,0.05,64,1000)$. A network plateauing here *is* the optimal denoiser. (`oracle.gaussian_loss_floor`, any Gaussian process, scalar or vector.)

**Analytic oracle:** The optimal noise predictor for Gaussian data is linear and known: $\epsilon^\star(x^{(k)},k)=\sqrt{1-\bar\alpha_k}\,\Sigma_k^{-1}(x^{(k)}-\sqrt{\bar\alpha_k}\mu)$. Plugging $\epsilon^\star$ into the production sampler (`scripts/verify_optimal_denoiser.py`) isolates sampler/schedule error from learning error, a decomposition usually impossible in generative modeling. For non-Gaussian processes the **linear-denoiser bound** (Gaussian floor of the empirical covariance) plays the complementary role: a loss below it proves the network beats every linear denoiser.

**Calibrated statistics:** Output statistics are checked against exact targets, and because the deviation statistics are norms with strictly positive null expectation, raw percentages conflate estimator noise with model bias. Every vector-process check is therefore calibrated against exact-sampler replications at the same sample size: the report's target column is the finite-sample noise-floor mean, each check carries the replication standard error, and deviations read in z-units. High-variance scalar checks (terminal skewness) carry bootstrap standard errors. Held-out functional batteries (unused lags, the full $dL\times dL$ trajectory covariance, Gaussianity probes) guard against tuning to the watched metrics.

**Worked demonstration (scalar OU, $K{=}1000$, 6k steps, $N{=}10{,}000$, $L{=}64$):**

| | marginal variance | lag-1 autocorrelation |
|---|---|---|
| exact OU target | 0.5000 | 0.9512 |
| analytic oracle $\epsilon^\star$ through the sampler | 0.491 | 0.9500 |
| **trained 1D U-Net** | **0.4877** | **0.9494** |

Training loss plateaus at 0.10–0.12 around the 0.1071 floor (per-step wander is minibatch composition: the per-$k$ floor spans ≈1 at $k{=}0$ to ≈0 at $k{=}K$). The network sits at the loss floor *and* matches the oracle, so the residual ~2% variance gap is **sampler discretization bias**, not learning error. Nothing in the result is unexplained.

## 5. Findings

### 5.1 Terminal SNR and the choice of $K$ (a sampler finding)

The note suggests $K=200$ "works fine," but the linear $\beta\in[10^{-4},0.02]$ schedule only reaches $\bar\alpha_K\approx0.132$ there: ~36% of signal amplitude survives the forward process, so initializing the reverse process at pure $\mathcal N(0,I)$ is mis-specified. The oracle measures it exactly:

| $K$ | $\bar\alpha_K$ | oracle marginal variance (target 0.5) |
|---|---|---|
| 200 | 0.132 | 0.365 (−27%) |
| 1000 | $4\times10^{-5}$ | 0.491 (−1.8%) |

This reproduces the zero-terminal-SNR issue (Lin et al., 2024) with the learning component ruled out. All commands below default to $K=1000$.

### 5.2 Learning findings:

**Acceptance criterion** (pre-stated for any future process:) a process is *closed* when, on a fresh draw of 10,000 trajectories, every replication-calibrated check and a held-out functional set of tests, including lags out to $L/2$ and a Gaussianity probe where applicable, falls within 5% absolute deviation, with any statistically significant residual error-barred, replicated, and attributed via the analytic oracle and an explicit elimination chain.

**Rotation–contraction vector OU:** Rerun at 16k steps under the calibrated checks, it lands at the same wall as the oscillator: genuine second-order bias 1.7–2.2% ($z=+3.5/+3.5/+3.8$ against the noise floors), largest at the longest lag checked. Notably it reaches that wall at 32 channels, without the capacity increase the oscillator requires, its correlation length (≈20 steps) fits inside the receptive field, so the wall's height is set by the architecture–objective pair.

**Oscillator:** The best configuration ($K{=}1000$, 16k steps, 64 channels) reproduces the exact second-order law to 2.5–3.5% out to quarter-period lags, with the residual at $z=+3.2/+3.1/+4.5$ against calibrated noise floors. The residual survived a six-hypothesis elimination: sampler (the analytic oracle is statistically indistinguishable from the exact sampler at $K=1000$), capacity (32→64 channels cut the bias 3×, then saturated), training length (8k→16k flat), channel scale (per-channel normalization ablation: unchanged), receptive field ($L=32$ diagnostic: genuine bias unchanged), and optimization schedule (cosine LR decay, verified active from its one-sided convergence fingerprint). A held-out set of tests (lags 4/8/32, the full $dL\times dL$ trajectory covariance, marginal kurtosis) landed on the same bias curve, ruling out metric overfitting, and revealed structure: the bias grows with temporal horizon (2.5% at lag-1 to 7.4% at half-period), consistent with few-percent errors in learned dynamics compounding coherently in phase and envelope. Closed, with the half-period lag as the documented exception.

**Heat equation:** By site-basis statistics the best-behaved process in the roster (genuine deviations ≈1.8–2.0%), and the mode-resolved instruments show why that judgment would have been wrong: every mode above the second harmonic is *over-damped* by a uniform ≈4–7% in decay rate (a monotone-in-frequency correlation loss peaking at $z=+30$ on the fastest mode), all sixteen modes are *under-dispersed* (sign test alone: $p=2^{-16}$) with the deficit peaking at −4% in the mid-band, and the DC mode, whose coherence time of 80 steps exceeds the 64-step window, carries a level shift ($z=+6.8$ on the mean check, the only mean failure in the project). The spectral extremes are nearly perfect in variance, refuting the half of the standing prediction that said low-power modes would be neglected, the confirmed half is that *dynamics* degrade with frequency. No translation-symmetry breaking across degenerate mode pairs. Closed, with the mode-resolved over-damping and mid-band shrinkage as the documented, attributed exceptions.

**Synthesis:** Across all three vector-valued processes the conv-U-Net + uniform-$k$ $\varepsilon$-MSE pair errs specifically in *temporal structure*, decay rates, phases, long-horizon coherence, while matching marginal power almost everywhere. In five separately documented instances the training loss sat on its analytic floor while sample-space instruments measured biases between $z=+3$ and $z=+30$: the loss is a necessary diagnostic and a blind one, and the oracle plus replication-calibrated statistics are the operative instruments. This is the underlying idea of generated transients decorrelate too fast, which motivates measurement for long-context denoiser architectures and the baseline that autoregressive-rollout experiments on these same processes should be judged against.

## 6. Quickstart

```bash
pip install -r requirements.txt              # numpy, torch, matplotlib

# Full runs (GPU with --device cuda):
python -m scripts.run --process ou --K 1000 --steps 6000 --save-samples samples.npy
python -m scripts.run --process bm --mu 0.3 --K 1000 --steps 6000
python -m scripts.run --process gbm --K 1000 --steps 6000 --baseline
python -m scripts.run --process vou --K 1000 --steps 16000
python -m scripts.run --process osc --K 1000 --steps 16000 --base-channels 64 --normalize --lr-final 2e-5   # OSC closure config
python -m scripts.run --process heat --K 1000 --steps 16000 --base-channels 64 --n-samples 10000

# Optional: Kalman filtering demo on noisy measurements (OU)
python -m scripts.run --process ou --K 1000 --steps 6000 --kalman-demo --R 0.5
```

## 7. Repository layout

```
ou_diffusion/
  processes/        data sources with exact targets (the Process interface)
    base.py           exact_sample / validate / mean / covariance / transition
    ou.py             Ornstein-Uhlenbeck (stationary, Gaussian)
    brownian.py       Brownian motion +- drift (nonstationary, Gaussian)
    gbm.py            geometric Brownian motion (non-Gaussian, positive, skewed)
    vector_ou.py      vector OU + stochastic oscillator (matrix dynamics, d = 2)
    heat.py           1D stochastic heat equation (spatial field, stiff mode spectrum)
  baseline.py       FittedGaussianBaseline: moment-matched control model
  normalize.py      ChannelNormalizer: exactly-invertible per-channel standardization
  oracle.py         AnalyticGaussianDenoiser + gaussian_loss_floor (any Gaussian process)
  ou_process.py     Stage 1: exact OU sampler (Algorithm 1)
  schedule.py       Stage 2: DDPM schedule + q_sample (Eq. 6-8)
  kalman.py         Stage 3: scalar Kalman filter (Algorithm 3)
  unet1d.py         Stage 4: 1D U-Net eps-predictor (channels = state dims)
  train.py          Stage 4: training loop (Algorithm 4) + EMA + cosine LR decay
  sample.py         Stage 4: ancestral sampler (Algorithm 5)
  validate.py       OU statistics + irreducible loss floor (Phase-A interface)
scripts/
  run.py                      process-agnostic end-to-end pipeline
  run_ou.py                   Phase-A alias (forwards to run.py --process ou)
  verify_optimal_denoiser.py  analytic oracle per process, K sweep
  plot_samples.py             real-vs-generated figure
tests/                        per-stage and per-process unit tests
```

## 8. Status and My Direction

**Current:** Five exactly-verified processes (scalar OU, Brownian motion, geometric Brownian motion, vector OU / stochastic oscillator, 1D stochastic heat equation).

**Next: autoregressive diffusion on the same processes**, validated against the exact conditionals every process already exposes through `transition()` one step ahead $\mathcal N(Ax+b,\,Q)$, with closed-form $h$-step predictive laws, so gap becomes an exactly measurable curve, with Section 5's horizon and over-damping results as its baselines. Per-frame noise levels (Diffusion Forcing / Rolling Diffusion style).

## References

- Reading-group note: *A Fully Worked 1D Example: From a Scalar Wiener Process to a U-Net Diffusion Model* (2026) — pipeline design (Algorithms 1–5).
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Lin, Liu, Li, Yang. *Common Diffusion Noise Schedules and Sample Steps are Flawed.* WACV 2024 — zero terminal SNR.
- Särkkä. *Bayesian Filtering and Smoothing.* CUP 2013 — Kalman filtering background.
- Chen et al. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion.* NeurIPS 2024; Ruhe et al. *Rolling Diffusion Models.* ICML 2024 — next-phase directions.
