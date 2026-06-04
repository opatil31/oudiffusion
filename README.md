# Implementation of Prof Vishwanath's OU Diffusion Note
Author: Oankar Patil

My pipeline implementation here follows the reading-group note *"A Fully Worked 1D Example: From a Scalar Wiener Process to a U-Net Diffusion Model"* (Algorithms 1–5).

---
## 1. Why the Ornstein–Uhlenbeck process

The OU process is the linear scalar Itô SDE

$$dx_t = -\theta\, x_t\, dt + \sigma\, dW_t, \qquad \theta, \sigma > 0,$$

the continuous-time analogue of an AR(1) model: mean-reverting toward 0 at rate $\theta$, driven by a scalar Wiener process. It is the cleanest possible data source for a diffusion model because it is **exact at every level**:

- **Exact sampler.** The transition over a step $\Delta t$ is Gaussian with no discretization error:

$$x_{\ell} = a\, x_{\ell-1} + \sqrt{q}\,\xi, \qquad a = e^{-\theta \Delta t}, \quad q = \tfrac{\sigma^2}{2\theta}\left(1 - e^{-2\theta \Delta t}\right), \quad \xi \sim \mathcal N(0,1).$$

- **Exact target statistics.** The stationary distribution is $\mathcal N(0, s^2)$ with $s^2 = \sigma^2/2\theta$, and the lag-1 autocorrelation is $a$. Generated trajectories must reproduce both numbers.
- **Exact filter.** The model is linear-Gaussian, so the optimal estimator under measurement noise is the two-line scalar Kalman filter.
- **Exact denoiser and loss floor.** OU trajectories are *jointly Gaussian*, which makes the optimal noise predictor.

**Two clocks.** Throughout, physical time $\ell$ (indexing the trajectory) and diffusion time $k$ (indexing the noising/denoising ladder) are distinct axes. The generative model treats each trajectory $x^{(0)} \in \mathbb R^L$ as a fixed data point and never sees the physical SDE again, the structural echo between the OU transition and the DDPM forward step (both are "shrink and add Gaussian noise") lives on *different clocks*. Keeping the two clocks separate is the conceptual backbone of the planned autoregressive extension, where they form a frames × noise-levels lattice.

## 2. The pipeline

| Stage | Math | Module |
|---|---|---|
| 1. Data | exact AR(1) transition above; $x^{(0)} \in \mathbb R^{L}$, $N$ trajectories | `ou_diffusion/ou_process.py` |
| 2. Forward diffusion | $x^{(k)} = \sqrt{\bar\alpha_k}\, x^{(0)} + \sqrt{1-\bar\alpha_k}\,\epsilon$, linear $\beta$ schedule | `ou_diffusion/schedule.py` |
| 3. Filtering | scalar Kalman predict/update on $y_\ell = x_\ell + v_\ell$ | `ou_diffusion/kalman.py` |
| 4. Reverse process | 1D U-Net $\epsilon_\theta(x^{(k)}, k)$, MSE objective, ancestral sampling | `ou_diffusion/unet1d.py`, `train.py`, `sample.py` |
| Validation | marginal variance vs $s^2$, lag-1 autocorrelation vs $a$, loss vs floor | `ou_diffusion/validate.py` |

The U-Net is the note's 3-resolution design: $1\times L \to C\times L \to 2C\times\frac L2 \to 4C\times\frac L4$ and symmetrically back up with skip connections, the diffusion step injected into every residual block through a sinusoidal embedding.

**Filtering is a separate task from diffusion**, and the repo keeps them separate on purpose: diffusion noise is *designed*, added on the diffusion clock, and removed by the learned reverse process; measurement noise is *given*, lives on the physical clock, and is removed by the Kalman filter. For this linear-Gaussian signal both removals are Bayesian posterior computations, which is exactly why OU is the right place to first see their relationship.

## 3. Verification
**The training loss has a known floor.** Because $x^{(0)} \sim \mathcal N(0, \Sigma)$ with $\Sigma_{ij} = s^2 a^{|i-j|}$, the posterior covariance of $\epsilon$ given $x^{(k)}$ is $I - (1-\bar\alpha_k)\Sigma_k^{-1}$ with $\Sigma_k = \bar\alpha_k \Sigma + (1-\bar\alpha_k) I$, so the irreducible per-coordinate $\epsilon$-prediction MSE, averaged over uniformly sampled $k$, is

$$\mathcal L^\star = \frac1K \sum_k \Big[ 1 - (1-\bar\alpha_k)\, \tfrac1L\, \mathrm{tr}\,\Sigma_k^{-1} \Big] = \mathbf{0.1071} \quad (\theta{=}1, \sigma{=}1, \Delta t{=}0.05, L{=}64, K{=}1000).$$

A network whose loss plateaus here *is* the optimal denoiser; more training or capacity cannot help. (`validate.irreducible_eps_loss`)

**Output statistics against exact targets.** Marginal variance vs $s^2 = 0.5$; lag-1 autocorrelation vs $a = e^{-0.05} \approx 0.9512$.

### Results ($K = 1000$, $6{,}000$ training steps, $N = 10{,}000$, $L = 64$)

| | marginal variance | lag-1 autocorrelation |
|---|---|---|
| exact OU target | 0.5000 | 0.9512 |
| analytic oracle $\epsilon^\star$ through the sampler | 0.491 | 0.9500 |
| **trained 1D U-Net** | **0.4981** | **0.9494** |

Training loss plateaus at **0.10–0.12**, oscillating around the **0.1071** floor (the per-step fluctuation is minibatch noise: the per-$k$ floor spans $\approx 1$ at $k{=}0$ down to $\approx 0$ at $k{=}K$, so batch composition moves single-batch estimates, readings slightly *below* the floor are high-$k$-heavy batches, not super-optimal learning).

**Error attribution.** The network sits at the information-theoretic loss floor *and* matches the oracle's output statistics, so the residual $\sim 2\%$ variance gap to the exact target is **sampler discretization bias** (finite $K$, fixed-$\beta$ reverse variance, residual terminal signal), not learning error.
## 4. A measured finding: terminal SNR and the choice of $K$

The note suggests $K = 200$ diffusion steps works fine. However the linear $\beta \in [10^{-4}, 0.02]$ schedule only reaches $\bar\alpha_K \approx 0.132$ at $K = 200$: about $36\%$ of the signal *amplitude* survives the forward process, so initializing the reverse process at pure $\mathcal N(0, I)$ is mis-specified. The oracle makes this exactly measurable:

| $K$ | $\bar\alpha_K$ | oracle marginal variance (target 0.5) |
|---|---|---|
| 200 | 0.132 | 0.365 (−27%) |
| 1000 | $4\times10^{-5}$ | 0.491 (−1.8%) |

This is the known zero-terminal-SNR issue (Lin et al., 2024) reproduced in a setting where it can be quantified *exactly*, with the learning component ruled out. Remedies, in increasing order of effort: use $K = 1000$ (default in the commands below), we could also try switching to a cosine scheduler.

## 5. Quickstart

```bash
pip install -r requirements.txt              # numpy, torch

# Fast end-to-end sanity check (tiny config):
python -m scripts.run_ou --smoke

# Full run (about 25 min on 1 CPU core; minutes on GPU with --device cuda):
python -m scripts.run_ou --K 1000 --steps 6000 --save-samples samples.npy

# Optional: Kalman filtering demo on noisy measurements
python -m scripts.run_ou --K 1000 --steps 6000 --kalman-demo --R 0.5
```

## 6. Repository layout

```
ou_diffusion/
  ou_process.py     Stage 1: exact OU sampler (Algorithm 1)
  schedule.py       Stage 2: DDPM schedule + q_sample (Eq. 6-8)
  kalman.py         Stage 3: scalar Kalman filter (Algorithm 3)
  unet1d.py         Stage 4: 1D U-Net eps-predictor
  train.py          Stage 4: training loop (Algorithm 4) + EMA
  sample.py         Stage 4: ancestral sampler (Algorithm 5)
  validate.py       statistics vs targets
scripts/
  run_ou.py                   end-to-end pipeline
```dules and Sample Steps are Flawed.* WACV 2024 — zero terminal SNR.
- Särkkä. *Bayesian Filtering and Smoothing.* CUP 2013 — Kalman filtering background.
- Chen et al. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion.* NeurIPS 2024; Ruhe et al. *Rolling Diffusion Models.* ICML 2024 — Phase-B directions.
