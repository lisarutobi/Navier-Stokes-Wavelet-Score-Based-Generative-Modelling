# Stochastic wavelet emulator of 2-D Navier–Stokes

This repo learns the one-step transition law $p_\Delta(y \mid x)$ of a stochastically forced, Ekman-damped 2-D incompressible Navier–Stokes solver, using a score-based diffusion model that works in wavelet coordinates.

The solver is a Bubnov wavelet-Galerkin scheme on the periodic unit square (CDF 9/7, `bior4.4`) with skew-symmetric advection, an exact discrete Leray projection and an integrating-factor RK4 time stepper. The forcing is white in time and diagonal in stream-function wavelet coordinates. Given a state $x$, the model generates $y = x(t+\Delta)$ as a sample, not a point prediction, and the goal is to get the conditional law right.

The goal is not speed. With the TLM surrogate turned on, every emulator step runs several tangent-linear rollouts and costs more than the solver step it replaces. What this code is for is studying the kernel: how much of it is Gaussian, how much depends on $x$, and what a network actually adds on top of a linear model.

This started as my M2 internship project (LMPS / ONERA) and is research code. Things move.

## The idea in a few lines

Write the transition as

$$y = \Phi_\Delta(x) + \mathrm{curl}_h(r),$$

where $\Phi_\Delta$ is the noiseless solver over one lag and $r$ is the residual in discrete stream-function wavelet coordinates. Working with $r$ rather than velocity removes the divergence constraint: every $r$ gives a divergence-free $y$, so nothing ever needs projecting after sampling.

The score of $r \mid x$ is split into two parts:

- a diagonal Gaussian surrogate $\mathcal N(0, \mathrm{diag}\,\bar\Sigma(x))$, with $\bar\Sigma$ built from the exact covariance of the linear (Stokes + drag) problem, optionally rescaled by a TLM variance ratio $g(x)$;
- a learned residual $F_\theta$ trained by denoising score matching on a unit-variance target that is exactly zero when the data match the surrogate.

The last layer of the network is zero-initialised, so training starts from the surrogate and anything it learns is, by construction, a departure from it.

A few details that matter more than they look:

- Dequantisation. $\mathrm{curl}_h$ has a $2n$-dimensional kernel (constant plus Nyquist lines), so the data $r$ live on a subspace. Inside the loss they are lifted to $z = r + B c$ with $c$ drawn from the exact Gaussian conditional. Without this, the optimal network output is nonzero even for perfectly Gaussian data, and the whole "F = 0 means Gaussian" reading breaks.
- Dyadic shift group. Shifts of the grid by multiples of $2^{J-j_{\min}}$ cells act on wavelet coefficients as exact permutations. Everything linear commutes with them, so covariances block-diagonalise under a small 2-D DFT (`GROUP.py`). That is what makes the linear-Gaussian oracle and the surrogate exact at $J = 7, 8$ with no Monte Carlo.
- Linear floor. A training gain on its own means little: even for linear dynamics the diagonal surrogate misses real off-diagonal correlation. Every trained model is therefore compared, sample by sample, against the best $x$-independent linear predictor fitted on the same data (`GROUP.empirical_linear_floor`).

## Repository layout

| File | What's in it |
|---|---|
| `OPERATORS.py` | Connection coefficients (two- and three-term), DWT matrices, 1-D operator bundle, Leray projector |
| `ODE.py` | Banded circulant operators, skew advection, the time stepper, energy/dissipation functionals, spatial validation |
| `TG.py` | Taylor–Green and random initial conditions, deterministic and stochastic TG benchmarks, transition-pair dataset generator |
| `FORCING.py` | Forcing covariance, noise map, forced stepper (exponential Euler–Maruyama), level/orbit maps, energy budget checks |
| `STREAM.py` | Discrete stream function, kernel of $\mathrm{curl}_h$ and its dequantiser |
| `TLM.py` | Gaussian baseline algebra, linear covariance, TLM variance ratio, solver conditional ensembles |
| `GROUP.py` | Block-diagonal algebra over the dyadic shift group, linear-Gaussian oracle, empirical linear floor |
| `WSBG.py` | Forward OU process, calibration, conditional ancestral sampler |
| `DIFF.py` | Periodic U-Net, residual score-matching loss, training loop |
| `CERT.py` | A-priori and a-posteriori checks: forcing regime, resolution, MC floors, fixed-$x$ tests, gain verdict |
| `BENCH.py` | Solver vs emulator ensembles, closed-form linear benchmark |
| `GRID.py` | Rung × lag grid (the main experiment) |
| `PLOT.py` | Plotting |
| `MAIN.py` | Pipeline, configuration and CLI |

## Requirements

Python 3 with `jax`, `flax`, `optax`, `numpy`, `scipy`, `PyWavelets` and `matplotlib`. A GPU is strongly recommended for anything above the smoke run.

## Running

Start with the smoke run. It goes through the whole pipeline at $J = 5$ in a reasonable time:

```bash
NSWSGM_X64=1 python MAIN.py --smoke
```

Set `NSWSGM_X64=1`. Operator validation, the Richardson order test and the stochastic Taylor–Green benchmark need float64; data generation, training and sampling still run in float32. Without it you get a warning and some checks are skipped or meaningless.

Other entry points:

```bash
python MAIN.py --only-validate          # operators + deterministic checks, then stop
python MAIN.py --bench-tg               # also run the closed-form linear benchmark
python MAIN.py --grid                   # also run the rung x lag grid
```

Most physical and run parameters can be overridden from the command line, e.g. `--J 7 --jmin 4 --stride 50 --sigma-mode TLM`. The full list is at the bottom of `MAIN.py`; everything else lives in the `config` dict.

The default configuration is the production one: $J = 8$, $\nu = 10^{-4}$, $\mu_d = 0.5$, $dt = 10^{-3}$, $\Delta = 0.1$, 128 trajectories × 64 pairs. That is a long run.

### What the pipeline does

1. Builds and validates the wavelet operators.
2. Checks the deterministic solver: Taylor–Green, advection consistency, skew symmetry, energy on rough fields, temporal order.
3. Sets up the forcing, checks the forced solver against the closed-form law on the Taylor–Green shell, computes the linear surrogate.
4. Generates the transition dataset and checks stationarity, the energy budget and resolution.
5. Calibrates the forward process and compares the surrogate with solver ensembles at fixed $x$.
6. Trains the residual model and scores it against the linear floor.
7. Fixed-$x$ certificates: Gaussian sampler, conditional variance, mean shift, one-lag energy split.
8. Rolls out solver and emulator ensembles from the same state and compares their statistics.
9. (optional) Closed-form benchmark on the linear system.
10. (optional) The rung × lag grid.

Two physics gates (forcing regime in stage 3, resolution in stage 4) stop the run when `strict_physics` is on. The smoke config turns them into warnings, because $J = 5$ cannot host both a drag-dominated forced range and a resolved enstrophy tail. The smoke run tests the plumbing, not the physics.

### Outputs

- `./data/` – datasets as memmaps plus a `_meta.npz`. Generation resumes where it stopped and regenerates automatically if any parameter changed.
- `./checkpoints/` – training checkpoints. They carry a fingerprint of the calibration, surrogate and architecture.
- `./figures/` – plots and `metrics.json` with every scalar result.

The smoke run writes to `*_smoke` directories so it never touches a production run.

## Reading the results

After training, the network is compared to the empirical linear floor with a paired, trajectory-clustered standard error (`CERT.gain_verdict`):

- `+` the network beats every $x$-independent linear map, so it has learned some $x$-dependent structure;
- `0` nothing beyond the linear floor;
- `-` worse than a closed-form linear map, so it is under-trained and nothing can be concluded;
- `?` fewer than 5 validation trajectories, no usable SE.

A `+` does not mean non-Gaussian. It could be a mean shift, an $x$-dependent covariance or genuine non-Gaussianity, and a loss cannot tell these apart. That question is decided at fixed $x$ against a solver ensemble (stage 7 and the grid's shift panel).

The grid exists to separate those cases. Rows are surrogate "rungs":

- L1 – $g = 1$, no $x$-dependence;
- L1.5 – orbit-pooled TLM ratio;
- L2 – per-coefficient TLM ratio, shrunk toward its orbit mean.

Columns are lags $\Delta$. A gain that disappears going from L1.5 to L2 was within-orbit variance structure that the surrogate could have supplied. A gain that survives L2 and grows with $\Delta$ is the interesting kind. Compare across rows with `xmse` (all rungs in a column share one forward process), not with the F-loss.

## Known limitations

- The closed-form benchmarks and the energy identity assume the Bubnov scheme (`test_space='primal'`). Petrov–Galerkin still runs, but advection is not exactly energy-neutral there.
- At the production lag ($\Delta = 0.1$, about 1.6 eddy times) the short-lag expansion behind the TLM surrogate is at its limit. Treat $\Sigma_{\mathrm{TLM}}(x)$ as a leading-order model, not a fixed point.
- Pairs from one trajectory are correlated (they are 0.2 time units apart, much less than a drag time), so all standard errors are clustered by trajectory. Keep enough validation trajectories.
- The TLM probes dominate the cost of data generation.
