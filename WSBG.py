r"""WSBG.py -- wavelet score-based model of the transition kernel p_Delta(y | x).

Generative variable
-------------------
r = S(y - m(x)),  m = Phi_Delta(x) (noiseless flow map), S the discrete
stream-function map (STREAM.py), in WAVELET coordinates (n^2 numbers).
y = m + curl_h(r) exactly.  r lies on the codimension-2n subspace Ran; the
model is trained on its exact Gaussian dequantisation z = r + B_w c
(STREAM.make_kernel_dequantizer), which has a density on R^{n^2} and the
same image under curl_h.  NOTHING in this module projects: the kernel part
of a generated sample is simply annihilated by curl_h (see TLM).

Why not velocity coordinates.  r_vel lives in the linear subspace V.
(a) The Gaussian surrogate N(0, diag Sigma_bar) is not supported on V;
(b) the forward drift e^{-lam_ou tau} with level-dependent rates does not map
V to V (P does not commute with it); (c) projecting the denoiser improves the
MEAN in L2 but the ancestral noise is ambient and the final projection
discards its normal part (measured at J = 5 with F = 0: the projected
Gaussian sampler keeps 0.54 of the target energy).

Forward (noising) process, independently per coefficient i of level j(i):

    dR = -lam_j R dtau + sqrt(2 lam_j) sigma_i dB,     R_0 = z,
    R_tau | z ~ N(a_i z, s_i),   a_i = e^{-lam_j tau},
                                 s_i = sigma_i^2 (1 - a_i^2),

lam_j = lambda0 2^{beta j} (OU rates, ``lambda_diag``): finer scales are noised
faster, so that at any tau the SNR ordering across levels follows the
physical ordering of scales.

sigma_i^2 := E[z_i^2] (z the dequantised residual), ORBIT-pooled over the
training pairs (unbiased for the stationary law, and invariant under the
dyadic-shift augmentation): the FORWARD-PROCESS SCALING.  Not to be confused
with Sigma_bar_i(x), the variance of the Gaussian SURROGATE (its
calibration is TLM / GROUP); kappa_i = Sigma_bar_i / sigma_i^2.  sigma^2
only shapes the forward process.  At fixed (a, s) the minimiser
F* = E[F_tgt | R, x] does not involve sigma^2, but the F-loss VALUE does,
through beta^2 = s / Sg and the tau range (DIFF, "F = 0 loss").  Hence
F-losses are comparable only on a shared forward process, and across rungs
GRID compares xmse, not the F-loss.  Rungs of GRID share one forward process
per lag.

Reverse process
---------------
Exact Gaussian (ancestral) transitions of the OU bridge, driven by the
denoiser  rhat = E[r | R_tau, x]  (TLM.denoise):

        R_s | R_t ~ N( (a_ts s_s R_t + a_s s_ts rhat) / s_t ,
                   eta^2 [ s_s s_ts / s_t + (a_s s_ts / s_t)^2 Var(r | R_t) ] ),
    a_ts = a_t / a_s,  s_ts = s_t - a_ts^2 s_s.

The second variance term (law of total variance) is dropped by plain DDPM,
which is exact only as n_steps -> infinity.  Here Var(r | R_t) is taken from
the surrogate, Sigma_bar s_t / Sg: the sampler is then EXACT for the
surrogate at any number of steps (a Gaussian chain whose every conditional
is the true one), and for the true law the approximation is at the level of
the second moment of the denoising error only.

Surrogate.  Sigma_bar(x) = Sigma_bar_z * g(x) (TLM module docstring:
g = 1 is exact for linear data; the TLM multipliers are an approximation in
the kernel-coupled coefficients, never a fixed point).

Prior.  R_T ~ N(0, a_T^2 Sigma_bar + s_T): the surrogate marginal at T.  Its
KL to the true marginal is O(max_i SNR_i(T)) = O(snr_lo).
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from OPERATORS import WaveletOperators1D, LerayProjector
from FORCING import build_level_map_2d, build_orbit_map_2d
from TLM import (ou_coefficients, baseline_coefficients, whiten,
                       denoise, conditional_log_snr)
from STREAM import StreamMap


# 2.  Diffusion model: schedule and calibration

class WaveletDiffusionModel:
    """Scale-adaptive OU diffusion on r = S(y - Phi_Delta(x))."""

    def __init__(self, ops_1d: WaveletOperators1D, projector: LerayProjector,
                 beta: float = 1.5, lambda0: float = 1.0):
        """``projector`` is the pipeline's: its Nyquist filter defines the
        velocity space in which r = S(y - m) is taken."""
        if not isinstance(projector, LerayProjector):
            raise TypeError("pass the pipeline's LerayProjector")
        self.ops_1d = ops_1d
        self.J, self.jmin = int(ops_1d.J), int(ops_1d.jmin)
        self.W_fwd, self.W_inv = ops_1d.W_fwd, ops_1d.W_inv
        self.stream = StreamMap(ops_1d, projector.filter_nyquist, dtype=jnp.float32)
        self._to_psi = jax.jit(self.stream.psi_w_from_state)
        self._from_psi = jax.jit(self.stream.state_from_psi_w)
        self.n = 2 ** self.J
        self.n_psi = self.n * self.n                 # generative dimension
        self.n_state = 2 * self.n * self.n              # velocity dimension
        self.beta, self.lambda0 = float(beta), float(lambda0)

        self.levels = list(range(self.jmin - 1, self.J))
        self.level_per_coeff = build_level_map_2d(self.J, self.jmin, n_comp=1)
        self.level_per_coeff_state = build_level_map_2d(self.J, self.jmin, n_comp=2)
        self.level_index_per_coeff = self.level_per_coeff - (self.jmin - 1)
        self.orbit_per_coeff = build_orbit_map_2d(self.J, self.jmin, n_comp=1)
        self.n_orbits = int(self.orbit_per_coeff.max()) + 1
        self.lambda_per_level = self.lambda0 * 2.0 ** (self.beta * np.asarray(self.levels, float))
        self.lambda_diag = self.lambda_per_level[self.level_index_per_coeff]
        L = len(self.levels)
        self.level_masks = np.stack([self.level_index_per_coeff == l
                                     for l in range(L)]).astype(np.float32)

        self.sigma_sq_diag: Optional[np.ndarray] = None   # E[z^2] (dequantised), orbit-pooled
        self.m_scale_diag: Optional[np.ndarray] = None    # sqrt E[(S m)^2]
        self.x_scale_diag: Optional[np.ndarray] = None    # sqrt E[(S x)^2]
        self.t_min: Optional[float] = None
        self.T_diff: Optional[float] = None

    def to_psi(self, x):
        """velocity wavelet state(s) -> stream-function wavelet coefficients."""
        return self._to_psi(jnp.asarray(x, jnp.float32))

    def from_psi(self, p):
        return self._from_psi(jnp.asarray(p, jnp.float32))

    
    def calibrate(self, dataset, idx_pairs=None, chunk=256, verbose=True,
                  dequant=None, g=None, seed=0):
        """Forward-process scaling: sigma_i^2 = E[z_i^2], sqrt E[(S m)_i^2],
        sqrt E[(S x)_i^2], orbit-pooled.

        z is the dequantised residual (pass ``dequant`` =
        STREAM.make_kernel_dequantizer, the one used in training, and the
        surrogate multiplier ``g``; default dataset['rho']).  Uses only the
        TRAINING pairs (pass idx_pairs), streamed from memmaps.
        """
        g_all = np.asarray(dataset["rho"] if g is None else g, np.float32)
        dkey = jax.random.PRNGKey(seed)
        X, Mm, pi = dataset["x"], dataset["m"], np.asarray(dataset["pair_index"])
        idx = np.arange(len(pi)) if idx_pairs is None else np.sort(np.asarray(idx_pairs))
        o = self.orbit_per_coeff
        Sr = np.zeros(self.n_orbits)
        Sm = np.zeros(self.n_orbits)
        Sx = np.zeros(self.n_orbits)
        cnt = 0
        for s in range(0, len(idx), chunk):
            b = idx[s:s + chunk]
            y = np.asarray(X[pi[b, 1]], np.float32)
            x = np.asarray(X[pi[b, 0]], np.float32)
            m = np.asarray(Mm[b], np.float32)
            r = self.to_psi(y - m)
            if dequant is not None:
                dkey, sk = jax.random.split(dkey)
                r = dequant(r, jnp.asarray(g_all[b]), sk)
            r = np.asarray(r, np.float64)
            pm = np.asarray(self.to_psi(m), np.float64)
            px = np.asarray(self.to_psi(x), np.float64)
            Sr += np.bincount(o, weights=(r * r).sum(0), minlength=self.n_orbits)
            Sm += np.bincount(o, weights=(pm * pm).sum(0), minlength=self.n_orbits)
            Sx += np.bincount(o, weights=(px * px).sum(0), minlength=self.n_orbits)
            cnt += len(b)
        size = np.bincount(o, minlength=self.n_orbits) * cnt
        s2, m2, x2 = Sr / size, Sm / size, Sx / size
        if not np.all(np.isfinite(s2)) or s2.max() <= 0:
            raise RuntimeError("residual second moments are zero or non-finite")
        # Guard only: with the dequantiser every orbit has positive variance.
        s2 = np.maximum(s2, 1e-12 * s2.max())
        m2 = np.maximum(m2, 1e-12 * m2.max())
        x2 = np.maximum(x2, 1e-12 * x2.max())
        self.sigma_sq_diag = s2[o]
        self.m_scale_diag = np.sqrt(m2[o])
        self.x_scale_diag = np.sqrt(x2[o])
        if verbose:
            lv = {j: float(np.mean(self.sigma_sq_diag[self.level_per_coeff == j]))
                  for j in self.levels}
            print("   [WaveletDiffusionModel] E[z_psi^2] per level: "
                  + "  ".join(f"j={j}:{v:.2e}" for j, v in lv.items()))

    
    def set_time_range(self, sigma_bar_z, g_lo=None, g_hi=None,
                       snr_hi=1e2, snr_lo=1e-3, verbose=True):
        r"""[t_min, T_diff] (= [tau_min, tau_max]) from the CONDITIONAL SNR.

        SNR_i(tau) = kappa_i g_i / expm1(2 lam_i tau),  kappa_i = Sigma_bar_z,i/sigma_i^2.
        t_min : every coefficient has SNR >= snr_hi (smallest kappa g binds);
        T_diff: every coefficient has SNR <= snr_lo (largest kappa g binds).
        g_lo / g_hi bound the surrogate multiplier over the data, orbit-wide
        (n_orb) or per coefficient (n^2); None = 1 (rung L1).
        Returns (t_min, T_diff).
        """
        if self.sigma_sq_diag is None:
            raise RuntimeError("calibrate() first")
        o = self.orbit_per_coeff

        def _per_coeff(v):
            if v is None:
                return np.ones(self.n_psi)
            v = np.asarray(v, float)
            return v if v.shape[-1] == self.n_psi else v[o]

        lo, hi = _per_coeff(g_lo), _per_coeff(g_hi)
        k = np.asarray(sigma_bar_z) / self.sigma_sq_diag
        lam = self.lambda_diag
        self.t_min = float(np.min(np.log1p(k * lo / snr_hi) / (2 * lam)))
        self.T_diff = float(np.max(np.log1p(k * hi / snr_lo) / (2 * lam)))
        if not (0 < self.t_min < self.T_diff):
            raise RuntimeError(f"degenerate diffusion time range "
                               f"[{self.t_min:.3e}, {self.T_diff:.3e}]")
        # The surrogate actually used is Sigma_bar_z * g(x): judge THAT one.
        # sqrt(lo hi) is the geometric centre of the data range of g.
        k_eff = k * np.sqrt(lo * hi)
        if verbose:
            print(f"   [WaveletDiffusionModel] kappa g = Sigma_bar(x)/E[z^2] in "
                  f"[{k_eff.min():.3f}, {k_eff.max():.3f}]"
                  + ("" if g_lo is None else f" (g = 1 part alone: [{k.min():.3f}, {k.max():.3f}])")
                  + f"; tau in [{self.t_min:.3e}, {self.T_diff:.3e}]")
        if k_eff.min() < 0.2 or k_eff.max() > 5.0:
            warnings.warn("Sigma_bar(x) and E[z^2] differ by more than 5x on some "
                          "orbits: the surrogate is a poor baseline there "
                          "(Delta too large, or the nonlinear mean shift dominates).")
        return self.t_min, self.T_diff

    def fingerprint(self):
        return {"sigma_sq_diag": np.asarray(self.sigma_sq_diag),
                "lambda_diag": np.asarray(self.lambda_diag),
                "m_scale_diag": np.asarray(self.m_scale_diag),
                "x_scale_diag": np.asarray(self.x_scale_diag)}



# 3.  Conditional ancestral sampler (batched)

def make_conditional_sampler(diffusion: WaveletDiffusionModel, apply_F: Callable,
                             sigma_bar_z: np.ndarray):
    """generate(params, key, m, x, g, ...) -> dict, BATCHED.

    m, x : velocity wavelet states, shape (2n^2,) or (B, 2n^2), m = Phi_Delta(x)
    g    : surrogate multiplier, width n_orb or n^2, shape (w,) or (B, w)
           (None = 1, rung L1)
    apply_F(params, Rc, cond_m, cond_x, logsnr_levels) -> F, all (B, n^2)
    except logsnr_levels (B, L); params=None uses F == 0 (pure Gaussian
    surrogate sampler, EXACT for the surrogate at any n_steps).

    One network evaluation per diffusion step for the whole batch.

    Options: n_steps; eta (1 = exact ancestral; 0 = posterior-mean path, a
    diagnostic, not a sampler); guidance g (scales the learned correction,
    TLM (**)); posterior_variance (law-of-total-variance term);
    F_clip (safety clip, fraction reported).

    Returns r (terminal R_0, stream-function coordinates) and
    y = m + curl_h(r), exactly in V_F; leading shape as the input m.
    """
    lam = jnp.asarray(diffusion.lambda_diag, jnp.float32)
    sig = jnp.asarray(diffusion.sigma_sq_diag, jnp.float32)
    mscale = jnp.asarray(diffusion.m_scale_diag, jnp.float32)
    xscale = jnp.asarray(diffusion.x_scale_diag, jnp.float32)
    sbar_z = jnp.asarray(sigma_bar_z, jnp.float32)
    orbit = jnp.asarray(diffusion.orbit_per_coeff)
    masks = jnp.asarray(diffusion.level_masks)
    counts = jnp.sum(masks, axis=1)
    n_psi = diffusion.n_psi
    to_psi = diffusion.stream.psi_w_from_state
    from_psi = diffusion.stream.state_from_psi_w

    @partial(jax.jit, static_argnames=("n_steps", "use_net"))
    def _impl(params, key, m, x, g, T, t_min, n_steps, eta, guidance, F_clip,
              post_var_factor, use_net):
        B = m.shape[0]
        sbar = sbar_z[None, :] * (g if g.shape[-1] == n_psi
                                  else jnp.take(g, orbit, axis=1))       # (B, n^2)
        ts = jnp.concatenate([jnp.exp(jnp.linspace(jnp.log(T), jnp.log(t_min),
                                                   n_steps)), jnp.zeros((1,))])
        k0, kscan = random.split(key)
        aT, sT = ou_coefficients(lam, sig, T)
        SgT, _, _ = baseline_coefficients(aT, sT, sbar)
        R0 = jnp.sqrt(SgT) * random.normal(k0, (B, n_psi))
        cm = to_psi(m) / mscale
        cx = to_psi(x) / xscale

        def step(R, inp):
            t, s, ki = inp
            a_t, s_t = ou_coefficients(lam, sig, t)
            a_s, s_s = ou_coefficients(lam, sig, s)
            a_ts = jnp.exp(-lam * (t - s))
            s_ts = sig * (-jnp.expm1(-2.0 * lam * (t - s)))
            Sg, _, _ = baseline_coefficients(a_t, s_t, sbar)
            if use_net:
                lsnr = (conditional_log_snr(a_t, s_t, sbar) @ masks.T) / counts
                F = apply_F(params, whiten(R, Sg), cm, cx, lsnr).astype(jnp.float32)
            else:
                F = jnp.zeros_like(R)
            clipped = jnp.mean((jnp.abs(F) > F_clip).astype(jnp.float32))
            F = guidance * jnp.clip(F, -F_clip, F_clip)
            r_hat = denoise(R, a_t, s_t, sbar, F)
            st = jnp.maximum(s_t, 1e-30)
            mean = (a_ts * s_s * R + a_s * s_ts * r_hat) / st
            g_ = a_s * s_ts / st
            var = (jnp.maximum(s_s * s_ts / st, 0.0)
                   + g_ * g_ * post_var_factor * sbar * s_t / Sg)
            Rn = mean + eta * jnp.sqrt(var) * random.normal(ki, R.shape)
            return Rn, clipped

        r0, clip = jax.lax.scan(step, R0, (ts[:-1], ts[1:],
                                           random.split(kscan, n_steps)))
        return {"y": m + from_psi(r0), "z": r0, "clip_frac": clip}

    def generate(params, key, m, x, g=None, n_steps=100, eta=1.0,
                 guidance=1.0, F_clip=np.inf, T=None, t_min=None,
                 posterior_variance=True):
        m = jnp.asarray(m, jnp.float32)
        single = m.ndim == 1
        m = jnp.atleast_2d(m)
        x = jnp.broadcast_to(jnp.asarray(x, jnp.float32), m.shape)

        if g is None:
            g = jnp.ones((m.shape[0], diffusion.n_orbits), jnp.float32)
        g = jnp.asarray(g, jnp.float32)
        if g.ndim == 1:
            g = g[None, :]
        g = jnp.broadcast_to(g, (m.shape[0], g.shape[-1]))

        out = _impl(params, key, m, x, g,
                    diffusion.T_diff if T is None else T,
                    diffusion.t_min if t_min is None else t_min,
                    int(n_steps), float(eta), float(guidance), float(F_clip),
                    1.0 if posterior_variance else 0.0, params is not None)
        if single:
            out = {"y": out["y"][0], "z": out["z"][0], "clip_frac": out["clip_frac"]}
        return out

    return generate