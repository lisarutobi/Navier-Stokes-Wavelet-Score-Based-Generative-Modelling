r"""TLM.py -- Gaussian baseline of the transition kernel.

Notation (shared with WSBG / DIFF):
    m(x) = Phi_Delta(x)            deterministic lag-Delta map
    S                               velocity -> psi-wavelet stream map (STREAM)
    Sigma_bar                       diagonal Gaussian surrogate variance
                                    (Sigma_bar_r of r, Sigma_bar_z of the
                                    dequantised z; see sigma_bar_linear)
    rho(x)                          x-dependent correction, Sigma_bar(x) = Sigma_bar rho(x)
    (a, s) = (a(tau), s(tau))       OU mean / variance coefficients of the noising
    eps ~ N(0, I)                   noising variable (NOT eps_inj)

Denoising identities:
    r = S(y - m(x)),   R_tau = a r + sqrt(s) eps,   Sg = a^2 Sigma_bar + s
    F_tgt = ( a sqrt(Sigma_bar) eps - sqrt(s) r / sqrt(Sigma_bar) ) / sqrt(Sg)
(**) E[r | R, x] = a Sigma_bar R / Sg - sqrt(s Sigma_bar / Sg) F*,  F* = E[F_tgt | R, x]
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from FORCING import orbit_pool, linear_symbol

from FORCING import orbit_pool, linear_symbol
from STREAM import curl_kernel_basis_w


RHO_FLOOR = 0.05
"""Lower bound on the surrogate multiplier g(x).  Applied ONCE, to the quantity
used as g (orbit mean of rho, or shrunk rho), never to the raw per-coefficient
estimate: the orbit mean of a floored field is biased upward."""



# 1.  Estimators of diag Sigma

def _linear_pieces(ops1d, projector, dt, nu, mu_d, dtype):
    """(modal, E):  modal(R, E) = e^{dt L} R  on a scaling-basis field."""
    z, modal = linear_symbol(ops1d, projector, dt, nu, mu_d, dtype=dtype)
    return modal, jnp.exp(z).astype(dtype)


def _linear_response_fn(ops1d, projector, stream, noise, dt, nu, mu_d, stride, dtype):
    """zeta (n_in,) -> (S, n^2):  psi_w of e^{k dt L} B_F zeta,  k = 0 .. S-1.

    Used ONLY where the whole propagator trajectory is needed (the 
    orbit columns of Sigma_lin, and the dense oracle).  lax.scan, not a
    Python loop: the unrolled version put `stride` copies of `modal` in the
    XLA graph.
    """
    modal, E = _linear_pieces(ops1d, projector, dt, nu, mu_d, dtype)

    def resp(zeta):
        U0, V0 = noise.from_normals(zeta)

        def body(c, _):
            U, V = c
            return (modal(U, E), modal(V, E)), stream.psi_w_from_velocity_s(U, V)

        _, out = jax.lax.scan(body, (U0, V0), None, length=int(stride))
        return out                                          # (S, n^2)
    return resp


def _linear_residual_fn(ops1d, projector, stream, noise, dt, nu, mu_d, stride, dtype):
    r"""keys (S, 2) -> r = sqrt(dt) sum_{k<S} e^{(S-1-k) dt L} w_k   (psi_w, n^2).

    The lag-Delta linear residual of ONE probe, by the Horner recursion

        a_0 = 0,  a_{k+1} = e^{dt L} a_k + w_k   ==>   a_S = sum_k e^{(S-1-k) dt L} w_k,

    i.e. O(1) fields in memory and O(S) work.  The previous implementation
    built the full (S, S, n^2) matrix of responses and kept only its
    anti-diagonal: at J = 7, stride = 100, chunk = 32 that is 21 GiB in
    float32 and it is what killed the run with RESOURCE_EXHAUSTED.
    psi_w_from_velocity_s is linear, so it is applied ONCE, at the end.
    """
    modal, E = _linear_pieces(ops1d, projector, dt, nu, mu_d, dtype)
    n_in = int(noise.n_in)
    n = int(stream.n)
    sdt = jnp.asarray(np.sqrt(dt), dtype)

    def resid(keys):                                        # keys (S, 2)
        def body(c, k):
            U, V = c
            dU, dV = noise.from_normals(random.normal(k, (n_in,), dtype))
            return (modal(U, E) + dU, modal(V, E) + dV), None

        zero = jnp.zeros((n, n), dtype)
        (U, V), _ = jax.lax.scan(body, (zero, zero), keys)
        return sdt * stream.psi_w_from_velocity_s(U, V)
    return resid


def sigma_bar_linear(ops1d, projector, stream, noise: Callable, dt: float,
                     nu: float, mu_d: float, stride: int, orbit: np.ndarray,
                     n_probe: int = 512, chunk: int = 32, seed: int = 0,
                     n_fixed_point: int = 8, dtype=jnp.float32,
                     verbose: bool = True):
    r"""(B0) surrogate variances  ->  (Sigma_bar_z, Sigma_bar_r), per coefficient.

    Sigma_bar_r : orbit-pooled diag of Sigma_lin = Cov(r) for the linear
                  (Stokes + drag) scheme, r = S(y - m) in Ran.  : one
                  column per velocity orbit (FORCING.orbit_representatives),
                  S columns of the propagator each.
    Sigma_bar_z : the variance the diffusion model actually sees, i.e. of
                  the DEQUANTISED variable z = r + B_w c, defined as the fixed
                  point  D = pool diag Cov(z_D),  z_D = deq_D(r).
                  Only D = Sigma_bar_z makes the proposition "z ~ N(0, D) when
                  r is Gaussian" self-consistent.  Using Sigma_bar_r instead
                  (previous version) gives kappa != 1 by construction
                  (min 0.942 at J=6) and makes the F = 0 sampler UNDER-disperse
                  the physical residual curl_h(z) by up to 6 % (median 3 %).
                  Computed as  Sigma_bar_r ()  +  pool(E z^2 - E r^2)
                  (Monte Carlo with common random numbers: a control variate,
                  only the O(5 %) kernel correction carries MC error).

    ``noise`` must be the FORCING.make_forced_stepper noise map (it carries
    ``from_normals``, ``n_in`` and ``orbit_in``), ``stream`` the StreamMap.

    For 'stream' forcing the pipeline uses GROUP.sigma_bar_linear_
    instead: same Sigma_bar_r, and Sigma_bar_z as the  fixed point
    (this Monte-Carlo version is off by ~1e-2 at 256 probes).  This function
    is kept for the 'velocity' forcing only.
    """
    from STREAM import make_kernel_dequantizer
    from FORCING import orbit_representatives
    orbit = np.asarray(orbit)
    n_orb = int(orbit.max()) + 1
    n2 = stream.n2
    n_in = int(noise.n_in)
    in_orbit = np.asarray(noise.orbit_in)
    resp = jax.jit(_linear_response_fn(ops1d, projector, stream, noise, dt, nu,
                                       mu_d, stride, dtype))

    # ----  orbit-pooled diag of Sigma_lin (r variable) -----------------
    acc = np.zeros(n_orb)
    for k in orbit_representatives(in_orbit):
        e = np.zeros(n_in)
        e[k] = 1.0
        R = np.asarray(resp(jnp.asarray(e, dtype)), np.float64)      # (S, n2)
        acc += dt * np.bincount(orbit, weights=(R * R).sum(0), minlength=n_orb)
    sigma_bar_r = acc[orbit]          # no 1/|orbit|: see the lemma in orbit_representatives

    # ---- dequantisation fixed point (control variate) ----------------------
    probe_r = jax.jit(jax.vmap(_linear_residual_fn(
        ops1d, projector, stream, noise, dt, nu, mu_d, stride, dtype)))

    keys = random.split(random.PRNGKey(seed), n_probe * stride).reshape(n_probe, stride, -1)
    rs = [probe_r(keys[i:i + chunk]) for i in range(0, n_probe, chunk)]
    dkeys = random.split(random.PRNGKey(seed + 1), len(rs))
    mr2 = sum(np.asarray(jnp.sum(r * r, 0), np.float64) for r in rs) / n_probe

    # Free end-to-end check of the ness lemma: the MC second moment of r
    # and the  orbit columns estimate the SAME diagonal.
    pr, ps = (orbit_pool(v[None, :], orbit).ravel() for v in (mr2, sigma_bar_r))
    lem = float(np.max(np.abs(pr / ps - 1.0)))
    lem_tol = 6.0 * np.sqrt(2.0 / (n_probe * np.bincount(orbit).min()))
    if lem > lem_tol:
        raise RuntimeError(f"orbit-column Sigma_lin disagrees with the Monte-Carlo "
                           f"estimate by {lem:.2e} (tol {lem_tol:.2e}): the ness "
                           f"lemma, from_normals or the propagator is wrong")
    sigma_bar_z = sigma_bar_r.copy()
    it, change = -1, float("nan")
    for it in range(int(n_fixed_point)):
        deq = make_kernel_dequantizer(stream, sigma_bar_z, orbit, dtype=dtype)
        mz2 = sum(np.asarray(jnp.sum(deq(r, jnp.ones((r.shape[0], n_orb), dtype), dk) ** 2, 0),
                             np.float64) for r, dk in zip(rs, dkeys)) / n_probe
        corr = orbit_pool((mz2 - mr2)[None, :], orbit).ravel()[orbit]
        sigma_bar_z_new = sigma_bar_r + corr
        if not np.all(np.isfinite(sigma_bar_z_new)) or np.any(sigma_bar_z_new <= 0):
            raise FloatingPointError("dequantised fixed point left the positive cone")
        change = float(np.max(np.abs(sigma_bar_z_new / sigma_bar_z - 1.0)))
        sigma_bar_z = sigma_bar_z_new
        if change < 1e-4:
            break
    if verbose:
        k = sigma_bar_r / sigma_bar_z
        print(f"   [MALLIAVIN] Sigma_lin  ({len(orbit_representatives(in_orbit))} "
              f"orbit columns); range [{sigma_bar_r.min():.2e}, {sigma_bar_r.max():.2e}]; "
              f"dequantised fixed point after {it + 1} it. (last change {change:.1e}): "
              f"Sigma_bar_r / Sigma_bar_z in [{k.min():.3f}, {k.max():.3f}]")
    return sigma_bar_z, sigma_bar_r


def linear_gaussian_oracle(ops1d, projector, stream, noise: Callable, dt: float,
                           nu: float, mu_d: float, stride: int,
                           diffusion, sigma_bar_z: np.ndarray,
                           taus=None, n_tau: int = 48, dequant_basis=None,
                           chunk: int = 32, dtype=jnp.float64,
                           weighting: str = "unit", verbose: bool = True) -> dict:
    r"""Best achievable F-loss if p(r | x) were  the linear Gaussian N(0, Sigma_lin).

    The surrogate q = N(0, diag Sigma_bar) ignores the off-diagonal part of
    Sigma_lin, so even for  Gaussian (linear) data F* = E[F_tgt | R]
    is nonzero and LINEAR in R.  For whitened u = z / sqrt(Sigma_bar),
    C = Cov(u), alpha = a sqrt(Sigma_bar/Sg), beta = sqrt(s/Sg):

        F_tgt = alpha eps - beta u,       Rc = alpha u + beta eps,
        MSE*  = diag[Cov F] - diag[Cov(F,Rc) Cov(Rc)^{-1} Cov(Rc,F)].

    Returns per level the oracle MSE at the tau where SNR_j = 1 (same tau as
    DIFF.diagnose_levels) and the oracle value of the full training loss
    (tau ~ logU[t_min, T], 'unit' weighting).  A trained network is learning
    NON-Gaussian / nonlinear structure only to the extent that it beats
    these numbers.  Dense: n^2 x n^2 (J <= 6).  For 'stream' forcing use
    GROUP.oracle_from_diffusion: identical to 1e-15 (per level included),
     at any J.  Measured gains are NOT small: 0.032 (J=5 smoke),
    0.060 (J=7 production), mostly in the finest level.
    """
    n2 = stream.n2
    if n2 > 4096:
        raise ValueError("linear_gaussian_oracle is dense: use J <= 6")
    resp = jax.jit(jax.vmap(_linear_response_fn(ops1d, projector, stream, noise,
                                                dt, nu, mu_d, stride, dtype)))
    Sig = np.zeros((n2, n2))
    I = np.eye(int(noise.n_in))
    for i in range(0, int(noise.n_in), chunk):
        C = np.asarray(resp(jnp.asarray(I[i:i + chunk], dtype)), np.float64)
        for k in range(C.shape[1]):
            Sig += dt * C[:, k, :].T @ C[:, k, :]
    # covariance of the dequantised variable
    B = curl_kernel_basis_w(stream) if dequant_basis is None else dequant_basis
    D = np.asarray(sigma_bar_z, float)
    A = B.T @ (B / D[:, None])
    T = np.eye(n2) - B @ np.linalg.solve(A, B.T / D[None, :])
    Sz = T @ Sig @ T.T + B @ np.linalg.solve(A, B.T)
    # Sz is the  covariance of the dequantised linear residual; its
    # orbit-pooled diagonal equals D only at the fixed point of
    # sigma_bar_linear (MC control variate): report the mismatch.
    from FORCING import orbit_pool
    orb = diffusion.orbit_per_coeff
    dz = orbit_pool(np.diag(Sz)[None, :], orb).ravel()[orb]
    fp_err = float(np.max(np.abs(dz / D - 1.0)))
    Cu = Sz / np.sqrt(np.outer(D, D))
    sig2 = np.asarray(diffusion.sigma_sq_diag, float)
    lam = np.asarray(diffusion.lambda_diag, float)
    lev = diffusion.level_index_per_coeff

    def mse_at(tau):
        a = np.exp(-lam * tau)
        s = sig2 * (-np.expm1(-2.0 * lam * tau))
        Sg = a * a * D + s
        al, be = a * np.sqrt(D / Sg), np.sqrt(s / Sg)
        CFR = np.diag(al * be) - be[:, None] * Cu * al[None, :]
        CR = al[:, None] * Cu * al[None, :] + np.diag(be ** 2)
        K = np.linalg.solve(CR, CFR.T).T
        base = al ** 2 + be ** 2 * np.diag(Cu)
        return base, base - np.einsum("ij,ij->i", K, CFR)

    kap = D / sig2
    out = {"levels": {}}
    for li, j in enumerate(diffusion.levels):
        m = lev == li
        tau = np.log1p(kap[m].mean()) / (2.0 * diffusion.lambda_per_level[li])
        base, mse = mse_at(tau)
        out["levels"][int(j)] = (float(base[m].mean()), float(mse[m].mean()))
    if taus is None:
        taus = np.exp(np.linspace(np.log(diffusion.t_min), np.log(diffusion.T_diff), n_tau))
    L = len(diffusion.levels)
    losses, bases = [], []
    for tau in taus:
        base, mse = mse_at(tau)
        losses.append(np.mean([mse[lev == li].mean() for li in range(L)]))
        bases.append(np.mean([base[lev == li].mean() for li in range(L)]))
    out["loss"] = float(np.mean(losses))
    out["base_loss"] = float(np.mean(bases))
    out["gain"] = out["base_loss"] - out["loss"]
    out["fixed_point_err"] = fp_err
    offd = Sz / np.sqrt(np.outer(np.diag(Sz), np.diag(Sz))) - np.eye(n2)
    out["max_offdiag_corr"] = float(np.max(np.abs(offd)))
    if verbose:
        print(f"   [oracle] linear-Gaussian expected training loss = {out['loss']:.5f} "
              f"(F = 0: {out['base_loss']:.5f}; gain {out['gain']:.2e}); max |corr| off-diagonal = {out['max_offdiag_corr']:.3f}; "
              f"dequantised fixed point  to {fp_err:.1e}")

    if weighting != "unit":
        raise ValueError(
            "linear_gaussian_oracle returns the UNWEIGHTED (v-prediction) F-loss; "
            f"it is not comparable with weighting={weighting!r}.  Either train with "
            "'unit' or weight mse_at() by min(SNR, gamma) / (1 + SNR).")
    return out


def _make_tlm_diag(det_step, noise, stream, dt, stride, n_probe, seed, dtype):
    r"""(U_s, V_s) -> diag Sigma_TLM(x) in psi-wavelet coordinates, (n^2,).

    Linear-noise approximation along the NOISELESS path x_{k+1} = Phi_dt(x_k):
        delta_0 = 0,  delta_{k+1} = D Phi_dt(x_k) delta_k + sqrt(dt) w_k,
        Sigma_TLM(x) = Cov(S delta_stride).
    n_probe FIXED noise paths (common random numbers): a deterministic function
    of x, identical in the dataset and in the emulator.  Each entry is
    Sigma_ii * chi^2_{n_probe} / n_probe (S delta is Gaussian).
    det_step / noise must be those of FORCING.make_forced_stepper.
    """
    sdt = jnp.asarray(np.sqrt(dt), dtype)
    keys = random.split(random.PRNGKey(seed), n_probe * stride).reshape(n_probe, stride, -1)

    def tlm_diag(U, V):
        zero = (jnp.zeros_like(U), jnp.zeros_like(V))

        def one(kp):
            def body(carry, k):
                x, d = carry
                x1, d1 = jax.jvp(det_step, (x,), (d,))
                w = noise(k)
                return (x1, (d1[0] + sdt * w[0], d1[1] + sdt * w[1])), None
            (_, dS), _ = jax.lax.scan(body, ((U, V), zero), kp)
            return stream.psi_w_from_velocity_s(*dS) ** 2
        return jnp.mean(jax.vmap(one)(keys), axis=0)
    return tlm_diag


def make_sigma_diag_fn(det_step, noise, stream, sigma_bar_r, dt, stride,
                       n_probe=16, seed=12345, dtype=jnp.float32):
    """(U_s, V_s) -> rho_full (n^2,) = diag Sigma_TLM(x) / Sigma_bar_r, RAW
    (not pooled, not floored).  Rung L2 uses it only through shrink_rho_full."""
    tlm_diag = _make_tlm_diag(det_step, noise, stream, dt, stride, n_probe, seed, dtype)
    denom = jnp.asarray(np.asarray(sigma_bar_r, float), dtype)
    return lambda U, V: tlm_diag(U, V) / denom


def make_sigma_ratio_fn(det_step, noise, stream, sigma_bar_r, orbit, dt, stride,
                        n_probe=16, seed=12345, dtype=jnp.float32):
    """(U_s, V_s) -> rho (n_orb,) = max(orbit_mean(rho_full), RHO_FLOOR).

    Sigma_bar_r is orbit-constant, so orbit_mean(rho_full) = sum_O Sigma_TLM,ii
    / sum_O Sigma_bar_r,i.  Same probes as make_sigma_diag_fn: equal, to
    roundoff, to the 'rho' TG stores in 'TLM_FULL' mode (rung L1.5)."""
    orbit = np.asarray(orbit)
    n_orb = int(orbit.max()) + 1
    orb = jnp.asarray(orbit)
    cnt = jnp.asarray(np.bincount(orbit, minlength=n_orb), dtype)
    rho_full = make_sigma_diag_fn(det_step, noise, stream, sigma_bar_r, dt, stride,
                                  n_probe=n_probe, seed=seed, dtype=dtype)

    def ratio(U, V):
        pooled = jax.ops.segment_sum(rho_full(U, V), orb, num_segments=n_orb) / cnt
        return jnp.maximum(pooled, RHO_FLOOR)
    return ratio


def shrink_rho_full(rho_full: np.ndarray, orbit: np.ndarray, n_probe: int,
                    floor: float = RHO_FLOOR):
    r"""Empirical-Bayes shrinkage of the RAW per-coefficient ratio to its orbit mean.

    rho_hat_i = rho_i chi^2_n / n  (n = n_probe), hence
        E[rho_hat_i^2] = rho_i^2 (1 + 2/n),
        Var(rho_hat_i) = 2 rho_i^2 / n = 2 E[rho_hat_i^2] / (n + 2).
    Within an orbit:  Var_within = signal + noise,  and the minimum-MSE linear
    estimator with the orbit as prior is
        rho_i = rho_bar_O + w_O (rho_hat_i - rho_bar_O),
        w_O   = max(0, Var_within,O - noise_O) / Var_within,O.
    The probe paths are shared by all coefficients and Cov(X^2, Y^2) =
    2 Cov(X, Y)^2 >= 0, so the probe noise is non-negatively correlated inside
    an orbit.  The within-orbit noise is therefore AT MOST noise_O, and the
    shrinkage is conservative (w biased toward 0).

    Returns (rho_shrunk floored at ``floor``, w_per_orbit, snr_within).
    snr_within = mean_O signal_O / noise_O is the validity flag of the L2 row:
    below ~1, w -> 0 and L2 degenerates to L1.5 by construction.
    """
    R = np.asarray(rho_full, np.float64)
    orb = np.asarray(orbit)
    n_orb = int(orb.max()) + 1
    cnt = np.bincount(orb, minlength=n_orb).astype(float)

    def pool(A):
        return np.stack([np.bincount(orb, weights=r, minlength=n_orb) for r in A]) / cnt

    mean = pool(R)
    dev = R - mean[:, orb]
    within = (np.stack([np.bincount(orb, weights=d ** 2, minlength=n_orb)
                        for d in dev]) / np.maximum(cnt - 1.0, 1.0)).mean(0)
    noise = (2.0 / (float(n_probe) + 2.0)) * pool(R ** 2).mean(0)
    signal = np.maximum(within - noise, 0.0)
    w = signal / np.maximum(within, 1e-300)
    snr = float(np.mean(signal / np.maximum(noise, 1e-300)))
    out = mean[:, orb] + w[orb][None, :] * dev
    return np.maximum(out, floor).astype(np.float32), w, snr

def solver_conditional_ensemble(forced_step: Callable, stream, U, V,
                                stride: int, n_rep: int = 64, seed: int = 0,
                                chunk: int = 16) -> np.ndarray:
    """GROUND TRUTH for a single x: n_rep independent forced Delta-rollouts.

    Returns S y (n_rep, n^2) in stream-function wavelet coordinates.  m(x) is
    fixed, so Var(S y | x) = Var(r | x) and E[S y | x] - S m(x) is the
    conditional mean shift (O(Delta Tr Q), the leading non-Gaussian effect).
    """
    @jax.jit
    def run(keys):
        def one(key):
            def body(c, k):
                return forced_step(c, k), None
            (Uy, Vy), _ = jax.lax.scan(body, (U, V), random.split(key, stride))
            return stream.psi_w_from_velocity_s(Uy, Vy)
        return jax.vmap(one)(keys)

    keys = random.split(random.PRNGKey(seed), n_rep)
    return np.concatenate([np.asarray(run(keys[i:i + chunk]), np.float64)
                           for i in range(0, n_rep, chunk)])




# 2.  Gaussian baseline algebra (residual variable, all diagonal)

_TINY = 1e-30


def ou_coefficients(lam, sig2, tau):
    """a = exp(-lam tau), s = sig2 (1 - a^2), broadcast over (B, n)."""
    x = lam * tau
    return jnp.exp(-x), sig2 * (-jnp.expm1(-2.0 * x))


def baseline_coefficients(a, s, sbar):
    """Sg = a^2 Sigma_bar + s,  c_out = a sqrt(Sigma_bar/Sg),  c_skip = sqrt(s/Sg)."""
    Sg = jnp.maximum(a * a * sbar + s, _TINY)
    c_out = a * jnp.sqrt(sbar / Sg)
    c_skip = jnp.sqrt(s / Sg)
    return Sg, c_out, c_skip


def conditional_log_snr(a, s, sbar):
    """log(a^2 Sigma_bar / s): the SNR of what x does NOT determine."""
    return jnp.log(jnp.maximum(a * a * sbar, _TINY)) - jnp.log(jnp.maximum(s, _TINY))


def whiten(R, Sg):
    """Rc = R / sqrt(Sg): unit variance under the surrogate at every tau."""
    return R / jnp.sqrt(Sg)


def f_target(r, eps, a, s, sbar):
    """Unit-variance, Gaussian-orthogonal regression target (module docstring)."""
    Sg = jnp.maximum(a * a * sbar + s, _TINY)
    sb = jnp.sqrt(jnp.maximum(sbar, _TINY))
    return (a * sb * eps - jnp.sqrt(s) * r / sb) / jnp.sqrt(Sg)


def denoise(R, a, s, sbar, F):
    """E[r | R, x] = a Sigma_bar R / Sg - sqrt(s Sigma_bar / Sg) F   (no 1/a)."""
    Sg = jnp.maximum(a * a * sbar + s, _TINY)
    return a * sbar * R / Sg - jnp.sqrt(jnp.maximum(s * sbar, 0.0) / Sg) * F



# 3.  Tests with a known answer

def gaussian_sampler_test(sample_fn, n_samples: int, sbar: np.ndarray,
                          orbit: np.ndarray, seed: int = 0, batch: int = 64) -> dict:
    """With F_theta == 0 the sampler must draw r ~ N(0, Sigma_bar) .

    ``sample_fn(key, B) -> (B, n^2)`` must run the PRODUCTION sampler with zero
    network output.  Reports the orbit-pooled variance ratio, which must be 1
    within sqrt(2/(n_samples |orbit|)) (a LOWER bound on the MC error when the
    coefficients of an orbit are correlated, e.g. for the physical S curl_h z).
    """
    nb = -(-n_samples // batch)
    keys = random.split(random.PRNGKey(seed), nb)
    rs = np.concatenate([np.asarray(sample_fn(keys[i], min(batch, n_samples - i * batch)),
                                    np.float64) for i in range(nb)])
    got = orbit_pool(rs.var(0)[None, :], orbit).ravel()
    want = orbit_pool(np.asarray(sbar, float)[None, :], orbit).ravel()
    ratio = got / want
    size = np.bincount(orbit).min()
    tol = 4.0 * np.sqrt(2.0 / (n_samples * size))
    return {"max_abs_dev": float(np.max(np.abs(ratio - 1.0))),
            "tolerance": float(tol), "ratio": ratio,
            "ok": bool(np.max(np.abs(ratio - 1.0)) < tol)}