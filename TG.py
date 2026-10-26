"""Initial conditions, solver benchmarks and the transition-pair dataset.

All calls go through the ODE API:
  * Bubnov-Galerkin operators (test_space='primal', M = G, PRODUCTION) or
    Petrov-Galerkin (test_space='dual', M = I, primal Gram = L2 metric);
  * skew-symmetric (Temam) advection;
  * level-J constraint space and  G-orthogonal Leray projection;
  * ONE time integrator: ODE.build_scaling_basis_stepper.  Every benchmark
    here validates that integrator, never a separate reference loop;
  * the dataset is generated with the FORCED stepper of FORCING.py.

Initial conditions are  Galerkin coefficients of the analytic field
(OPERATORS.samples_to_coeffs with the multiplier of the chosen test space), and
every error is measured against the  Galerkin coefficients of the
analytic solution, in the L2 (primal-Gram) norm.
"""

import warnings
import os
import time
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp


from OPERATORS import (WaveletOperators1D, LerayProjector, samples_to_coeffs,
                       galerkin_multiplier)
from ODE import (build_scaling_basis_stepper, scaling_kinetic_energy_fn,
                 _resolve_scheme, make_scaling_pieces, advective_spectral_radius,
                 _resolve_dtype, IMAG_AXIS_STABILITY, l2_inner_s)
from FORCING import phi_noise_multiplier
from jax import random



# TAYLOR-GREEN VORTEX

class TaylorGreenVortex:
    def __init__(self, k0: int, J: int, jmin: int = 0):
        self.k0 = k0
        self.J = J
        self.jmin = jmin
        self.n = 2 ** J
        self.dx = 1.0 / self.n
        self.x = np.linspace(0, 1, self.n, endpoint=False)
        self.y = np.linspace(0, 1, self.n, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        kx = ky = 2 * k0 * np.pi
        u_raw = np.sin(kx * self.X) * np.cos(ky * self.Y)
        v_raw = -np.cos(kx * self.X) * np.sin(ky * self.Y)
        self._amp_norm = np.sqrt(np.mean(u_raw ** 2 + v_raw ** 2)) or 1.0

    def velocity_field(self, phase_x: float = 0.0, phase_y: float = 0.0):
        kx = ky = 2 * self.k0 * np.pi
        u = np.sin(kx * self.X + phase_x) * np.cos(ky * self.Y + phase_y)
        v = -np.cos(kx * self.X + phase_x) * np.sin(ky * self.Y + phase_y)
        return u / self._amp_norm, v / self._amp_norm

    def to_scaling(self, phase_x: float = 0.0, phase_y: float = 0.0, *,
                   test_space: str) -> Tuple[np.ndarray, np.ndarray]:
        """ Galerkin coefficients in the SCALING basis."""
        u, v = self.velocity_field(phase_x, phase_y)
        A = galerkin_multiplier(self.n, test_space)
        return samples_to_coeffs(u, A), samples_to_coeffs(v, A)

    def to_wavelet(self, W_fwd: np.ndarray, phase_x: float = 0.0,
                   phase_y: float = 0.0, *, test_space: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
        Us, Vs = self.to_scaling(phase_x, phase_y, test_space=test_space)
        return W_fwd @ Us @ W_fwd.T, W_fwd @ Vs @ W_fwd.T

    def reconstruct(self, u_wav, v_wav, W_inv) -> Tuple[np.ndarray, np.ndarray]:
        return W_inv @ u_wav @ W_inv.T, W_inv @ v_wav @ W_inv.T

    def _solution(self, t: float, nu: float, phase_x: float = 0.0,
                       phase_y: float = 0.0, mu_d: float = 0.0
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """CLOSED-FORM time-dependent Taylor-Green vortex (Taylor & Green, 1937).

        With kx = ky = k (essential, it is what makes this an  solution):

            u = sin(kx+px) cos(ky+py),  v = -cos(kx+px) sin(ky+py)

        1. Divergence free:  d_x u + d_y v = 0 identically.
        2. Laplacian eigenfunction:  nabla^2 u = -2 k^2 u.
        3. The advection term is  a GRADIENT,
               (u.grad)u = -grad[ (1/4)(cos 2kx + cos 2ky) ],
           and, the field being divergence free, the skew-symmetric form
           coincides with the convective one here.  A gradient is absorbed
           into the pressure, so the full nonlinear system reduces to the
           heat equation  du/dt = nu nabla^2 u - mu_d u = -(2 nu k^2 + mu_d) u,
           i.e. pure exponential decay of a FIXED spatial profile.

        More generally (used by the stochastic benchmark): in 2-D,
        (u.grad)u = grad(|u|^2/2) - omega u^perp and, if nabla^2 psi = -c psi,
        omega u^perp = c psi grad psi = grad(c psi^2/2).  So EVERY
        stream function in one Laplacian eigenspace has a gradient
        nonlinearity.  For k0 = 2 the shell m^2 + n^2 = 2 k0^2 = 8 contains
        only (+-2, +-2): a 4-dimensional invariant subspace spanned by the
        four phases (px, py) in {0, pi/2}^2 of the TG field.

        CAVEAT for benchmarking.  Precisely because the nonlinearity is a pure
        gradient, this test exercises NOTHING of the advection operator beyond
        "the Leray projector removes it".  It validates the viscous solve, the
        projector and the wavelet transform; it does NOT validate the triad
        connection coefficients or the skew-symmetrisation.  Use
        ODE.validate_advection_operator / ODE.validate_skew_symmetry for those,
        and GaussianStreamfunctionIC for genuinely multi-scale dynamics.
        """
        k = 2 * self.k0 * np.pi
        decay = np.exp(-(2.0 * nu * (k ** 2) + mu_d) * t)
        u0, v0 = self.velocity_field(phase_x, phase_y)
        return u0 * decay, v0 * decay



# Benchmarks

def _project_to_scaling(proj, ops1d, Uw, Vw):
    """Project a wavelet-basis pair with the production projector and return
    the SCALING-basis pair (float64)."""
    n = Uw.shape[0]; n2 = n * n
    x, _ = proj.project(np.concatenate([Uw.reshape(-1), Vw.reshape(-1)]))
    Wi = ops1d.W_inv
    return Wi @ x[:n2].reshape(n, n) @ Wi.T, Wi @ x[n2:].reshape(n, n) @ Wi.T


def _l2_norm_s(ops1d, U, V):
    return float(np.sqrt(max(l2_inner_s(ops1d, (U, V), (U, V)), 0.0)))


def validate_taylor_green_(ops1d: WaveletOperators1D, projector, *, k0: int = 2,
                                nu: float = 0.01, dt: float = 1e-4, n_steps: int = 200,
                                tol: float = 1e-3, advection_scheme: str = "if_rk4",
                                triad_rtol: float = 0.0, dtype=None,
                                mu_d: float = 0.0, verbose: bool = True) -> dict:
    """Taylor-Green benchmark ON THE PRODUCTION STEPPER (advection on).

    Error = ||u_h(T) - c(u_(T))||_{L2} / ||c(u_(T))||_{L2}, with
    c(.) the  Galerkin coefficients and the L2 norm through the primal
    Gram matrix.  It splits  into two independent parts:

      solver  = ||u_h(T) - e^{l_h T} u_h(0)|| / ||.||   l_h = nu lambda_h - mu_d,
                lambda_h the DISCRETE Laplacian eigenvalue of the shell.  The
                discrete shell is invariant (P N(u, u) = 0 to roundoff on the
                shell) and if_rk4 integrates the linear part , so this is
                ROUNDOFF; anything larger is a bug.
      spatial = |e^{(l_h - l) T} - 1|: the eigenvalue error of the Galerkin
                Laplacian (spectral in h).

    Two further outputs isolate the nonlinear term:
      nonlinear_vs_linear  ||u_h(T) - u_h^{lin}(T)|| / ||u_h(T)||, the same run
                           with advection OFF: the advection of a shell field
                           is a gradient, so this is roundoff;
      off_shell            1 - sum_i <u_h(T), e_i>^2 / ||u_h(T)||^2, the energy
                           fraction that left the 4-dim TG shell (roundoff).
    Neither validates the triad stencils' energy TRANSFER (the shell
    nonlinearity is a gradient): ODE.validate_advection_operator does.
    """
    J, jmin, test_space = ops1d.J, ops1d.jmin, ops1d.test_space
    W_fwd = ops1d.W_fwd
    n = 2 ** J

    tg = TaylorGreenVortex(k0, J, jmin)
    Uw, Vw = tg.to_wavelet(W_fwd, test_space=test_space)
    U0, V0 = _project_to_scaling(projector, ops1d, Uw, Vw)

    def run(advection):
        step = build_scaling_basis_stepper(ops1d, projector, dt, nu,
                                           advection_scheme=advection_scheme,
                                           triad_rtol=triad_rtol, dtype=dtype,
                                           mu_d=mu_d, advection=advection)
        UV = (jnp.asarray(U0), jnp.asarray(V0))
        for _ in range(n_steps):
            UV = step(UV)
        return tuple(np.asarray(z, dtype=np.float64) for z in jax.block_until_ready(UV))

    U_num, V_num = run(True)
    U_lin, V_lin = run(False)

    T = n_steps * dt
    basis, lam_h, _ = taylor_green_shell_basis(ops1d, projector, k0)
    decay_h = np.exp((nu * lam_h - mu_d) * T)
    norm_num = _l2_norm_s(ops1d, U_num, V_num)
    solver_err = (_l2_norm_s(ops1d, U_num - decay_h * U0, V_num - decay_h * V0)
                  / (_l2_norm_s(ops1d, decay_h * U0, decay_h * V0) + 1e-300))
    spatial_err = abs(np.exp((nu * lam_h + 2.0 * nu * (2 * np.pi * k0) ** 2) * T) - 1.0)
    nl_vs_lin = _l2_norm_s(ops1d, U_num - U_lin, V_num - V_lin) / (norm_num + 1e-300)
    amps = np.array([l2_inner_s(ops1d, (U_num, V_num), (b[0], b[1])) for b in basis])
    off_shell = float(1.0 - np.sum(amps ** 2) / (norm_num ** 2 + 1e-300))
    u_ex, v_ex = tg._solution(T, nu, mu_d=mu_d)
    A = galerkin_multiplier(n, test_space)
    U_ex, V_ex = samples_to_coeffs(u_ex, A), samples_to_coeffs(v_ex, A)
    rel = (_l2_norm_s(ops1d, U_num - U_ex, V_num - V_ex)
           / (_l2_norm_s(ops1d, U_ex, V_ex) + 1e-300))
    x_num = np.concatenate([(W_fwd @ U_num @ W_fwd.T).reshape(-1),
                            (W_fwd @ V_num @ W_fwd.T).reshape(-1)])
    if verbose:
        print(f"  [validate_taylor_green_] J={J}, test_space={test_space}, "
              f"scheme={_resolve_scheme(advection_scheme)}, T={T:.3g}, "
              f"mu_d={mu_d:g}, nonlinear times k U T = {2 * np.pi * k0 * T:.2f}: "
              f"rel L2 error = {rel:.3e} = solver {solver_err:.1e} (roundoff expected) "
              f"+ spatial eigenvalue error {spatial_err:.1e}; advection on vs off "
              f"{nl_vs_lin:.1e}; off-shell energy {off_shell:.1e}; ||C x|| normalised = "
              f"{projector.normalised_residual(x_num):.2e}")
    if rel > tol:
        warnings.warn(f"Taylor-Green relative error {rel:.3e} exceeds tol={tol:.3e}")
    return {"rel_err": float(rel), "solver_err": float(solver_err),
            "spatial_err": float(spatial_err), "nonlinear_vs_linear": float(nl_vs_lin),
            "off_shell": off_shell}



# Multi-scale (turbulence-like) initial condition

class GaussianStreamfunctionIC:
    """Gaussian random divergence-free field  (u, v) = (d_y psi, -d_x psi).
 
        psi = sum_{m,n} a_mn [ z^cc cos(k_m x) cos(k_n y)
                             + z^cs cos(k_m x) sin(k_n y)
                             + z^sc sin(k_m x) cos(k_n y)
                             + z^ss sin(k_m x) sin(k_n y) ],
 
    all z ~ N(0,1) iid, over EVERY (m,n) in {1..k_max}^2 (no subset), with
    a_mn = (m^2 + n^2)^{-p/2} and p = (3 + alpha)/2.
 
    Before normalisation psi is a Gaussian random field with covariance
    diagonal in this basis.  ``velocity_field`` then divides by the sample
    RMS, and the dataset generator rescales to the energy E0, so the law
    actually used is NOT Gaussian (it lives on an energy sphere).  This is
    harmless -- the burn-in forgets the initial law -- but no statement
    below relies on Gaussianity of the IC.
 
    Spectral slope.  psi_k ~ k^{-p} gives u_k ~ k^{1-p}, hence
    E(k) ~ k |u_k|^2 ~ k^{3-2p}.  For a target E(k) ~ k^{-alpha},
    p = (3+alpha)/2; with alpha = 5/3 that is 7/3, NOT 5/3 (which would give
    E(k) ~ k^{+4/3}, energy INCREASING with wavenumber).
 
    All modes are kept: a random SUBSET of modes would make the initial law
    singular (supported on a finite union of subspaces).
    """
 
    def __init__(self, J: int, jmin: int = 0, k_max: int = 4,
                 alpha: float = 5 / 3, seed: int = 0):
        self.J, self.jmin, self.k_max, self.alpha = J, jmin, k_max, alpha
        self.n = 2 ** J
        self.dx = 1.0 / self.n
        if 2 * k_max >= self.n:
            warnings.warn(f"k_max={k_max} is not resolved on a {self.n}-point "
                          f"grid; require 2*k_max < n.")
        x = np.linspace(0, 1, self.n, endpoint=False)
        self.X, self.Y = np.meshgrid(x, x, indexing='ij')
 
        rng = np.random.default_rng(seed)
        p = (3.0 + alpha) / 2.0
        ms = np.arange(1, k_max + 1)
        self.modes = [(int(m), int(nn)) for m in ms for nn in ms]
        self.amn = np.array([(m ** 2 + nn ** 2) ** (-p / 2.0)
                             for (m, nn) in self.modes])
        # four independent Gaussians per mode: cc, cs, sc, ss
        self.zeta = rng.standard_normal((len(self.modes), 4))
 
    def velocity_field(self):
        u = np.zeros((self.n, self.n))
        v = np.zeros((self.n, self.n))
        for (m, nn), a, z in zip(self.modes, self.amn, self.zeta):
            kx, ky = 2 * np.pi * m, 2 * np.pi * nn
            cx, sx = np.cos(kx * self.X), np.sin(kx * self.X)
            cy, sy = np.cos(ky * self.Y), np.sin(ky * self.Y)
            # psi = a (z0 cx cy + z1 cx sy + z2 sx cy + z3 sx sy)
            # u = d_y psi
            u += a * ky * (z[0] * cx * (-sy) + z[1] * cx * cy
                           + z[2] * sx * (-sy) + z[3] * sx * cy)
            # v = -d_x psi
            v -= a * kx * (z[0] * (-sx) * cy + z[1] * (-sx) * sy
                           + z[2] * cx * cy + z[3] * cx * sy)
        rms = np.sqrt(np.mean(u ** 2 + v ** 2)) or 1.0
        return u / rms, v / rms
 
    def to_scaling(self, *, test_space: str):
        u, v = self.velocity_field()
        A = galerkin_multiplier(self.n, test_space)
        return samples_to_coeffs(u, A), samples_to_coeffs(v, A)
 
    def to_wavelet(self, W_fwd: np.ndarray, *, test_space: str):
        Us, Vs = self.to_scaling(test_space=test_space)
        return W_fwd @ Us @ W_fwd.T, W_fwd @ Vs @ W_fwd.T
 
 


# Transition-pair dataset (stochastically forced, statistically stationary)

DATASET_FORMAT_VERSION = 6     # bump whenever the stored layout or meaning changes


def _psi_orbit(ops_1d):
    from FORCING import build_orbit_map_2d
    return build_orbit_map_2d(ops_1d.J, ops_1d.jmin, n_comp=1)


def generate_transition_pair_dataset(
    ops_1d, spec, projector, dt, nu, mu_d, stride, *,
    n_trajectories=64, n_burn=20000, n_pairs_per_traj=64, pair_gap=1, seed=42,
    k_max=5, spec_alpha=5 / 3, E0=None, batch_size=8, path="./transition",
    save_every_batches=4, advection_scheme="if_rk4", triad_rtol=0.0,
    sigma_mode="TLM", sigma_probes=16, sigma_lin=None, stream=None,
    stoch_integrator="exp", dtype=np.float32, verbose=True, advection=True,
):
    r"""Paired dataset  D_Delta = {(x_p, y_p, m_p, rho_p)}  for p_Delta(y | x).

        x_p = x(t_p),  y_p = x(t_p + Delta),  m_p = Phi_Delta(x_p),
        t_{p+1} = t_p + pair_gap * Delta,   Delta = stride * dt,
        rho_p = orbit-pooled TLM ratio at x_p (MALLIAVIN; 1 for 'B0').

    sigma_mode: 'B0' (no probes), 'TLM' (rho, n_orb per pair) or 'TLM_FULL'
    (additionally rho_full, the un-pooled, un-floored per-coefficient ratio,
    n^2 per pair, from the SAME probes; rho = max(orbit_pool(rho_full),
    RHO_FLOOR) ).  Probes dominate the cost: n_probe * stride tangent
    steps per pair (GRID.grid_cost_report).

    y | x is an  draw from p_Delta(. | x) of the scheme; x is drawn from
    the law at time t_burn + t_p, which approaches mu_infty exponentially
    (rate ~ mu_d; the drift t-statistic of MAIN stage 4 is the check).  The
    lags between pairs are simulated with the same forced scheme, only not
    stored.  pair_gap > 1 decorrelates the conditioning states (consecutive
    x are nearly identical for Delta << 1/mu_d).

    Storage.  pair_gap == 1: states are contiguous, (P + 1) rows per
    trajectory, pair p = rows (p, p+1)  [identical to format 4].
    pair_gap  > 1: 2 P rows per trajectory, pair p = rows (2p, 2p+1).
    E_path, Z_path (n_traj, P*pair_gap + 1): energy and Galerkin dissipation
    after EVERY lag of the retained window (uniform spacing Delta), for
    the budget checks.  E_burn (n_traj, n_burn // stride): same during
    burn-in.  E, Z: values at the stored rows.

    Burn-in.  n_burn micro-steps are discarded; n_burn dt mu_d >= 3 drag
    times (1/mu_d, the slowest relaxation) is enforced as a warning.  The
    initial energy is set to E0.  Pass the stationary estimate
    (CERT.forcing_regime_report['E_stationary_estimate']); the default
    eps_inj / (2 mu_d) is the drag-only UPPER bound and starts every
    trajectory far above the invariant measure.

    Files (memmaps):  {path}_x.npy, {path}_m.npy, {path}_meta.npz.
    """
    from jax import random
    from FORCING import make_forced_stepper
    from ODE import make_wavelet_maps, scaling_dissipation_fn
    from TLM import make_sigma_ratio_fn, make_sigma_diag_fn, RHO_FLOOR

    if not isinstance(ops_1d, WaveletOperators1D):
        raise TypeError("ops_1d must be the WaveletOperators1D bundle")
    if not isinstance(projector, LerayProjector):
        raise TypeError("pass the pipeline's LerayProjector (one velocity space)")
    if spec.diag_Q is None or not np.isfinite(spec.eps_inj):
        raise RuntimeError("estimate eps_inj / diag(Q) before generating data")
    if sigma_mode not in ("B0", "TLM", "TLM_FULL"):
        raise ValueError("sigma_mode must be 'B0', 'TLM' or 'TLM_FULL'")
    if sigma_mode != "B0" and (sigma_lin is None or stream is None):
        raise ValueError(f"sigma_mode={sigma_mode!r} needs sigma_lin and stream")

    full = sigma_mode == "TLM_FULL"
    G_ = int(pair_gap)
    if G_ < 1:
        raise ValueError("pair_gap must be >= 1")
    J, jmin = ops_1d.J, ops_1d.jmin
    filter_nyquist = bool(projector.filter_nyquist)
    if mu_d > 0 and n_burn * dt * mu_d < 3.0:
        warnings.warn(f"burn-in = {n_burn*dt*mu_d:.2f} drag times (1/mu_d) < 3: "
                      f"the retained states are NOT yet stationary.")
    if E0 is None:
        E0 = spec.eps_inj / (2.0 * mu_d) if mu_d > 0 else None

    work = np.dtype(dtype)
    forced_step, det_step, noise = make_forced_stepper(
        ops_1d, projector, dt, nu, spec, mu_d=mu_d,
        advection_scheme=advection_scheme, triad_rtol=triad_rtol,
        filter_nyquist=filter_nyquist, dtype=work,
        stoch_integrator=stoch_integrator, advection=advection)
    from FORCING import orbit_pool as _orbit_pool
    psi_orb = _psi_orbit(ops_1d)
    if sigma_mode == "B0":
        ratio_fn = None
    elif full:       # raw rho_full; rho = max(orbit_pool(rho_full), RHO_FLOOR) below
        ratio_fn = make_sigma_diag_fn(det_step, noise, stream, sigma_lin, dt, stride,
                                      n_probe=sigma_probes, dtype=work)
    else:
        ratio_fn = make_sigma_ratio_fn(det_step, noise, stream, sigma_lin, psi_orb,
                                       dt, stride, n_probe=sigma_probes, dtype=work)

    n = 2 ** J
    d = 2 * n * n
    dx = 1.0 / n
    P = int(n_pairs_per_traj)
    n_orb = int(psi_orb.max()) + 1
    n_rho = n * n if full else n_orb 
    nb = n_burn // stride
    energy_fn = scaling_kinetic_energy_fn(ops_1d, dx, dtype=work)
    diss_fn = scaling_dissipation_fn(ops_1d, dx, dtype=work)
    _, s2w = make_wavelet_maps(ops_1d, dtype=work)

    # ---- initial conditions (float64, projected, energy set to E0) --------
    rng = np.random.default_rng(seed)
    U0s, V0s = [], []
    for _ in range(n_trajectories):
        ic = GaussianStreamfunctionIC(J, jmin, k_max=k_max, alpha=spec_alpha,
                                      seed=int(rng.integers(0, 2 ** 31 - 1)))
        uw, vw = ic.to_wavelet(ops_1d.W_fwd, test_space=ops_1d.test_space)
        U0, V0 = _project_to_scaling(projector, ops_1d, uw, vw)
        if E0 is not None:
            G = ops_1d.G_vj
            e = 0.5 * dx * dx * (np.sum(U0 * (G @ U0 @ G.T)) + np.sum(V0 * (G @ V0 @ G.T)))
            U0, V0 = U0 * np.sqrt(E0 / e), V0 * np.sqrt(E0 / e)
        U0s.append(U0)
        V0s.append(V0)
    U0_all = jnp.asarray(np.asarray(U0s), dtype=work)
    V0_all = jnp.asarray(np.asarray(V0s), dtype=work)

    def umax(c):
        return jnp.max(jnp.abs(c[0]) + jnp.abs(c[1]))

    def rollout_one(U0, V0, key):
        def micro(c, k):
            c = forced_step(c, k)
            return c, umax(c)

        def lag(c, ks):                                   # one Delta
            c, um = jax.lax.scan(micro, c, ks)
            return c, (energy_fn(*c), diss_fn(*c), jnp.max(um))

        def det_delta(c):
            return jax.lax.scan(lambda z, _: (det_step(z), None), c, None,
                                length=stride)[0]

        kb, kr = random.split(key)
        uv, (E_burn, _, _) = jax.lax.scan(
            lag, (U0, V0), random.split(kb, nb * stride).reshape(nb, stride, -1))

        def macro(c, ks):                                 # ks (G_, stride, 2)
            x = s2w(*c)
            m = s2w(*det_delta(c))
            rh = (jnp.ones((n_rho,), work) if ratio_fn is None
                  else ratio_fn(c[0], c[1]))
            um0 = umax(c)
            c, (e1, z1, u1) = lag(c, ks[0])
            y = s2w(*c)
            c, (e2, z2, u2) = jax.lax.scan(lag, c, ks[1:])   # G_-1 unrecorded lags
            es = jnp.concatenate([e1[None], e2])
            zs = jnp.concatenate([z1[None], z2])
            um = jnp.max(jnp.concatenate([um0[None], u1[None], u2]))
            return c, (x, y, m, rh, es, zs, um)

        e0, z0 = energy_fn(*uv), diss_fn(*uv)
        _, (xs, ys, ms, rh, es, zs, um) = jax.lax.scan(
            macro, uv, random.split(kr, P * G_ * stride).reshape(P, G_, stride, -1))
        return (xs, ys, ms, rh,
                jnp.concatenate([e0[None], es.reshape(-1)]),
                jnp.concatenate([z0[None], zs.reshape(-1)]),
                um, E_burn)

    rollout_batch = jax.jit(jax.vmap(rollout_one))

    # ---- layout ---------------------------------------------------------------
    rpt = P + 1 if G_ == 1 else 2 * P                     # stored rows per traj
    NT = P * G_ + 1                                       # path samples per traj
    if G_ == 1:
        local_pairs = np.stack([np.arange(P), np.arange(P) + 1], 1)
        row_lag = np.arange(P + 1)
    else:
        local_pairs = np.stack([2 * np.arange(P), 2 * np.arange(P) + 1], 1)
        row_lag = np.stack([G_ * np.arange(P), G_ * np.arange(P) + 1], 1).reshape(-1)
    pair_index = (np.arange(n_trajectories)[:, None, None] * rpt
                  + local_pairs[None]).reshape(-1, 2).astype(np.int64)
    traj_pair = np.repeat(np.arange(n_trajectories), P)
    traj_state = np.repeat(np.arange(n_trajectories), rpt)

    # ---- files, signature, resume --------------------------------------------
    x_path, m_path, meta_path = path + "_x.npy", path + "_m.npy", path + "_meta.npz"
    Ns, Np = n_trajectories * rpt, n_trajectories * P
    sig = dict(J=J, jmin=jmin, test_space=str(ops_1d.test_space), dt=float(dt),
               nu=float(nu), mu_d=float(mu_d), stride=int(stride),
               n_burn=int(n_burn), P=P, pair_gap=G_,
               n_trajectories=int(n_trajectories),
               seed=int(seed), k_max=int(k_max), spec_alpha=float(spec_alpha),
               q0=float(spec.q0), alpha_F=float(spec.alpha_F),
               eps_inj=float(spec.eps_inj),
               scheme=str(_resolve_scheme(advection_scheme)),
               filter_nyquist=filter_nyquist, triad_rtol=float(triad_rtol),
               sigma_mode=str(sigma_mode), sigma_probes=int(sigma_probes),
               stoch_integrator=str(stoch_integrator),
               work_dtype=str(work), format_version=DATASET_FORMAT_VERSION,
               advection=bool(advection), forcing_coords=str(spec.coords),
               E0=-1.0 if E0 is None else float(E0))   # nan would never compare equal
    EP = np.zeros((n_trajectories, NT)); ZP = np.zeros((n_trajectories, NT))
    UM = np.zeros(Np)
    RHO = np.ones((Np, n_orb)); EB = np.zeros((n_trajectories, nb))
    n_done, X, M, RF = 0, None, None, None
    rho_path = path + "_rhofull.npy"      # (Np, n^2), memmap: too big for npz
    if os.path.exists(meta_path) and os.path.exists(x_path) and os.path.exists(m_path):
        ck = np.load(meta_path, allow_pickle=False)
        if all(k in ck.files and np.asarray(ck[k]).item() == v for k, v in sig.items()):
            X = np.load(x_path, mmap_mode="r+")
            M = np.load(m_path, mmap_mode="r+")
            if full and os.path.exists(rho_path):
                RF = np.load(rho_path, mmap_mode="r+")
                if RF.shape != (Np, n * n):
                    RF = None
            if (X.shape == (Ns, d) and M.shape == (Np, d)
                    and (RF is not None or not full)):
                n_done = int(ck["n_done"])
                EP[:], ZP[:], UM[:] = ck["E_path"], ck["Z_path"], ck["umax"]
                RHO[:], EB[:] = ck["rho"], ck["E_burn"]
                if verbose:
                    print(f"  [dataset] resume at {n_done}/{n_trajectories}")
            else:
                X = M = None
        elif verbose:
            print("  [dataset] parameters changed: regenerating")
    if X is None:
        X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(Ns, d))
        M = np.lib.format.open_memmap(m_path, mode="w+", dtype=np.float32, shape=(Np, d))
        RF = None
    if full and RF is None:
        RF = np.lib.format.open_memmap(rho_path, mode="w+", dtype=np.float32,
                                       shape=(Np, n * n))

    def rows_E():
        return EP[:, row_lag].reshape(-1), ZP[:, row_lag].reshape(-1)

    def save(nd):
        X.flush(); M.flush()
        if RF is not None:
            RF.flush()
        E_rows, Z_rows = rows_E()
        np.savez(meta_path, n_done=nd, E=E_rows, Z=Z_rows, E_path=EP, Z_path=ZP,
                 umax=UM, rho=RHO, E_burn=EB, pair_index=pair_index,
                 traj_id=traj_pair, traj_id_state=traj_state, **sig)

    root = random.PRNGKey(seed)
    for bi, i in enumerate(range(n_done, n_trajectories, batch_size)):
        t0 = time.time()
        B = min(batch_size, n_trajectories - i)
        keys = jnp.stack([random.fold_in(root, i + b) for b in range(B)])
        xs, ys, ms, rh, ep, zp, um, eb = (np.asarray(a) for a in jax.block_until_ready(
            rollout_batch(U0_all[i:i + B], V0_all[i:i + B], keys)))
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            raise FloatingPointError(f"non-finite states in trajectories {i}..{i+B-1}: "
                                     "reduce dt (CFL) or the forcing amplitude")
        if G_ == 1:
            states = np.concatenate([xs, ys[:, -1:]], axis=1)          # (B, P+1, d)
        else:
            states = np.stack([xs, ys], axis=2).reshape(B, 2 * P, d)
        X[i * rpt:(i + B) * rpt] = states.reshape(-1, d).astype(np.float32)
        M[i * P:(i + B) * P] = ms.reshape(-1, d).astype(np.float32)
        EP[i:i + B], ZP[i:i + B] = ep, zp
        UM[i * P:(i + B) * P] = um.reshape(-1)
        rh_flat = rh.reshape(-1, n_rho)
        if full:
            RF[i * P:(i + B) * P] = rh_flat.astype(np.float32)
            # Sigma_bar_r is orbit-constant, so the floored orbit pool of the
            # per-coefficient ratio IS the L1.5 ratio (MALLIAVIN): one dataset
            # serves all three rungs.
            RHO[i * P:(i + B) * P] = np.maximum(_orbit_pool(rh_flat, psi_orb), RHO_FLOOR)
        else:
            RHO[i * P:(i + B) * P] = rh_flat
        EB[i:i + B] = eb
        if verbose:
            print(f"  [dataset] {i + B}/{n_trajectories} trajectories "
                  f"({time.time() - t0:.1f}s)", flush=True)
        if (bi + 1) % save_every_batches == 0:
            save(i + B)
    save(n_trajectories)

    cfl_factor = float(dt * advective_spectral_radius(ops_1d))
    E_rows, Z_rows = rows_E()
    return {"x": np.load(x_path, mmap_mode="r"), "m": np.load(m_path, mmap_mode="r"),
            "pair_index": pair_index, "traj_id": traj_pair,
            "traj_id_state": traj_state, "E": E_rows, "Z": Z_rows,
            "E_path": EP, "Z_path": ZP, "umax": UM,
            "rho": RHO, "E_burn": EB, "Delta": float(stride * dt),
            "rho_full": (np.load(rho_path, mmap_mode="r") if full else None),
            "cfl": cfl_factor * UM,
            "cfl_limit": float(IMAG_AXIS_STABILITY[_resolve_scheme(advection_scheme)]),
            "signature": sig}


def validate_time_integrator_order(ops1d, projector, *, nu=1e-4, dt=4e-4, T=2e-2,
                                   scheme="if_rk4", seed=1, triad_rtol=0.0,
                                   verbose=True):
    """Richardson test on the PRODUCTION stepper, on a genuinely nonlinear field.

    1.  It MUST run in float64: in float32 the successive Richardson
        differences sit at machine epsilon for T ~ 20 dt and the observed
        order measures roundoff.  Set NSWSGM_X64=1 before importing ODE, or
        this function refuses to run.
    2.  T must be long enough that the temporal error dominates roundoff;
        T >= ~100 dt is a safe default.

    Expected orders: 'imex_euler' 1, 'imex_cn' 2, 'if_rk4' 4.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "validate_time_integrator_order needs float64: set the environment "
            "variable NSWSGM_X64=1 before importing ODE.  In float32 the "
            "Richardson differences sit at machine epsilon and the measured "
            "order is meaningless.")
    scheme = _resolve_scheme(scheme)
    J, jmin, proj = ops1d.J, ops1d.jmin, projector
    Wf = ops1d.W_fwd

    tgs = GaussianStreamfunctionIC(J, jmin, k_max=5, alpha=5 / 3, seed=seed)
    uw, vw = tgs.to_wavelet(Wf, test_space=ops1d.test_space)
    U0, V0 = (jnp.asarray(z) for z in _project_to_scaling(proj, ops1d, uw, vw))

    sols = []
    for r in (1, 2, 4):
        h = dt / r
        n_steps = int(round(T / h))
        if abs(n_steps * h - T) > 1e-12 * T:
            raise ValueError(f"T={T} is not an integer multiple of dt/{r}")
        step = build_scaling_basis_stepper(
            ops1d, proj, h, nu, advection_scheme=scheme, triad_rtol=triad_rtol)
        UV = (U0, V0)
        for _ in range(n_steps):
            UV = step(UV)
        sols.append(np.stack([np.asarray(z) for z in jax.block_until_ready(UV)]))

    e1 = np.linalg.norm(sols[0] - sols[1])
    e2 = np.linalg.norm(sols[1] - sols[2])
    ref = np.linalg.norm(sols[2])
    order = float(np.log2(e1 / max(e2, 1e-300)))
    if verbose:
        print(f"  [validate_time_integrator_order] scheme={scheme}, T={T:.3g}: "
              f"||u_dt - u_dt/2||/||u|| = {e1/ref:.3e}, "
              f"||u_dt/2 - u_dt/4||/||u|| = {e2/ref:.3e}, "
              f"observed order = {order:.2f}")
    if min(e1, e2) / ref < 1e-12:
        warnings.warn("Richardson differences are near float64 epsilon; the "
                      "measured order is not meaningful.  Increase T or dt.")
        return order, e1 / ref
    expected = {"imex_euler": 1.0, "imex_cn": 2.0, "if_rk4": 4.0}[scheme]
    if abs(order - expected) > 0.25:
        warnings.warn(f"Observed temporal order {order:.2f} does not match the "
                      f"expected {expected:.0f} for scheme='{scheme}'.")
    return order, e1 / ref


# Stochastic Taylor-Green: closed-form law of the FORCED NONLINEAR solver

TG_PHASES = ((0.0, 0.0), (np.pi / 2, 0.0), (0.0, np.pi / 2), (np.pi / 2, np.pi / 2))


def taylor_green_shell_basis(ops1d, projector, k0: int = 2, dtype=None):
    r"""L2-orthonormal basis (e_1..e_4) of the discrete TG shell and its decay rate.

    e_i are the  Galerkin coefficients of the four TG phases (k0, k0),
    Leray-projected (a no-op up to roundoff: they are  discretely
    divergence free) and Gram-Schmidt orthonormalised in L2.  Every e_i is an
     eigenvector of the discrete linear operator (all operators are
    Fourier multipliers and the shell contains a single |theta| pair):

        e^{dt L} e_i = e^{l_h dt} e_i,   l_h = nu * lambda_h - mu_d,
        lambda_h = discrete symbol of the Laplacian at (k0, k0) ~ -2 (2 pi k0)^2.

    Returns (basis (4, 2, n, n) float64, l_h, eigen-residual).
    """
    J = ops1d.J
    n = 2 ** J
    if 2 * k0 >= n // 2:
        raise ValueError("k0 not resolved")
    tg = TaylorGreenVortex(k0, J, ops1d.jmin)
    Wf = ops1d.W_fwd
    basis = []
    for px, py in TG_PHASES:
        Us, Vs = tg.to_scaling(px, py, test_space=ops1d.test_space)
        U, V = _project_to_scaling(projector, ops1d, Wf @ Us @ Wf.T, Wf @ Vs @ Wf.T)
        for b in basis:
            c = l2_inner_s(ops1d, (U, V), b)
            U, V = U - c * b[0], V - c * b[1]
        nrm = np.sqrt(l2_inner_s(ops1d, (U, V), (U, V)))
        basis.append((U / nrm, V / nrm))
    Spec, modal, _, _, _ = make_scaling_pieces(ops1d, projector, dtype=jnp.float64)
    lam_s = []
    res = 0.0
    for b in basis:
        LU = np.asarray(modal(jnp.asarray(b[0]), Spec))       # (Delta_h (+) Delta_h) e
        LV = np.asarray(modal(jnp.asarray(b[1]), Spec))
        lam = l2_inner_s(ops1d, (LU, LV), b)
        lam_s.append(lam)
        rU, rV = LU - lam * b[0], LV - lam * b[1]
        res = max(res, float(np.sqrt(l2_inner_s(ops1d, (rU, rV), (rU, rV))) / abs(lam)))
    spread = float(np.ptp(lam_s) / abs(np.mean(lam_s)))
    # Roundoff floor, NOT a physical tolerance.  lam_s is a Rayleigh quotient
    # taken through a W-sandwich and two Gram products, so its relative
    # accuracy degrades with n (measured spread up to 3e-8, residual up to
    # 5e-5 at J <= 6).  `res` is the quantity with physical content (it bounds
    # the deviation of the  orbit A0 exp(-gamma t) e); the spread is a
    # coarse consistency check only.
    if spread > 1e-6 or res > 1e-3:
        raise RuntimeError(
            f"TG shell is not a single discrete eigenspace: eigenvalues {lam_s} "
            f"(relative spread {spread:.2e}, tol 1e-6; eigenvector residual "
            f"{res:.2e}, tol 1e-3)")
    return np.stack([np.stack(b) for b in basis]), float(np.mean(lam_s)), res


def validate_stochastic_taylor_green(ops1d, projector, dt, nu, mu_d, *, k0=2,
                                     E_star=0.25, E0=None, n_members=32,
                                     T=None, n_rec=24, advection_scheme="if_rk4",
                                     dtype=None, seed=0, verbose=True) -> dict:
    r"""Closed-form law of the FORCED, NONLINEAR production stepper.

    Forcing supported on the TG shell:  B_F zeta = sqrt(q) sum_i zeta_i e_i.
    The shell is invariant under the nonlinear dynamics (gradient
    nonlinearity, see TaylorGreenVortex._solution), so with
    u(0) = sum_i A_i(0) e_i the amplitudes A_i = <u, e_i>_{L2} are
    INDEPENDENT scalar OU processes, gamma = -l_h = 2 nu k^2 + mu_d (discrete):

        A_i(t) | A_i(0) ~ N( A_i(0) e^{-gamma t},  s2 (1 - e^{-2 gamma t}) ),
        s2 = q / (2 gamma),       E[E(t)] = 1/2 sum_i (mean_i^2 + var_i).

    The exponential Euler-Maruyama scheme reproduces this law  at any
    dt (e_i are eigenvectors of L), so every deviation is a solver error:
    noise injection, phi(dt L), advection not cancelling on the shell, or
    the projector.  q is set so that the stationary energy is E_star.

    Metrics (K = n_members, over all recorded times and modes):
        z_mean = max |abar - mean| / sqrt(var / K)     ~ |N(0,1)| max  (< ~3.5)
        var_dev = max |s^2 / var - 1| / sqrt(2/(K-1))   ~ same scale
        off_shell = max (E - 1/2 sum A_i^2) / E : energy that left the shell.
    CAVEAT.  The discrete shell is  invariant (P N(u, u) = 0 to
    roundoff on the shell, see validate_taylor_green_), but TG is
    linearly UNSTABLE at these Reynolds numbers: ROUNDOFF seeds off-shell
    modes that grow exponentially, so off_shell grows in time (much faster
    in float32).  off_shell is an ENERGY fraction, i.e. quadratic in the
    off-shell amplitude: 1e-16 in float64 means amplitudes ~1e-8.  It is
    reported, not asserted; run in float64 (NSWSGM_X64=1).
    """
    work = _resolve_dtype(dtype)
    n = 2 ** ops1d.J
    dx = 1.0 / n
    basis, lam_h, eig_res = taylor_green_shell_basis(ops1d, projector, k0)
    l_h = nu * lam_h - mu_d
    gamma = -l_h
    s2 = 0.5 * E_star                        # E_stat = 1/2 * 4 * s2
    q = 2.0 * gamma * s2
    E0 = E_star if E0 is None else E0
    a0 = np.array([np.sqrt(2.0 * E0), 0.0, 0.0, 0.0])
    if T is None:
        T = 1.0 / gamma
    n_sub = max(1, int(round(T / (n_rec * dt))))
    t_rec = dt * n_sub * np.arange(n_rec + 1)

    det_step = build_scaling_basis_stepper(ops1d, projector, dt, nu,
                                           advection_scheme=advection_scheme,
                                           dtype=work, mu_d=mu_d)
    B = jnp.asarray(basis, work)                                   # (4, 2, n, n)
    amp = jnp.asarray(np.sqrt(dt * q) * float(phi_noise_multiplier(jnp.asarray(dt * l_h))), work)
    Gj = jnp.asarray(ops1d.G_vj, work)
    energy = scaling_kinetic_energy_fn(ops1d, dx, dtype=work)

    def proj_amp(c):
        GU = Gj @ c[0] @ Gj.T
        GV = Gj @ c[1] @ Gj.T
        return (jnp.einsum('kab,ab->k', B[:, 0], GU) + jnp.einsum('kab,ab->k', B[:, 1], GV)) * dx * dx

    def step(c, key):
        U, V = det_step(c)
        w = amp * random.normal(key, (4,), work)
        f = jnp.einsum('k,kcab->cab', w, B)
        return (U + f[0], V + f[1]), None

    @jax.jit
    def advance(cs, keys):                         # cs: (K, n, n) pair
        def one(c, ks):
            c, _ = jax.lax.scan(step, c, ks)
            return c
        return jax.vmap(one)(cs, keys)

    obs = jax.jit(jax.vmap(lambda U, V: (proj_amp((U, V)), energy(U, V))))
    x0 = np.einsum('k,kcab->cab', a0, basis)
    K = int(n_members)
    cs = (jnp.asarray(np.broadcast_to(x0[0], (K, n, n)), work),
          jnp.asarray(np.broadcast_to(x0[1], (K, n, n)), work))
    key = random.PRNGKey(seed)
    A = np.zeros((n_rec + 1, K, 4))
    E = np.zeros((n_rec + 1, K))
    a_, e_ = obs(*cs)
    A[0], E[0] = np.asarray(a_), np.asarray(e_)
    for r in range(1, n_rec + 1):
        key, sk = random.split(key)
        keys = random.split(sk, K * n_sub).reshape(K, n_sub, -1)
        cs = advance(cs, keys)
        a_, e_ = obs(*cs)
        A[r], E[r] = np.asarray(a_), np.asarray(e_)
    if not np.all(np.isfinite(E)):
        raise FloatingPointError("stochastic TG benchmark diverged")

    mean_th = a0[None, :] * np.exp(-gamma * t_rec)[:, None]           # (R, 4)
    var_th = s2 * (-np.expm1(-2.0 * gamma * t_rec))[:, None] * np.ones((1, 4))
    E_th = 0.5 * np.sum(mean_th ** 2 + var_th, axis=1)
    abar = A.mean(1)
    avar = A.var(1, ddof=1)
    ok_t = t_rec > 0
    z_mean = np.abs(abar - mean_th)[ok_t] / np.sqrt(var_th[ok_t] / K)
    var_dev = np.abs(avar[ok_t] / var_th[ok_t] - 1.0) / np.sqrt(2.0 / (K - 1))
    off = (E - 0.5 * np.sum(A ** 2, axis=2)) / E
    zE = (E.mean(1) - E_th)[ok_t] / (E.std(1, ddof=1)[ok_t] / np.sqrt(K))
    # final-time standardised amplitudes: Kolmogorov-Smirnov vs N(0, 1)
    from scipy.stats import kstest
    zfin = ((A[-1] - mean_th[-1]) / np.sqrt(var_th[-1])).ravel()
    ks = kstest(zfin, "norm")
    out = {"t": t_rec, "A": A, "E": E, "mean_th": mean_th, "var_th": var_th,
           "E_th": E_th, "gamma": gamma, "gamma_continuum": 2.0 * nu * (2 * np.pi * k0) ** 2 + mu_d,
           "eig_residual": eig_res, "z_mean_max": float(z_mean.max()),
           "var_dev_max": float(var_dev.max()), "zE_max": float(np.abs(zE).max()),
           "off_shell_max": float(np.max(np.abs(off))), "off_shell_t": np.max(np.abs(off), axis=1),
           "ks_pvalue": float(ks.pvalue), "K": K, "q": q, "s2": s2,
           "dtype": jnp.dtype(work).name}
    # the three statistics are max-of-many |N(0,1)|-like numbers
    # Student-t quantile (K - 1 dof: the energy z uses the sample sd),
    # Bonferroni over all recorded (time, mode) tests, family-wise level 1e-3
    from scipy.stats import t as student_t
    n_tests = z_mean.size
    thr = float(student_t.ppf(1.0 - 1e-3 / (2.0 * max(n_tests, 1)), K - 1))
    out["threshold"] = thr
    out["ok"] = bool(out["z_mean_max"] < thr and out["var_dev_max"] < thr
                     and out["zE_max"] < thr and ks.pvalue > 1e-3)
    if verbose:
        print(f"  [validate_stochastic_taylor_green] k0={k0}, gamma_h={gamma:.5f} "
              f"(continuum {out['gamma_continuum']:.5f}), eigen-residual {eig_res:.1e}, "
              f"K={K}, T={t_rec[-1]:.3g}, {out['dtype']}")
        print(f"      max z(mean) = {out['z_mean_max']:.2f}, max z(var) = "
              f"{out['var_dev_max']:.2f}, max z(E) = {out['zE_max']:.2f} "
              f"(threshold {thr:.2f}); KS p = {ks.pvalue:.3f}; "
              f"off-shell energy fraction <= {out['off_shell_max']:.1e} -> "
              f"{'OK' if out['ok'] else 'FAIL'}")
    return out