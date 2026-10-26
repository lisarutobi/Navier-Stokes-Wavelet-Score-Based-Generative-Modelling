r"""BENCH.py - ensemble evaluation of the emulator and closed-form benchmarks.

1.  Solver-vs-emulator ensemble comparison (``compare_ensembles``)
    Both ensembles start from the SAME burned-in dataset state x0 and use
    independent noise.  Pathwise agreement is impossible (different noise),
    so only LAWS are compared, at every lag t_k = k Delta:

      E(t), Z(t)       mean +- sd over members; z-score of the difference
      spread(t)        E_K || u_i - ubar ||_{L2}^2 / 2   (ensemble dispersion)
      acc(t)           L2 cosine between the two ensemble MEANS (skill of the
                       predictable part; -> noise level as both means -> 0)
      at snapshots     isotropic spectra, per-level velocity RMS / kurtosis,
                       vorticity PDFs, two-sample KS of member energies
                       (members are independent -> valid p-values)

2.  Closed-form stochastic benchmark of the diffusion model
    (``run_tg_diffusion_benchmark``)
    The FULL pipeline (dataset, surrogate, dequantiser, training,
    sampler, rollout) run on the Stokes + drag system (advection off).  Its
    transition law is exactly Gaussian, N(e^{Delta L} x, Sigma_Delta), and,
    with u(0) = A0 e_TG (the single-mode Taylor-Green field, an eigenvector
    of L with eigenvalue -gamma):

      E[a(t)]  = A0 e^{-gamma t},             a = <u, e_TG>_{L2}
      Var a(t) = f_t(gamma) q_TG,              f_t = (1 - e^{-2 gamma t})/(2 gamma),
                                              q_TG = <e_TG, Q e_TG>_{L2}
      E[E(t)]  = e^{-2 gamma t} A0^2/2 + E_noise(t),
      E_noise(t) = 1/2 dx^2 Tr( G2 Q (I - e^{2 t L}) (-2L)^{-1} ),

    all EXACT for the exponential Euler-Maruyama scheme (FORCING docstring)
    and computed without Monte Carlo (one column per dyadic orbit).  The
    only surrogate error is the off-diagonal part of Sigma_lin, which is
    known exactly (GROUP.oracle_from_diffusion).  Three certificates:

      (i)   the EMPIRICAL linear floor must reproduce the exact oracle gain
            within its SE (this is the one place where the floor used on
            nonlinear data has a known answer);
      (ii)  the network must sit AT the floor (CERT.gain_verdict '0'): on
            linear data there is nothing x-dependent to learn;
      (iii) the emulator must reproduce the closed-form curves within MC error.

    (A closed-form NONLINEAR stochastic law exists only on the 4-dim TG
    shell - TG.validate_stochastic_taylor_green - and is singular in
    R^{n^2}: no density, so it cannot be a diffusion target.)
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from scipy.linalg import eigh
from scipy.stats import ks_2samp, wasserstein_distance

from ODE import (make_wavelet_maps, scaling_kinetic_energy_fn,
                 scaling_dissipation_fn, _resolve_filter, make_scaling_pieces)
from CERT import galerkin_spectra
from FORCING import (build_level_map_2d, build_orbit_map_2d,
                     orbit_representatives, make_noise_fn, make_forced_stepper)




# 0.  Shared maps

def make_phi_delta(ops1d, det_step, stride: int, dtype=jnp.float32):
    """Batched noiseless flow map over one lag in wavelet coordinates,
    (B, 2n^2) -> (B, 2n^2).  ``det_step`` MUST be the map returned by
    FORCING.make_forced_stepper (the one that produced m in the dataset)."""
    w2s, s2w = make_wavelet_maps(ops1d, dtype=dtype)

    def one(U, V):
        return jax.lax.scan(lambda c, _: (det_step(c), None), (U, V), None,
                            length=int(stride))[0]

    @jax.jit
    def phi(x):
        x = jnp.asarray(x, dtype)
        U, V = w2s(jnp.atleast_2d(x))
        U, V = jax.vmap(one)(U, V)
        out = s2w(U, V)
        return out if x.ndim == 2 else out[0]
    return phi


def make_observables(ops1d, stream, dtype=jnp.float32):
    """jit fns on batched wavelet states: (E, Z) and vorticity (scaling grid)."""
    dx = 1.0 / 2 ** ops1d.J
    w2s, _ = make_wavelet_maps(ops1d, dtype=dtype)
    energy = scaling_kinetic_energy_fn(ops1d, dx, dtype=dtype)
    diss = scaling_dissipation_fn(ops1d, dx, dtype=dtype)

    @jax.jit
    def EZ(x):
        U, V = w2s(x)
        E = jax.vmap(energy)(U, V)
        Z = jax.vmap(diss)(U, V)
        Ub, Vb = jnp.mean(U, 0), jnp.mean(V, 0)
        spread = jnp.mean(jax.vmap(energy)(U - Ub, V - Vb))
        mean_state = jnp.concatenate([jnp.mean(x[:, :x.shape[1] // 2], 0),
                                      jnp.mean(x[:, x.shape[1] // 2:], 0)])
        return E, Z, spread, mean_state

    @jax.jit
    def vort(x):
        U, V = w2s(x)
        return stream.vorticity_s(U, V)

    return EZ, vort


def l2_inner_w(ops1d, x, y):
    """<x_h, y_h>_{L2} for flattened wavelet states (G_wav is the L2 Gram)."""
    n = 2 ** ops1d.J
    n2 = n * n
    G = ops1d.G
    dx = 1.0 / n
    X = np.asarray(x, float)
    Y = np.asarray(y, float)
    return float((np.sum(X[:n2].reshape(n, n) * (G @ Y[:n2].reshape(n, n) @ G.T))
                  + np.sum(X[n2:].reshape(n, n) * (G @ Y[n2:].reshape(n, n) @ G.T)))
                 * dx * dx)


def level_moments(X, J, jmin):
    """Per-level RMS and per-orbit-standardised excess kurtosis (velocity states)."""
    X = np.asarray(X, np.float64)
    lev = build_level_map_2d(J, jmin, n_comp=2)
    orb = build_orbit_map_2d(J, jmin, n_comp=2)
    n_orb = int(orb.max()) + 1
    m2 = (np.bincount(orb, weights=(X ** 2).mean(0), minlength=n_orb)
          / np.bincount(orb, minlength=n_orb))
    Zs = X / np.sqrt(np.maximum(m2[orb], 1e-300))
    out = {}
    for j in range(jmin - 1, J):
        mk = lev == j
        out[int(j)] = {"rms": float(np.sqrt(np.mean(X[:, mk] ** 2))),
                       "kurtosis": float(np.mean(Zs[:, mk] ** 4) - 3.0)}
    return out



# 1.  Ensemble rollouts

def solver_ensemble(ops1d, forced_step, stride, x0, n_members, n_lags, *,
                    EZ, snap_lags=(), seed=0, dtype=jnp.float32, chunk=16,
                    verbose=True):
    """K forced-solver members from the same wavelet state x0."""
    w2s, s2w = make_wavelet_maps(ops1d, dtype=dtype)

    @jax.jit
    def advance(U, V, keys):                      # keys (B, stride, 2)
        def one(u, v, ks):
            return jax.lax.scan(lambda c, k: (forced_step(c, k), None), (u, v), ks)[0]
        return jax.vmap(one)(U, V, keys)

    return _rollout(lambda x, key: _solver_lag(x, key, advance, w2s, s2w, stride),
                    x0, n_members, n_lags, EZ, snap_lags, seed, dtype, chunk,
                    "solver", verbose)


def _solver_lag(x, key, advance, w2s, s2w, stride):
    U, V = w2s(x)
    keys = random.split(key, x.shape[0] * stride).reshape(x.shape[0], stride, -1)
    U, V = advance(U, V, keys)
    return s2w(U, V)


def emulator_ensemble(generate, params, phi_delta, x0, n_members, n_lags, *,
                      EZ, rho_fn=None, snap_lags=(), seed=0, chunk=16,
                      sampler_kw=None, verbose=True):
    """K emulator members: x_{k+1} ~ p_theta(. | x_k), m_k = Phi_Delta(x_k).

    ``rho_fn`` (x -> rho(x), batched) supplies the L1.5 multiplier at every
    lag; it runs the TLM probes, which dominate the emulator's cost."""
    kw = dict(sampler_kw or {})
    clip = []

    def lag(x, key):
        g = None if rho_fn is None else rho_fn(x)
        out = generate(params, key, phi_delta(x), x, g, **kw)
        clip.append(float(jnp.max(out["clip_frac"])))
        return out["y"]

    res = _rollout(lag, x0, n_members, n_lags, EZ, snap_lags, seed,
                   jnp.float32, chunk, "emulator", verbose)
    res["clip_max"] = max(clip) if clip else 0.0
    return res


def _rollout(lag_fn, x0, K, n_lags, EZ, snap_lags, seed, dtype, chunk, name, verbose):
    x0 = np.asarray(x0, np.float32)
    snap_lags = sorted(set(int(s) for s in snap_lags))
    E = np.zeros((n_lags + 1, K))
    Z = np.zeros((n_lags + 1, K))
    spread = np.zeros(n_lags + 1)
    means = np.zeros((n_lags + 1, x0.size), np.float32)
    snaps = {}
    key = random.PRNGKey(seed)
    t0 = time.time()
    xs = [jnp.asarray(np.broadcast_to(x0, (min(chunk, K - i), x0.size)), dtype)
          for i in range(0, K, chunk)]

    def record(k):
        X = jnp.concatenate(xs)
        e, z, sp, mu = EZ(X)
        E[k], Z[k], spread[k], means[k] = np.asarray(e), np.asarray(z), float(sp), np.asarray(mu)
        if k in snap_lags:
            snaps[k] = np.asarray(X, np.float32)

    record(0)
    for k in range(1, n_lags + 1):
        key, *sub = random.split(key, len(xs) + 1)
        xs = [lag_fn(x, s) for x, s in zip(xs, sub)]
        record(k)
        if not np.all(np.isfinite(E[k])):
            raise FloatingPointError(f"{name} ensemble diverged at lag {k}")
        if verbose and (k % max(1, n_lags // 5) == 0 or k == n_lags):
            print(f"    [{name}] lag {k}/{n_lags}  <E> = {E[k].mean():.4e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return {"E": E, "Z": Z, "spread": spread, "mean": means, "snaps": snaps}



# 2.  Solver-vs-emulator comparison

def compare_ensembles(ops1d, stream, sol, emu, Delta, nu, mu_d, eps_inj, *,
                      ref_states=None, vort_fn=None) -> dict:
    """Statistics and scalar metrics of two ensembles sharing x0 (see module doc)."""
    J, jmin = ops1d.J, ops1d.jmin
    n = 2 ** J
    n2 = n * n
    Wi = ops1d.W_inv
    L = sol["E"].shape[0]
    t = Delta * np.arange(L)
    K_s, K_e = sol["E"].shape[1], emu["E"].shape[1]

    def zdiff(a, b):
        se = np.sqrt(a.var(1, ddof=1) / a.shape[1] + b.var(1, ddof=1) / b.shape[1])
        return (a.mean(1) - b.mean(1)) / np.maximum(se, 1e-300)

    out = {"t": t, "Delta": Delta, "K_solver": K_s, "K_emulator": K_e,
           "_sol": {k: sol[k] for k in ("E", "Z", "spread")},
           "_emu": {k: emu[k] for k in ("E", "Z", "spread")}}
    out["zE"] = zdiff(emu["E"], sol["E"])
    out["zZ"] = zdiff(emu["Z"], sol["Z"])
    out["spread_ratio"] = np.sqrt(emu["spread"][1:] / np.maximum(sol["spread"][1:], 1e-300))
    acc = np.zeros(L)
    for k in range(L):
        a, b = emu["mean"][k], sol["mean"][k]
        acc[k] = l2_inner_w(ops1d, a, b) / np.sqrt(
            max(l2_inner_w(ops1d, a, a) * l2_inner_w(ops1d, b, b), 1e-300))
    out["acc"] = acc
    # Reference ACC: expected cosine between the means of two INDEPENDENT
    # ensembles (sizes K_e, K_s) drawn from the SOLVER law.  With
    # mu2 = |E u|^2 (unbiased) and Tr = E|u - E u|^2 (= 2 spread K/(K-1)):
    #     ACC_ref = mu2 / sqrt((mu2 + Tr/K_e)(mu2 + Tr/K_s)).
    # acc < acc_ref means lost predictable signal; without it acc is unreadable.
    acc_ref = np.zeros(L)
    for k in range(L):
        b = sol["mean"][k]
        tr = 2.0 * sol["spread"][k] * K_s / max(K_s - 1, 1)
        mu2 = max(l2_inner_w(ops1d, b, b) - tr / K_s, 0.0)
        acc_ref[k] = mu2 / np.sqrt(max((mu2 + tr / K_e) * (mu2 + tr / K_s), 1e-300))
    out["acc_ref"] = acc_ref

    # stationary identity over the second half (ensemble x time)
    h = L // 2
    for name, d in (("solver", sol), ("emulator", emu)):
        lhs = 2 * mu_d * d["E"][h:] + 2 * nu * d["Z"][h:]
        per_member = lhs.mean(0)
        out[f"identity_{name}"] = {
            "rel_gap": float((per_member.mean() - eps_inj) / eps_inj),
            "se": float(per_member.std(ddof=1) / np.sqrt(len(per_member)) / eps_inj),
            "window_drag_times": float(mu_d * Delta * (L - h)),
            "per_member": per_member}
    ps, pe = out["identity_solver"].pop("per_member"), out["identity_emulator"].pop("per_member")
    out["identity_diff"] = {
        "rel": float((pe.mean() - ps.mean()) / eps_inj),
        "se": float(np.sqrt(pe.var(ddof=1) / len(pe) + ps.var(ddof=1) / len(ps)) / eps_inj)}

    def spectra(X):
        """Per-member Galerkin shell spectra (the scheme's own E), shells k >= 1."""
        X = np.asarray(X, float)
        E = np.stack([galerkin_spectra(x[None], ops1d)["E_k"][1:] for x in X])
        return np.arange(1, E.shape[1] + 1), E

    # snapshots
    snap = {}
    for k in sorted(set(sol["snaps"]) & set(emu["snaps"])):
        S, Em = sol["snaps"][k], emu["snaps"][k]
        rec = {"t": k * Delta}
        kk, sp_s = spectra(S)
        _, sp_e = spectra(Em)
        rec["k"], rec["spec_solver"], rec["spec_emulator"] = kk, sp_s, sp_e
        ms, me = sp_s.mean(0), sp_e.mean(0)
        # complete shells only (k <= n/2): the corner shells are partial and
        # dominate an unweighted log-RMSE.  Reported per band.
        good = (ms > 0) & (me > 0) & (kk <= n // 2)
        lr = np.log(me[good] / ms[good])
        kg = kk[good]
        rec["spec_log_rmse"] = float(np.sqrt(np.mean(lr ** 2)))
        rec["spec_log_bias_bands"] = {
            band: float(np.mean(lr[(kg > lo) & (kg <= hi)]))
            for band, (lo, hi) in {"k<=n/8": (0, n // 8), "n/8<k<=n/4": (n // 8, n // 4),
                                   "n/4<k<=n/2": (n // 4, n // 2)}.items()}
        rec["levels_solver"] = level_moments(S, J, jmin)
        rec["levels_emulator"] = level_moments(Em, J, jmin)
        rec["level_var_log_ratio"] = {
            j: float(2 * np.log(rec["levels_emulator"][j]["rms"]
                                / rec["levels_solver"][j]["rms"]))
            for j in rec["levels_solver"]}
        if vort_fn is not None:
            ws = np.asarray(vort_fn(jnp.asarray(S)), float).ravel()
            we = np.asarray(vort_fn(jnp.asarray(Em)), float).ravel()
            rec["vort_solver"], rec["vort_emulator"] = ws, we
            rec["vort_W1"] = float(wasserstein_distance(ws, we) / ws.std())
        if k > 0:
            ks = ks_2samp(sol["E"][k], emu["E"][k])
            rec["energy_ks_p"] = float(ks.pvalue)
        snap[k] = rec
    out["snapshots"] = snap
    if ref_states is not None:
        R = np.asarray(ref_states, np.float32)
        out["ref_spectrum"] = spectra(R)[1].mean(0)
        out["ref_levels"] = level_moments(R, J, jmin)
        if vort_fn is not None:
            out["ref_vort"] = np.asarray(vort_fn(jnp.asarray(R)), float).ravel()

    last = max(snap) if snap else None
    burn = L // 2
    out["metrics"] = {
        "max_abs_zE": float(np.max(np.abs(out["zE"][1:]))),
        "max_abs_zZ": float(np.max(np.abs(out["zZ"][1:]))),
        "rel_err_meanE_late": float(abs(emu["E"][burn:].mean() / sol["E"][burn:].mean() - 1)),
        "rel_err_meanZ_late": float(abs(emu["Z"][burn:].mean() / sol["Z"][burn:].mean() - 1)),
        "spread_ratio_final": float(out["spread_ratio"][-1]),
        "spread_ratio_median": float(np.median(out["spread_ratio"])),
        "acc_final": float(acc[-1]),
        "acc_ref_final": float(acc_ref[-1]),
        "identity_diff": out["identity_diff"]["rel"],
        "identity_diff_se": out["identity_diff"]["se"],
    }
    if last is not None:
        r = snap[last]
        out["metrics"].update({
            "spec_log_rmse_final": r["spec_log_rmse"],
            "spec_log_bias_bands_final": r["spec_log_bias_bands"],
            "vort_W1_over_sd_final": r.get("vort_W1", float("nan")),
            "energy_ks_p_final": r.get("energy_ks_p", float("nan")),
            "max_abs_level_var_log_ratio_final": float(max(abs(v) for v in r["level_var_log_ratio"].values())),
        })
    return out


def print_comparison(cmp: dict, label: str = "solver vs emulator"):
    m = cmp["metrics"]
    print(f"    [{label}] K = {cmp['K_solver']} / {cmp['K_emulator']}, "
          f"T = {cmp['t'][-1]:.3g}")
    print(f"      max |z| of <E>(t) difference = {m['max_abs_zE']:.2f}, of <Z>(t) = "
          f"{m['max_abs_zZ']:.2f}   (|N(0,1)| max over {len(cmp['t']) - 1} correlated lags)")
    print(f"      late-window rel. error: <E> {m['rel_err_meanE_late']:.3f}, "
          f"<Z> {m['rel_err_meanZ_late']:.3f}")
    print(f"      spread ratio emu/solver: final {m['spread_ratio_final']:.3f}, "
          f"median {m['spread_ratio_median']:.3f} (1 = correct dispersion)")
    print(f"      ACC of ensemble means (final) = {m['acc_final']:.3f} "
          f"(expected for two solver ensembles of these sizes: {m['acc_ref_final']:.3f})")
    if "spec_log_rmse_final" in m:
        print(f"      final: spectrum log-RMSE {m['spec_log_rmse_final']:.3f}, "
              f"vorticity W1/sd {m['vort_W1_over_sd_final']:.3f}, "
              f"energy KS p {m['energy_ks_p_final']:.3f}, "
              f"max |log var ratio| per level {m['max_abs_level_var_log_ratio_final']:.3f}")
    if "spec_log_bias_bands_final" in m:
        print("      final spectrum mean log(E_emu/E_sol) per band: " + ", ".join(
            f"{b} {v:+.3f}" for b, v in m["spec_log_bias_bands_final"].items()))
    for name in ("solver", "emulator"):
        d = cmp[f"identity_{name}"]
        print(f"      budget 2mu<E>+2nu<Z> vs eps ({name}, {d['window_drag_times']:.2f} drag "
              f"times, conditional on x0): {d['rel_gap']:+.2%} +- {d['se']:.2%}")
    d = cmp["identity_diff"]
    print(f"      budget, emulator - solver (paired): {d['rel']:+.2%} +- {d['se']:.2%} "
          f"({d['rel'] / max(d['se'], 1e-300):+.1f} SE)")



# 3.  Closed-form linear (Stokes + drag) statistics

class LinearOUClosedForm:
    r"""Exact energy and eigenmode statistics of the linear forced scheme.

    Bubnov only (L symmetric in scaling coordinates, G = M, so the modal
    basis K V = M V diag(lam_h), V^T M V = I is G-orthonormal: with the
    modal analysis map V^{-1} = V^T M, x^T G x = ||V^{-1} x||^2).  For a
    column c of B_F (continuous Q),
    ||e^{tL} c||_G^2 = sum_ab e^{2 l_ab t} (V^{-1} C V^{-T})_ab^2,
    l_ab = nu (lam_a + lam_b) - mu_d.
    Tr over all columns = sum over dyadic orbits |O| x (representative
    column), because the weight is shift invariant (FORCING lemma).
    """

    def __init__(self, ops1d, projector, spec, nu, mu_d, dtype=None):
        if ops1d.test_space != "primal":
            raise ValueError("closed forms assume the Bubnov scheme (test_space='primal')")
        work = jnp.result_type(float) if dtype is None else jnp.dtype(dtype)
        n = 2 ** ops1d.J
        self.n, self.dx = n, 1.0 / n
        Mv = np.asarray(ops1d.M_vj, float)
        lam_h, V_mod = eigh(np.asarray(ops1d.K_vj, float), Mv)
        self.Vinv = V_mod.T @ Mv                                          # modal analysis
        self.l = nu * (lam_h[:, None] + lam_h[None, :]) - mu_d            # (n, n) < 0
        if np.max(self.l) >= 0:
            raise ValueError("closed form needs mu_d > 0 (all modes damped)")
        _, _, _, project, _ = make_scaling_pieces(ops1d, projector, dtype=work)
        self.noise = make_noise_fn(ops1d, project, spec, dtype=work,
                                   filter_nyquist=_resolve_filter(projector, None))
        orbit = np.asarray(spec.orbit_per_coeff)
        size = np.bincount(orbit).astype(float)
        col = jax.jit(self.noise.from_normals)
        acc = np.zeros((n, n))
        for o_, k in enumerate(orbit_representatives(orbit)):
            e = np.zeros(spec.n_in)
            e[k] = 1.0
            U, V = (np.asarray(a, float) for a in col(jnp.asarray(e, work)))
            acc += size[o_] * ((self.Vinv @ U @ self.Vinv.T) ** 2
                               + (self.Vinv @ V @ self.Vinv.T) ** 2)
        self.qhat = acc                                   # modal diag of Q (G-weighted)
        self.eps_inj = 0.5 * self.dx ** 2 * float(np.sum(acc))
        self.nu, self.mu_d = nu, mu_d
        self._G = np.asarray(ops1d.G_vj, float)
        self._work = work

    def f(self, l, t):
        return -np.expm1(2.0 * l * t) / (-2.0 * l)

    def energy_noise(self, t):
        """E_noise(t) = 1/2 dx^2 Tr(G2 C_t); t may be an array; t = inf allowed."""
        t = np.atleast_1d(np.asarray(t, float))
        out = []
        for tt in t:
            w = (1.0 / (-2.0 * self.l)) if np.isinf(tt) else self.f(self.l, tt)
            out.append(0.5 * self.dx ** 2 * float(np.sum(w * self.qhat)))
        return np.array(out)

    def dissipation_noise(self, t):
        """<Z_gal> of the noise part: Z = -1/2 dx^2 x^T G2 (Delta_h (+) Delta_h) x."""
        t = np.atleast_1d(np.asarray(t, float))
        spec_ = (self.l + self.mu_d) / self.nu            # lam_a + lam_b
        out = []
        for tt in t:
            w = (1.0 / (-2.0 * self.l)) if np.isinf(tt) else self.f(self.l, tt)
            out.append(-0.5 * self.dx ** 2 * float(np.sum(spec_ * w * self.qhat)))
        return np.array(out)

    def mode_forcing(self, e_s):
        """q_e = <e, Q e>_{L2} = dx^4 || B_F^T G2 e ||^2 (one VJP), e = (U, V) scaling."""
        work = self._work
        G = jnp.asarray(self._G, work)
        dx2 = self.dx ** 2
        g = (dx2 * (G @ jnp.asarray(e_s[0], work) @ G.T),
             dx2 * (G @ jnp.asarray(e_s[1], work) @ G.T))
        zero = jnp.zeros((self.noise.n_in,), work)
        _, vjp = jax.vjp(self.noise.from_normals, zero)
        (w,) = vjp(g)
        return float(jnp.sum(w * w))


def run_tg_diffusion_benchmark(o, proj, cfg, *, fig_dir, verbose=True) -> dict:
    """Closed-form verification of the diffusion emulator (module doc, part 2)."""
    from FORCING import make_forcing_spec, estimate_eps_inj, rescale_to_eps_inj
    from STREAM import StreamMap, make_kernel_dequantizer
    from TG import generate_transition_pair_dataset, taylor_green_shell_basis
    from WSBG import WaveletDiffusionModel, make_conditional_sampler
    from DIFF import train_conditional_score_model, split_by_trajectory
    from GROUP import sigma_bar_linear_exact, oracle_from_diffusion, empirical_linear_floor
    from CERT import gain_verdict
    from PLOT import plot_tg_diffusion_benchmark

    J, jmin = o.J, o.jmin
    dt, nu, mu_d, stride = cfg["dt"], cfg["nu"], cfg["mu_d"], cfg["stride"]
    Delta = dt * stride
    spec = make_forcing_spec(J, jmin, 1.0, cfg["alpha_F"])
    spec = rescale_to_eps_inj(estimate_eps_inj(o, proj, spec, verbose=False), cfg["eps_inj"])
    step, det_step, noise = make_forced_stepper(
        o, proj, dt, nu, spec, mu_d=mu_d, dtype=np.float32, advection=False)
    stream = StreamMap(o, proj.filter_nyquist, dtype=jnp.float32)
    orbit = build_orbit_map_2d(J, jmin, n_comp=1)
    amb = jnp.result_type(float)
    stream_hp = StreamMap(o, proj.filter_nyquist, dtype=amb)
    noise_hp = make_forced_stepper(o, proj, dt, nu, spec, mu_d=mu_d, dtype=amb,
                                   advection=False)[2]
    sigma_bar_z, sigma_bar_r, blocks = sigma_bar_linear_exact(
        o, proj, stream_hp, noise_hp, dt, nu, mu_d, stride, orbit, dtype=amb,
        verbose=verbose)
    dequant = make_kernel_dequantizer(stream, sigma_bar_z, orbit)

    # -- exact reference --------------------------
    cf = LinearOUClosedForm(o, proj, spec, nu, mu_d)
    basis, lam_h, _ = taylor_green_shell_basis(o, proj, cfg["k0"])
    e_tg = basis[0]
    gamma = -(nu * lam_h - mu_d)
    q_tg = cf.mode_forcing((e_tg[0], e_tg[1]))
    E_inf = float(cf.energy_noise(np.inf)[0])
    Z_inf = float(cf.dissipation_noise(np.inf)[0])
    if verbose:
        print(f"    exact: eps_inj(closed form) = {cf.eps_inj:.5e} (spec {spec.eps_inj:.5e}); "
              f"<E>_inf = {E_inf:.4e}; identity 2 mu <E> + 2 nu <Z> = "
              f"{2 * mu_d * E_inf + 2 * nu * Z_inf:.5e}; gamma_TG = {gamma:.4f}")

    # -- data + training --------------------------
    ds = generate_transition_pair_dataset(
        o, spec, proj, dt, nu, mu_d, stride,
        n_trajectories=cfg["bench_n_trajectories"], n_burn=cfg["n_burn"],
        n_pairs_per_traj=cfg["bench_n_pairs_per_traj"], pair_gap=cfg["pair_gap"],
        seed=cfg["seed"] + 101, k_max=cfg["k_max"], spec_alpha=cfg["spec_alpha"],
        E0=E_inf, batch_size=cfg["dataset_gen_batch_size"], path=cfg["bench_data_path"],
        sigma_mode="B0", advection=False, verbose=verbose)
    idx_tr, idx_val = split_by_trajectory(ds["traj_id"], cfg["val_split"], cfg["seed"])
    diffusion = WaveletDiffusionModel(o, proj, beta=cfg["beta"], lambda0=cfg["lambda0"])
    diffusion.calibrate(ds, idx_tr, dequant=dequant, seed=cfg["seed"], verbose=verbose)
    diffusion.set_time_range(sigma_bar_z, snr_hi=cfg["snr_hi"], snr_lo=cfg["snr_lo"],
                             verbose=verbose)
    tr = train_conditional_score_model(
        ds, diffusion, sigma_bar_z, idx_tr, idx_val, n_epochs=cfg["bench_n_epochs"],
        batch_size=cfg["batch_size"], lr=cfg["lr"], seed=cfg["seed"],
        channels=cfg["channels"], weighting=cfg["weighting"],
        ema_decay=cfg["ema_decay"], patience=cfg["patience"],
        checkpoint_path=os.path.join(cfg["checkpoint_dir"],
                                     f"bench_linear_J{J}_jmin{jmin}.pkl"),
        dequant=dequant, verbose=verbose)
    floor = empirical_linear_floor(ds, diffusion, sigma_bar_z, idx_tr, idx_val,
                                   dequant=dequant, apply_F=tr["apply_F"],
                                   params=tr["params"], n_tau=cfg["floor_n_tau"],
                                   seed=cfg["seed"], verbose=verbose)
    oracle = oracle_from_diffusion(blocks, diffusion, sigma_bar_z,
                                   taus=floor["taus"], verbose=verbose)
    verdict = gain_verdict(floor["excess"], floor["excess_se"])        # expected '0'
    # certificate of the floor itself: on linear data it must reproduce the oracle
    z_floor = float((floor["gain_lin"] - oracle["gain"]) / max(floor["gain_lin_se"], 1e-300))
    if verbose:
        print(f"      floor gain {floor['gain_lin']:+.5f} +- {floor['gain_lin_se']:.5f} vs "
              f"exact oracle {oracle['gain']:+.5f} ({z_floor:+.1f} SE); network: "
              f"{verdict['verdict']}")

    # -- rollouts from the single TG mode ------------------
    generate = make_conditional_sampler(diffusion, tr["apply_F"], sigma_bar_z)
    phi = make_phi_delta(o, det_step, stride)
    EZ, vort = make_observables(o, stream)
    Wf = o.W_fwd
    A0 = np.sqrt(2.0 * E_inf)
    x0 = A0 * np.concatenate([(Wf @ e_tg[0] @ Wf.T).ravel(), (Wf @ e_tg[1] @ Wf.T).ravel()])
    n_lags = int(cfg["bench_n_lags"])
    K = int(cfg["bench_n_members"])
    snap_lags = [0, n_lags // 4, n_lags // 2, n_lags]
    kw = dict(n_steps=cfg["n_gen_steps"])
    emu = emulator_ensemble(generate, tr["params"], phi, x0, K, n_lags, EZ=EZ,
                            snap_lags=snap_lags, seed=cfg["seed"] + 5,
                            chunk=cfg["compare_chunk"], sampler_kw=kw, verbose=verbose)
    sol = solver_ensemble(o, step, stride, x0, K, n_lags, EZ=EZ, snap_lags=snap_lags,
                          seed=cfg["seed"] + 6, chunk=cfg["compare_chunk"], verbose=verbose)
    gauss = emulator_ensemble(generate, None, phi, x0, K, n_lags, EZ=EZ,
                              snap_lags=[n_lags], seed=cfg["seed"] + 7,
                              chunk=cfg["compare_chunk"], sampler_kw=kw, verbose=False)

    t = Delta * np.arange(n_lags + 1)
    E_th = np.exp(-2 * gamma * t) * 0.5 * A0 ** 2 + cf.energy_noise(t)
    amp_mean_th = A0 * np.exp(-gamma * t)
    amp_var_th = q_tg * (-np.expm1(-2 * gamma * t)) / (2 * gamma)

    def amps(snaps):
        return {k: np.array([l2_inner_w(o, x, x0 / A0) for x in X]) for k, X in snaps.items()}

    res = {"t": t, "E_th": E_th, "a_mean_th": amp_mean_th, "a_var_th": amp_var_th,
           "gamma": gamma, "q_tg": q_tg, "E_inf": E_inf, "Z_inf": Z_inf,
           "A0": A0, "K": K, "levels": floor["per_level"], "oracle": oracle,
           "floor": floor, "verdict": verdict, "z_floor": z_floor,
           "best_val": tr["best_val"], "val_base": tr["val_base"],
           "gain": tr["best_gain"], "gain_se": tr["best_gain_se"],
           "history": tr["history"]}
    for name, d in (("emulator", emu), ("solver", sol), ("gaussian_F0", gauss)):
        amp = amps(d["snaps"])
        zE = (d["E"].mean(1) - E_th)[1:] / (d["E"].std(1, ddof=1)[1:] / np.sqrt(K))
        za, rv = {}, {}
        for k, a in amp.items():
            if k == 0:
                continue
            za[k] = float((a.mean() - amp_mean_th[k]) / np.sqrt(amp_var_th[k] / K))
            rv[k] = float(a.var(ddof=1) / amp_var_th[k])
        res[name] = {"E": d["E"], "Z": d["Z"], "amp": amp,
                     "max_abs_zE": float(np.max(np.abs(zE))), "z_amp_mean": za,
                     "amp_var_ratio": rv}
    res["var_ratio_band"] = float(3.0 * np.sqrt(2.0 / (K - 1)))
    res["cmp"] = compare_ensembles(o, stream, sol, emu, Delta, nu, mu_d, spec.eps_inj,
                                   vort_fn=vort)
    # Student-t (K - 1 dof, the SE uses the sample sd), Bonferroni over lags,
    # family-wise level 1e-3
    from scipy.stats import t as student_t
    thr = float(student_t.ppf(1.0 - 1e-3 / (2.0 * max(n_lags, 1)), K - 1))
    e = res["emulator"]
    res["threshold"] = thr
    res["ok_curves"] = bool(e["max_abs_zE"] < thr
                            and all(abs(v) < thr for v in e["z_amp_mean"].values())
                            and all(abs(v - 1) < res["var_ratio_band"]
                                    for v in e["amp_var_ratio"].values()))
    res["ok_floor"] = bool(np.isfinite(z_floor) and abs(z_floor) < 3.0)
    res["ok_verdict"] = verdict["code"] == "0"
    res["ok"] = res["ok_curves"] and res["ok_floor"] and res["ok_verdict"]
    if verbose:
        print(f"    [TG diffusion benchmark] K={K}, T={t[-1]:.3g} ({gamma * t[-1]:.2f} TG decay times)")
        for name in ("solver", "gaussian_F0", "emulator"):
            d = res[name]
            print(f"      {name:>12s}: max|z| <E>(t) vs exact = {d['max_abs_zE']:.2f}; "
                  f"TG amplitude z(mean) = "
                  + ", ".join(f"{v:+.2f}" for v in d["z_amp_mean"].values())
                  + "; Var/Var_exact = "
                  + ", ".join(f"{v:.3f}" for v in d["amp_var_ratio"].values()))
        print(f"      curves (|z| < {thr:.2f}, |Var ratio - 1| < {res['var_ratio_band']:.3f}): "
              f"{'OK' if res['ok_curves'] else 'FAIL'};  floor = oracle: "
              f"{'OK' if res['ok_floor'] else 'FAIL'};  network at floor: "
              f"{'OK' if res['ok_verdict'] else 'FAIL'}  -> {'PASS' if res['ok'] else 'FAIL'}")
    plot_tg_diffusion_benchmark(res, save_path=os.path.join(fig_dir, "bench_tg_diffusion.png"))
    return res


def one_lag_energy_split(ops1d, stream, m_x, psi_y, psi_m):
    r"""E[E(y)|x] - E(m(x)) split into its two parts, from K iid samples of S y at fixed x.

        E(y) - E(m) = <m, c>_{L2} + E(c),     c = curl_h(r),  r = S y - S m.

    'cross' is carried by the conditional MEAN of c (its correlation with m);
    'self' by the conditional second moment of c.  y = m + curl_h(r) exactly
    (zero-mean V_F), so the split is an identity, not a model."""
    m = np.asarray(m_x, np.float64)
    r = np.asarray(psi_y, np.float64) - np.asarray(psi_m, np.float64)[None]
    c = np.asarray(stream.state_from_psi_w(jnp.asarray(r, stream.dtype)), np.float64)
    cross = np.array([l2_inner_w(ops1d, m, ci) for ci in c])
    self_ = np.array([0.5 * l2_inner_w(ops1d, ci, ci) for ci in c])
    se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v)))
    return {"cross": float(cross.mean()), "cross_se": se(cross),
            "self": float(self_.mean()), "self_se": se(self_),
            "total": float((cross + self_).mean()), "total_se": se(cross + self_)}