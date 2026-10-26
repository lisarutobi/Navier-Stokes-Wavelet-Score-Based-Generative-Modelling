r"""CERT.py -- a-priori and a-posteriori certificates of the forced run.

Every function here is pure NumPy and consumes objects the pipeline already
builds (ForcingSpec, WaveletOperators1D, dataset rows).  Nothing here changes
the solver or the model.

Conventions (used by every function below)
------------------------------------------
* Domain [0,1)^2.  Wavenumbers are ANGULAR (k = 2 pi x cycles) unless the
  name ends in ``_cyc``; spectral shells are indexed in cycles.  kappa is
  reserved for the surrogate ratio Sigma_bar / sigma^2 (WSBG).
* E_gal = 1/2 dx^2 x^T G2 x and Z_gal = -1/2 dx^2 x^T G2 (Delta_h (+) Delta_h) x are the
  energy and enstrophy of the SCHEME (ODE.scaling_kinetic_energy_fn,
  ODE.scaling_dissipation_fn).  The budget  d<E>/dt = eps_inj - 2 nu <Z_gal>
  - 2 mu_d <E_gal>  is  only with these, so every budget share below
  uses them, never a re-discretised spectral surrogate.
* chi_* are dimensionless mean-shift / non-Gaussianity ratios; rho(x) is the
  TLM variance ratio (TLM).

1.  ``forcing_regime_report``   (a priori, before any data are generated)
     per-level energy / enstrophy injection from the ForcingSpec
    (FORCING.estimate_eps_inj), the Rayleigh wavenumber of each level
    k_l^2 = eta_l / eps_l, and the linear viscous share
    nu k_l^2 / (nu k_l^2 + mu_d).  Gate: enstrophy injected at the
    finest level must be negligible (otherwise the grid-scale tail measured
    by ``resolution_report`` is forced, not resolved).

2.  ``galerkin_spectra`` / ``resolution_report``   (a posteriori)
    Shell spectra of E_gal, Z_gal and the Galerkin palinstrophy, obtained
    from the circulant symbols of G and K: they sum to E_gal, Z_gal
    (Parseval in scaling coordinates), include the corner shells
    n/2 < |k| <= n/sqrt(2), and need no "coefficients ~ point values"
    approximation.

3.  ``orbit_mc_floor`` / ``variance_test``   Monte-Carlo floors bootstrapped
    over iid ensemble members (no independence assumed inside an orbit).

4.  ``delta_regime``   Short-lag scaling of the non-Gaussian content of
    p_Delta(. | x): third cumulant ~ Delta^{3/2}, x-dependent mean shift
    ~ Delta^{5/2} (see the function docstring).

5.  ``gain_verdict``   reading of a training gain against the linear floor.

6.  Fixed-x statistics of the conditional law (solver vs emulator ensembles
    at the same x).  Every null / standard error is obtained by resampling
    or sign-flipping MEMBERS, which are iid; no assumption is made on the
    correlation between coefficients (an orbit is not an independent unit).
"""

from __future__ import annotations

import numpy as np



# 1.  Forcing regime (a priori,  injection)

def forcing_regime_report(spec, nu: float, mu_d: float, tol_top: float = 0.01,
                          verbose: bool = True) -> dict:
    r"""Where the forcing injects, and which sink removes it.

    ``spec`` must have passed through FORCING.estimate_eps_inj (it carries
    ``eps_per_level``, ``eta_per_level``).  Per input level l:

        eps_l, eta_l           energy / enstrophy injection rates
        k_l = sqrt(eta_l / eps_l)          (angular, Galerkin Rayleigh quotient)
        s_l = nu k_l^2 / (nu k_l^2 + mu_d)

    s_l is the  fraction of eps_l removed by viscosity in the linear
    (Stokes + drag) stationary state for a single Fourier shell, and a
    Jensen-type approximation for a level spanning several shells.  With
    advection it is a model: the inverse cascade moves energy to k < k_l,
    lowering the true viscous share.

    ok  <=>  eta_{J-1} / eta_inj < tol_top   (no enstrophy forced at the grid)
         and s_{l*} < 1/2 at the peak energy-injection level l*.
    """
    if spec.eps_per_level is None or spec.eta_per_level is None:
        raise RuntimeError("run FORCING.estimate_eps_inj on the spec first")
    levels = np.asarray(spec.levels)
    eps_l = np.asarray(spec.eps_per_level, float)
    eta_l = np.asarray(spec.eta_per_level, float)
    e_frac = eps_l / eps_l.sum()
    z_frac = eta_l / eta_l.sum()
    k_l = np.sqrt(eta_l / np.maximum(eps_l, 1e-300))
    visc = nu * k_l ** 2 / (nu * k_l ** 2 + mu_d)
    i_pk = int(np.argmax(e_frac))
    global_visc = float(np.sum(e_frac * visc))
    out = {"levels": levels, "injection_fraction": e_frac,
           "enstrophy_injection_fraction": z_frac,
           "wavenumber": k_l, "viscous_share": visc,
           "k_f": float(spec.k_f),
           "global_viscous_share": global_visc,
           "global_drag_share": 1.0 - global_visc,
           "injection_peak_level": int(levels[i_pk]),
           "top_level_enstrophy_injection": float(z_frac[-1]),
           "E_stationary_upper_bound": float(spec.eps_inj / (2.0 * mu_d)),
           "E_stationary_estimate": float(spec.eps_inj * (1.0 - global_visc) / (2.0 * mu_d))}
    reasons = []
    if out["top_level_enstrophy_injection"] >= tol_top:
        reasons.append(f"{z_frac[-1]:.3f} of the enstrophy is injected at the finest "
                       f"level (tol {tol_top}): increase alpha_F")
    if visc[i_pk] >= 0.5:
        reasons.append(f"viscous share {visc[i_pk]:.2f} >= 0.5 at the injection peak: "
                       f"no inverse cascade, drag is decorative (lower nu or raise J)")
    out["reasons"], out["ok"] = reasons, not reasons
    if verbose:
        print(f"  [forcing regime] alpha_F={spec.alpha_F:g}, nu={nu:g}, mu_d={mu_d:g}; "
              f"k_f = {out['k_f']:.1f} rad ({out['k_f'] / (2 * np.pi):.2f} cycles)")
        print("      level   k_l (rad)    energy inj  enstrophy inj  viscous share")
        for i, j in enumerate(levels):
            print(f"      j={j:<3d} {k_l[i]:9.1f}   {e_frac[i]:10.4f}   "
                  f"{z_frac[i]:12.4f}   {visc[i]:12.3f}")
        print(f"      predicted global split (linear): viscous {global_visc:.2f} / "
              f"drag {1 - global_visc:.2f}")
        print("      -> " + ("OK" if out["ok"] else "; ".join(reasons)))
    return out



# 2.  Spectra and resolution

def _circulant_symbol(C: np.ndarray) -> np.ndarray:
    """Real symbol of a SYMMETRIC circulant, aligned with np.fft.fft2 indices."""
    s = np.fft.fft(np.asarray(C, float)[0])
    if np.max(np.abs(s.imag)) > 1e-10 * (np.max(np.abs(s.real)) + 1e-300):
        raise ValueError("circulant is not symmetric")
    return s.real


def _shells(n: int) -> np.ndarray:
    kf = np.fft.fftfreq(n, d=1.0 / n)
    return np.rint(np.hypot(kf[:, None], kf[None, :])).astype(int)


def isotropic_spectra(U: np.ndarray, V: np.ndarray, reduce: str | None = "mean",
                      k_cut: int | None = None):
    r"""E(k) on integer shells, scaling coefficients treated as grid values.

    k: integer shell index (CYCLES; angular = 2 pi k).  All shells up to
    rint(n / sqrt(2)) are kept unless ``k_cut`` is given (the corners hold
    18% of the Fourier modes), so that
    sum_{k >= 1} E(k) = mean(|u|^2 + |v|^2)/2 - (mean-flow energy).

    This is a VISUAL estimator for plots.  Budget-consistent spectra are
    ``galerkin_spectra``.
    """
    U = np.asarray(U, float)
    V = np.asarray(V, float)
    n = U.shape[-1]
    uh = np.fft.fft2(U, axes=(-2, -1)) / n ** 2
    vh = np.fft.fft2(V, axes=(-2, -1)) / n ** 2
    e2 = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    kk = _shells(n)
    kmax = int(kk.max()) if k_cut is None else int(k_cut)
    lead = e2.shape[:-2]
    flat = e2.reshape(-1, n * n)
    Ek = np.stack([np.bincount(kk.ravel(), weights=r, minlength=kk.max() + 1)[: kmax + 1]
                   for r in flat])[:, 1:]
    k = np.arange(1, kmax + 1)
    if reduce == "mean":
        return k, Ek.mean(0)
    if reduce is None:
        return k, Ek.reshape(lead + (kmax,))
    raise ValueError(f"reduce must be 'mean' or None, got {reduce!r}")


def galerkin_spectra(states: np.ndarray, ops1d) -> dict:
    r"""Shell spectra of the scheme's own quadratic forms.

    With X the scaling coefficients of one component, X^ = fft2(X), g_hat,
    k_hat the symbols of G_vj, K_vj (K <= 0 is the Laplacian stencil) and
    l = k_hat / g_hat the symbol of Delta_h = M^{-1} K (Bubnov, M = G):

        E_gal = 1/2 dx^2 / n^2  sum_theta  g1 g2 |X^|^2
        Z_gal = 1/2 dx^2 / n^2  sum_theta  (-(l1 + l2)) g1 g2 |X^|^2
        P_gal = 1/2 dx^2 / n^2  sum_theta  (l1 + l2)^2  g1 g2 |X^|^2

    (P_gal = 1/2 ||grad omega_h||^2, the rate in dZ/dt|_visc = -2 nu P_gal.)
    Binned on integer shells, including the corners.  sum_k E(k) = E_gal and
    sum_k Z(k) = Z_gal to roundoff.  Bubnov only (Petrov has no symmetric
    energy form for Z).
    """
    if ops1d.test_space != "primal":
        raise ValueError("galerkin_spectra needs the Bubnov scheme (test_space='primal')")
    X = np.asarray(states, np.float64)
    n = 2 ** ops1d.J
    n2, dx = n * n, 1.0 / n
    Wi = np.asarray(ops1d.W_inv, float)
    g = _circulant_symbol(ops1d.G_vj)
    a = _circulant_symbol(ops1d.K_vj) / g
    gg = g[:, None] * g[None, :]
    lap = a[:, None] + a[None, :]                      # <= 0
    power = 0.0
    for c in range(2):
        S = np.einsum('ab,kbd,ed->kae', Wi, X[:, c * n2:(c + 1) * n2].reshape(-1, n, n), Wi)
        power = power + np.abs(np.fft.fft2(S, axes=(-2, -1))) ** 2
    w = 0.5 * dx * dx / n2 * power.mean(0)             # sample mean, (n, n)
    kk = _shells(n).ravel()
    kmax = int(kk.max())
    binE = np.bincount(kk, weights=(w * gg).ravel(), minlength=kmax + 1)
    binZ = np.bincount(kk, weights=(w * gg * (-lap)).ravel(), minlength=kmax + 1)
    binP = np.bincount(kk, weights=(w * gg * lap ** 2).ravel(), minlength=kmax + 1)
    return {"k": np.arange(kmax + 1), "E_k": binE, "Z_k": binZ, "P_k": binP,
            "E_gal": float(binE.sum()), "Z_gal": float(binZ.sum()),
            "P_gal": float(binP.sum())}


def resolution_report(states: np.ndarray, ops1d, nu: float, mu_d: float,
                      eps_inj: float, E_gal=None, Z_gal=None,
                      tail_tol: float = 0.05, verbose: bool = True) -> dict:
    r"""Resolution certificate from a sample of stationary velocity states.

    ``states`` (N, 2 n^2) flattened WAVELET velocity states (dataset rows);
    ``E_gal``, ``Z_gal`` optional arrays (e.g. ds['E_path'], ds['Z_path'])
    for the budget shares; if absent the sample itself is used.

    Criterion (the only gate):
        tail_Z := sum_{k > n/4} Z(k) / Z_gal < tail_tol
    i.e. the top octave INCLUDING the corners carries a negligible part of
    the enstrophy of the scheme.

    Reported, not gated: the Kraichnan wavenumber
        k_d = (eta / nu^3)^{1/6},   eta = 2 nu P_gal
    (angular; k_d_cyc = k_d / 2 pi).  It is an O(1)-constant estimate; for a
    k^{-3} enstrophy range between k_f and k_d the tail fraction is
    ~ ln(k_d / (n/4)) / ln(k_d / k_f), so a threshold on k_d is NOT
    independent of the tail criterion and is not used alongside it.
    """
    sp = galerkin_spectra(states, ops1d)
    n = 2 ** ops1d.J
    k, Zk, Ek = sp["k"], sp["Z_k"], sp["E_k"]
    k_q = n // 4
    tail_Z = float(Zk[k > k_q].sum() / max(sp["Z_gal"], 1e-300))
    tail_E = float(Ek[k > k_q].sum() / max(sp["E_gal"], 1e-300))
    eta = 2.0 * nu * sp["P_gal"]
    k_d = float((eta / nu ** 3) ** (1.0 / 6.0))
    k_d_cyc = k_d / (2.0 * np.pi)
    Eb = sp["E_gal"] if E_gal is None else float(np.mean(E_gal))
    Zb = sp["Z_gal"] if Z_gal is None else float(np.mean(Z_gal))
    out = {"k": k, "E_k": Ek, "Z_k": Zk,
           "tail_enstrophy_above_n_over_4": tail_Z,
           "tail_energy_above_n_over_4": tail_E,
           "k_dissipation": k_d_cyc, "k_nyquist": float(n // 2),
           "k_d_over_nyquist": k_d_cyc / (n // 2),
           "E_mean": Eb, "Z_mean": Zb,
           "budget_viscous_share": float(2.0 * nu * Zb / eps_inj),
           "budget_drag_share": float(2.0 * mu_d * Eb / eps_inj)}
    out["budget_closure"] = out["budget_viscous_share"] + out["budget_drag_share"]
    out["ok"] = bool(tail_Z < tail_tol)
    if verbose:
        print(f"  [resolution] <E_gal> = {Eb:.4e}, <Z_gal> = {Zb:.4e}; budget shares: "
              f"viscous {out['budget_viscous_share']:.2f}, drag "
              f"{out['budget_drag_share']:.2f} (sum {out['budget_closure']:.3f}; "
              f"= 1 only at stationarity, 1 - sum = (d<E>/dt) / eps_inj)")
        print(f"      Galerkin enstrophy above k = n/4 (incl. corners): {tail_Z:.3f} "
              f"(tol {tail_tol}); energy: {tail_E:.4f}")
        print(f"      k_d_cyc = {k_d_cyc:.1f} cycles vs Nyquist {n // 2} "
              f"(ratio {out['k_d_over_nyquist']:.2f}, informative) -> "
              + ("resolved" if out["ok"] else "UNDER-RESOLVED: enstrophy tail at the grid scale"))
    return out



# 3.  Monte-Carlo floors bootstrapped over members

def orbit_mc_floor(samples: np.ndarray, orbit: np.ndarray, n_boot: int = 400,
                   seed: int = 0) -> dict:
    r"""Sampling floor of  median |log(Var_emp / Var_true)|  over orbits.

    ``samples`` (K, d) iid ensemble members at a FIXED x; ``orbit`` (d,)
    orbit labels.  Members are independent, coefficients inside an orbit
    are not: bootstrapping over MEMBERS gives the null distribution of the
    orbit-pooled variance without any within-orbit independence assumption.
    ``naive_independent_floor`` = 0.674 sqrt(2 / (K |orbit|)) is reported
    only to show by how much independence would understate the floor.
    """
    Y = np.asarray(samples, np.float64)
    orb = np.asarray(orbit)
    K = Y.shape[0]
    n_orb = int(orb.max()) + 1
    cnt = np.bincount(orb, minlength=n_orb).astype(float)

    def pooled_var(Z):
        return np.bincount(orb, weights=Z.var(0, ddof=1), minlength=n_orb) / cnt

    ref = pooled_var(Y)
    rng = np.random.default_rng(seed)
    med, mx = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        lr = np.log(np.maximum(pooled_var(Y[idx]), 1e-300) / np.maximum(ref, 1e-300))
        med[b] = np.median(np.abs(lr))
        mx[b] = np.max(np.abs(lr))
    naive = 0.674 * np.sqrt(2.0 / (K * cnt.min()))
    return {"median": float(np.median(med)), "median_p95": float(np.quantile(med, 0.95)),
            "max_p95": float(np.quantile(mx, 0.95)),
            "naive_independent_floor": float(naive),
            "inflation": float(np.median(med) / max(naive, 1e-300)),
            "K": int(K), "orbit_size": int(cnt.min())}


def variance_test(emp_var_orbit: np.ndarray, model_var_orbit: np.ndarray,
                  floor: dict, verbose: bool = True, label: str = "") -> dict:
    """Orbit-pooled empirical vs model variance, against the bootstrap floor."""
    lr = np.log(np.maximum(np.asarray(emp_var_orbit, float), 1e-300)
                / np.maximum(np.asarray(model_var_orbit, float), 1e-300))
    med = float(np.median(np.abs(lr)))
    out = {"median_abs_log_ratio": med, "max_abs_log_ratio": float(np.max(np.abs(lr))),
           "floor_median": floor["median"], "floor_median_p95": floor["median_p95"],
           "significant": bool(med > floor["median_p95"])}
    if verbose:
        print(f"  [variance test{' ' + label if label else ''}] "
              f"median |log Var_emp/Var_model| = {med:.3f}; bootstrap floor "
              f"{floor['median']:.3f} (95% {floor['median_p95']:.3f}; naive "
              f"independence floor {floor['naive_independent_floor']:.3f}, "
              f"understated by {floor['inflation']:.1f}x) -> "
              + ("REAL discrepancy" if out["significant"] else "within Monte-Carlo noise"))
    return out



# 4.  Short-lag non-Gaussianity of p_Delta(. | x)

def delta_regime(Delta: float, eps_inj: float, k_f: float, u_rms: float,
                 n_members: int, z_target: float = 3.0, verbose: bool = True) -> dict:
    r"""Order-of-magnitude size of the non-Gaussian content at lag Delta.

    Write y = m + xi, m = Phi_Delta(x), and expand xi = xi_0 + xi_1 + xi_2 + ...
    for Delta << min(1/mu_d, 1/(nu k_f^2)):

      xi_0 = int e^{(Delta-s)L} B_F dW          Gaussian, E||xi_0||^2 ~ 2 eps Delta
      xi_1 = -int P DN(m) xi_0 ds               Gaussian, relative size k_f U Delta
      xi_2 = -int P N(xi_0, xi_0) ds            quadratic, size k_f eps Delta^2

    (a) Third cumulant (leading non-Gaussian signature):
          kappa_3 / sd^3 ~ k_f eps Delta^2 * (2 eps Delta) / (2 eps Delta)^{3/2}
          chi_3 = k_f sqrt(eps / 2) Delta^{3/2}.
    (b) Conditional MEAN shift.  E xi_2 = -int P div E[xi_0 (x) xi_0] ds is
        x-INDEPENDENT, and it VANISHES for a translation-invariant forcing
        covariance (the covariance of xi_0 is then constant in space).  With
        the wavelet-diagonal Q it is a lattice term, periodic on the dyadic
        grid 2^{-jmin}: measure it as the x-average of E[y - m | x], do not
        count it as learnable non-Gaussianity.  The first x-DEPENDENT mean
        term is E N(xi_0, xi_1) + E N(xi_1, xi_0):
          chi_mean = (k_f U Delta) chi_3 = k_f^2 U sqrt(eps / 2) Delta^{5/2}.

    ``k_f`` angular (use ForcingSpec.k_f), ``u_rms`` = sqrt(2 <E>).  Detection
    with K members needs chi > z_target / sqrt(K).  All constants are O(1);
    GRID.run_rung_lag_grid measures the actual ratios.
    """
    need = float(z_target / np.sqrt(n_members))
    c3 = k_f * np.sqrt(eps_inj / 2.0)
    cm = k_f ** 2 * u_rms * np.sqrt(eps_inj / 2.0)
    chi_3 = float(c3 * Delta ** 1.5)
    chi_mean = float(cm * Delta ** 2.5)
    out = {"chi_3": chi_3, "chi_mean": chi_mean, "detection_threshold": need,
           "detectable_cumulant": bool(chi_3 > need),
           "detectable_mean_shift": bool(chi_mean > need),
           "Delta": float(Delta),
           "Delta_required_cumulant": float((need / c3) ** (2.0 / 3.0)),
           "Delta_required_mean_shift": float((need / cm) ** 0.4),
           "Delta_over_tau_eddy": float(k_f * u_rms * Delta)}
    if verbose:
        print(f"  [Delta regime] Delta = {Delta:.3g} ({out['Delta_over_tau_eddy']:.2f} "
              f"eddy times 1/(k_f U)); K = {n_members}, threshold {need:.3f}")
        print(f"      third cumulant chi_3 = {chi_3:.3f} -> "
              f"{'detectable' if out['detectable_cumulant'] else 'below noise'} "
              f"(needs Delta >= {out['Delta_required_cumulant']:.3g})")
        print(f"      x-dependent mean shift chi_mean = {chi_mean:.3f} -> "
              f"{'detectable' if out['detectable_mean_shift'] else 'below noise'} "
              f"(needs Delta >= {out['Delta_required_mean_shift']:.3g})")
    return out



# 5.  Verdict on a training gain against the linear-Gaussian floor

def gain_verdict(excess: float, excess_se: float, z_thr: float = 3.0) -> dict:
    r"""Three-way reading of the PAIRED excess of the network over the
    empirical linear floor (GROUP.empirical_linear_floor):

        excess = loss_linear - loss_network,   z = excess / SE (trajectory clusters)

        z >  z_thr   '+': the network extracts x-DEPENDENT structure that no
                     x-independent linear map can (mean shift, x-dependent /
                     cross-scale covariance, or non-Gaussianity -- a loss
                     cannot tell these apart; decide at fixed x)
        |z| <= z_thr '0': nothing beyond the best x-independent linear predictor
        z < -z_thr   '-': the network is worse than a closed-form linear map
                     (under-trained); nothing can be concluded
    """
    if excess is None or not np.isfinite(excess):
        return {"z": float("nan"), "code": "?", "verdict": "no floor"}
    if not np.isfinite(excess_se) or excess_se <= 0.0:
        return {"z": float("nan"), "code": "?",
                "verdict": "UNDETERMINED: no usable standard error (< 5 validation "
                           "trajectories: a cluster SE on G - 1 < 4 dof is refused)"}
    z = float(excess / excess_se)
    if z > z_thr:
        return {"z": z, "code": "+", "verdict": "ABOVE the empirical linear floor: "
                "x-dependent structure learned (Gaussian or not: see fixed-x tests)"}
    if z < -z_thr:
        return {"z": z, "code": "-", "verdict": "BELOW the empirical linear floor: "
                "under-trained; no conclusion possible"}
    return {"z": z, "code": "0", "verdict": "AT the empirical linear floor: nothing "
            "beyond an x-independent linear predictor"}


# 6.  Fixed-x statistics of the conditional law (members are the iid units)

def _pooled_var(Y: np.ndarray, orb: np.ndarray) -> np.ndarray:
    """Orbit-pooled member variance, expanded back to coefficients."""
    n_orb = int(orb.max()) + 1
    v = np.bincount(orb, weights=Y.var(0, ddof=1), minlength=n_orb)
    return (v / np.bincount(orb, minlength=n_orb))[orb]


def _resample(Y, rng):
    return Y[rng.integers(0, Y.shape[0], Y.shape[0])]


def mean_shift_diagnostic(psi_samples: np.ndarray, psi_m: np.ndarray,
                          orbit: np.ndarray, n_boot: int = 200, seed: int = 0) -> dict:
    r"""Conditional mean shift at a FIXED x, from K iid members of S y.

        shift_i = (E[S y | x]_i - (S m(x))_i) / sd_i,    sd_i orbit-pooled,
        chi^2   = mean_i shift_i^2,     E[chi^2 | no shift] = 1/K,
        chi_NG  = sqrt(max(chi^2 - 1/K, 0)),   z = (chi^2 - 1/K) / SE_boot.

    SE_boot resamples MEMBERS (no assumption on the correlation between
    coefficients, as in orbit_mc_floor).  It is taken under the alternative,
    so z is conservative.  The sd is orbit-pooled so that a per-coefficient
    sd from K members (relative error sqrt(1/2K)) does not eat the shift.
    """
    Y = np.asarray(psi_samples, np.float64)
    orb = np.asarray(orbit)
    pm = np.asarray(psi_m, np.float64)
    K = Y.shape[0]

    def chi2(Z):
        return float(np.mean((Z.mean(0) - pm) ** 2 / np.maximum(_pooled_var(Z, orb), 1e-300)))

    c2 = chi2(Y)
    rng = np.random.default_rng(seed)
    se = float(np.std([chi2(_resample(Y, rng)) for _ in range(n_boot)], ddof=1))
    sd = np.sqrt(np.maximum(_pooled_var(Y, orb), 1e-300))
    raw = Y.mean(0) - pm
    return {"chi": float(np.sqrt(c2)), "raw": raw, "sd": sd, "shift": raw / sd,
            "chi_mc": float(1.0 / np.sqrt(K)),
            "chi_ng": float(np.sqrt(max(c2 - 1.0 / K, 0.0))),
            "z": float((c2 - 1.0 / K) / max(se, 1e-300)), "se_boot": se, "K": int(K)}


def conditional_mean_bias(psi_solver: np.ndarray, psi_emulator: np.ndarray,
                          psi_m: np.ndarray, orbit: np.ndarray, n_boot: int = 200,
                          seed: int = 0) -> dict:
    r"""Error of the emulator's conditional MEAN beyond Monte Carlo, at fixed x.

    In solver-sd units (sd_s orbit-pooled from the solver ensemble),
        b2 = mean_i ((ybar_e - ybar_s)_i / sd_s,i)^2 - (1/K_s + v/K_e),
        v  = mean_i Var_e,i / Var_s,i       (emulator MC variance, solver units),
    so E[b2] = 0 when the two conditional means agree.  bias = sqrt(max(b2, 0)).
    The F = 0 sampler has zero shift: its bias is the solver's chi_NG.
    z = b2 / SE_boot, SE by resampling the members of both ensembles.
    """
    Ys = np.asarray(psi_solver, np.float64)
    Ye = np.asarray(psi_emulator, np.float64)
    orb = np.asarray(orbit)

    def b2(A, B):
        vs = np.maximum(_pooled_var(A, orb), 1e-300)
        v = float(np.mean(_pooled_var(B, orb) / vs))
        return float(np.mean((B.mean(0) - A.mean(0)) ** 2 / vs)
                     - (1.0 / A.shape[0] + v / B.shape[0]))

    val = b2(Ys, Ye)
    rng = np.random.default_rng(seed)
    se = float(np.std([b2(_resample(Ys, rng), _resample(Ye, rng))
                       for _ in range(n_boot)], ddof=1))
    return {"b2": val, "bias": float(np.sqrt(max(val, 0.0))), "se_boot": se,
            "z": float(val / max(se, 1e-300))}


def shift_correlation(psi_solver: np.ndarray, psi_emulator: np.ndarray,
                      psi_m: np.ndarray, orbit: np.ndarray, n_null: int = 200,
                      seed: int = 0) -> dict:
    r"""corr(solver shift, emulator shift) at fixed x, with a sign-flip null.

    Null: an emulator that knows nothing of the shift.  Its mean is replaced
    by  psi_m + mean_k eps_k (Y_e,k - ybar_e),  eps_k = +-1 iid: same member
    covariance as the emulator, zero shift.  null_sd = sd of the correlation
    over n_null flips; z = corr / null_sd.  No assumption on the spatial
    correlation of the coefficients.
    """
    Ys = np.asarray(psi_solver, np.float64)
    Ye = np.asarray(psi_emulator, np.float64)
    orb = np.asarray(orbit)
    pm = np.asarray(psi_m, np.float64)
    sd = np.sqrt(np.maximum(_pooled_var(Ys, orb), 1e-300))
    sh_s = (Ys.mean(0) - pm) / sd
    sh_e = (Ye.mean(0) - pm) / sd
    corr = float(np.corrcoef(sh_s, sh_e)[0, 1])
    rng = np.random.default_rng(seed)
    C = Ye - Ye.mean(0)
    null = [float(np.corrcoef(sh_s, (rng.choice((-1.0, 1.0), C.shape[0]) @ C)
                              / C.shape[0] / sd)[0, 1]) for _ in range(n_null)]
    null_sd = float(np.std(null, ddof=1))
    return {"corr": corr, "null_sd": null_sd, "z": corr / max(null_sd, 1e-300),
            "shift_solver": sh_s, "shift_emulator": sh_e}


def within_orbit_variance_structure(psi_solver: np.ndarray, psi_emulator: np.ndarray,
                                    orbit: np.ndarray) -> dict:
    r"""Un-pooled test at fixed x: does the emulator reproduce WHERE, inside an
    orbit, the conditional variance is large?

        ell(Y)_i = log Var(Y)_i - mean_{i' in orb(i)} log Var(Y)_{i'},
        ceiling  = corr(ell(Y_s[:h]), ell(Y_s[h:]))      split-half reliability,
        score    = corr(ell(Y_s[:h]), ell(Y_e[:h]))      same ensemble size h.

    Every orbit-pooled test is blind to this structure.  The F = 0 surrogate
    (orbit-constant variance) scores 0 by construction, so score / ceiling is
    the fraction of the reproducible within-orbit structure the network adds.
    """
    Ys = np.asarray(psi_solver, np.float64)
    Ye = np.asarray(psi_emulator, np.float64)
    orb = np.asarray(orbit)
    n_orb = int(orb.max()) + 1
    cnt = np.bincount(orb, minlength=n_orb)

    def ell(Y):
        v = np.log(np.maximum(Y.var(0, ddof=1), 1e-300))
        return v - (np.bincount(orb, weights=v, minlength=n_orb) / cnt)[orb]

    h = min(Ys.shape[0], Ye.shape[0]) // 2
    ceil = float(np.corrcoef(ell(Ys[:h]), ell(Ys[h:2 * h]))[0, 1])
    score = float(np.corrcoef(ell(Ys[:h]), ell(Ye[:h]))[0, 1])
    return {"score": score, "ceiling": ceil, "h": int(h)}
