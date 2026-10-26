r"""FORCING.py -- stochastic forcing, Ekman drag, coefficient maps, energy budget.

Reduced SDE (scaling coordinates u_s; V = ker C the discrete divergence-free space)
    du_s = [ L u_s - P M2^{-1} N(u_s, u_s) ] dt + B_F dW,
    L    = nu (A_K (+) A_K) - mu_d I,        A_K := M^{-1} K,
    Q    := B_F B_F^T.

Two forcing parameterisations are implemented (``ForcingSpec.coords``):

'stream' (DEFAULT, production)
    B_F = F curl_h W Q0^{1/2},   W := W_inv (x) W_inv,   dW in R^{n^2},
    Q0 = diag(q) diagonal in STREAM-FUNCTION WAVELET coordinates.
    F curl_h maps every psi into V_F cap {zero mean flow}
    (STREAM.validate_stream_map), so (F1) and (F3) below hold by
    construction, no Leray projection of the noise is needed, and --
    the point of this choice -- the forcing covariance is diagonal in the
    SAME coordinates as the generative variable r = S(y - m), the Gaussian
    surrogate N(0, diag Sigma_bar) and the OU schedule.  For the linear
    (Stokes + drag) scheme,  S e^{tL} F curl_h = e^{tL} Pi_Ran  (all maps
    are Fourier multipliers), hence

        Sigma_lin = W_fwd Pi_Ran [ dt sum_k e^{k dt L} phi W Q0 W^T phi e^{k dt L} ] Pi_Ran W_fwd^T,

    which is NOT diagonal: Sigma_lin commutes with the dyadic shift group
    (ly), not with the wavelet coordinates.  Measured linear-Gaussian
    oracle gain of the DIAGONAL surrogate ('stream', , GROUP.py):
    0.032 at J=5 (smoke: alpha_F=7, Delta=5 dt) and 0.060 at J=7
    (production: alpha_F=5, Delta=100 dt), 62-75 % of it in the finest level
    J-1, with nearest-neighbour correlations up to -0.77 (J=5) / 0.95 (J=7)
    in the anisotropic blocks V_jmin (x) W_{J-1}: inside one octave the
    spectrum is steep, the band's energy sits at its lower edge, and the
    coefficients (sampled at the band's own Nyquist rate) are strongly
    correlated.  'stream' forcing is still the right choice (the forcing
    covariance itself is diagonal), but the network's training gain
    contains this LINEAR correlation and must be read against the oracle.

'velocity' (legacy, kept for A/B comparison)
    B_F = P F Pi_0 W Q0^{1/2},  dW in R^{2 n^2},  Q0 diagonal in VELOCITY
    wavelet coordinates.  S B_F is strongly non-diagonal in psi-wavelet
    coordinates (S ~ Delta^{-1} curl maps wavelets to vaguelettes).

P the G2-orthogonal Leray projector, F the Nyquist filter, Pi_0 the removal
of the spatial mean flow.  L, P, F, Pi_0, curl_h, S all lie in the
(circulant (x) circulant) algebra and commute; Q does NOT commute with L
(Q0 is wavelet-diagonal).

Time discretisation: exponential Euler-Maruyama
-----------------------------------------------
    u_{k+1} = Phi_dt(u_k) + sqrt(dt) phi(dt L) w_k,   w_k ~ N(0, Q),
    phi(dt L) := [ (e^{2 dt L} - 1) / (2 dt L) ]^{1/2}.

Linear part.  After K steps the covariance is
    C_K = dt sum_{k<K} e^{k dt L} phi Q phi e^{k dt L^T}.
For every weight W that commutes with L and is symmetric (G2, e^{sL}, any
f(L); Bubnov: L is symmetric in scaling coordinates), and every
eigenvector e of L,
    Tr(W C_K)       = Tr(W Q (I - e^{2 t L}) (-2L)^{-1}),       t = K dt,
    e^T W C_K W e   = f_t(l) e^T W Q W e,   f_t(l) = (1 - e^{2 l t})/(-2 l),
i.e. the kinetic energy, the enstrophy and the variance of any Fourier
(eigen)mode are  (identical to the continuous OU) at any dt, even
though Q does not commute with L.  The full covariance is  only to
O(dt^2 [L,[L,Q]]); every closed-form benchmark in this code base uses only
the  quantities (BENCH.py).  Plain Euler-Maruyama (phi = 1,
``stoch_integrator='em'``) has an O(dt |l|) error in all of them.

Weak order 1 overall (the advective part is frozen during the noise
increment).  The target of the generative model is a LAW.

Why forcing
-----------
Without it p_Delta(.|x) is a Dirac mass (no density, no score), the
TLM matrix is zero and unforced 2-D NS decays (no invariant measure).

(F1) Incompressibility: ran Q is contained in V BY CONSTRUCTION.
(F2) Trace class, uniformly in J:
       velocity: q_j = q0 2^{-j(1+alpha_F)};  stream: q_j = q0 2^{-j(3+alpha_F)},
     i.e. the same velocity spectrum (|grad psi_lambda| ~ 2^j).  Tr_{L2} Q
     < infinity uniformly in J iff alpha_F > 1.  Regularity of the
     stationary solution: E||u||_{H^s}^2 < infinity iff s < (1 + alpha_F)/2.
(F3) No mean-flow forcing.  For the Bubnov scheme the spatial mean of u_h
     is conserved by skew advection + projection; with zero-mean
     forcing and zero-mean data it is identically zero, and it is the one
     direction of V that is not a curl.  'stream' satisfies (F3)
     (the symbol of curl_h vanishes at k = 0); 'velocity' enforces it with Pi_0.

Ekman drag: folded into the linear symbol z = dt (nu Spec - mu_d).

Coefficient maps (single source of truth for the whole project)
---------------------------------------------------------------
level map   j(i)      forcing variances, OU rates, diagnostics.
orbit map   o(i)      orbits of the DYADIC SHIFT GROUP (shifts of the fine grid
                      by multiples of 2^(J-jmin) cells): the only translations
                      acting on wavelet coordinates as  permutations; the
                      forced dynamics are invariant under them.  One-point
                      statistics of the STATIONARY law are orbit-constant, so
                      orbit pooling is unbiased for them (NOT for statistics
                      conditioned on a fixed x).
``n_comp`` = 2 for the velocity state (u_w, v_w), 1 for a scalar field
(psi_w, the generative variable, and the 'stream' forcing input).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from OPERATORS import sandwich
from ODE import (make_scaling_pieces, build_scaling_basis_stepper,
                 scaling_kinetic_energy_fn, scaling_dissipation_fn,
                 make_wavelet_maps, _resolve_filter)



# 1.  Coefficient maps

def build_level_map_1d(J: int, jmin: int) -> np.ndarray:
    """Level label of every coefficient of the pywt 'periodization' layout.

    Layout [cA | cD_jmin | ... | cD_{J-1}], |cA| = 2^jmin, |cD_j| = 2^j.  The
    approximation block is labelled jmin - 1 (V_jmin sits strictly below
    W_jmin, and cA and cD_jmin must not share a rate).
    """
    n = 2 ** J
    lev = np.empty(n, dtype=int)
    lev[:2 ** jmin] = jmin - 1
    pos = 2 ** jmin
    for j in range(jmin, J):
        lev[pos:pos + 2 ** j] = j
        pos += 2 ** j
    return lev


def build_level_map_2d(J: int, jmin: int, n_comp: int = 2) -> np.ndarray:
    """j(i,k) = max(l(i), l(k)), repeated for each of the n_comp components.

    max, not min: a function fine along one axis has high-wavenumber content
    whatever its smoothness along the other.  Gives 6*4^j coefficients at
    level j >= jmin and 2*4^jmin at level jmin-1.
    """
    lev = build_level_map_1d(J, jmin)
    level2 = np.maximum(lev[:, None], lev[None, :]).reshape(-1)
    return np.concatenate([level2] * int(n_comp))


def build_orbit_map_1d(J: int, jmin: int) -> np.ndarray:
    """Orbit of each 1-D coefficient under shifts by multiples of 2^(J-jmin).

    Such a shift moves the level-j block by k 2^(j-jmin) positions, so the
    orbit of position p in that block is  p mod 2^(j-jmin).  2^(J-jmin)
    orbits, each of size 2^jmin.  (Checked against the numerical permutation
    by ``validate_orbit_map``.)
    """
    n = 2 ** J
    orb = np.empty(n, dtype=int)
    orb[:2 ** jmin] = 0
    pos, off = 2 ** jmin, 1
    for j in range(jmin, J):
        L, per = 2 ** j, 2 ** (j - jmin)
        orb[pos:pos + L] = off + (np.arange(L) % per)
        pos += L
        off += per
    return orb


def build_orbit_map_2d(J: int, jmin: int, n_comp: int = 2) -> np.ndarray:
    """Orbit id of every entry of the n_comp-component state:
    n_comp * 4^(J-jmin) orbits."""
    o1 = build_orbit_map_1d(J, jmin)
    no = int(o1.max()) + 1
    o2 = (o1[:, None] * no + o1[None, :]).reshape(-1)
    return np.concatenate([o2 + c * no * no for c in range(int(n_comp))])


def validate_orbit_map(ops1d) -> None:
    """The orbit labels must be invariant under the  dyadic permutation."""
    from ODE import dyadic_shift_permutation_1d
    J, jmin = ops1d.J, ops1d.jmin
    p = dyadic_shift_permutation_1d(ops1d.W_fwd, ops1d.W_inv, J, jmin, 1)
    o = build_orbit_map_1d(J, jmin)
    if not np.array_equal(o[p], o):
        raise RuntimeError("orbit map is not invariant under the dyadic shift")
    # and the group generated by p acts transitively on each orbit
    seen = np.zeros(len(o), bool)
    q = np.arange(len(o))
    for _ in range(2 ** jmin):
        seen[q] = True
        q = p[q]
        if np.array_equal(q, np.arange(len(o))):
            break
    if not seen.all():
        raise RuntimeError("dyadic shift group does not cover the orbits")


def orbit_pool(values: np.ndarray, orbit: np.ndarray) -> np.ndarray:
    """Mean of ``values`` (..., n_total) over each orbit -> (..., n_orbits)."""
    n_orb = int(orbit.max()) + 1
    cnt = np.bincount(orbit, minlength=n_orb).astype(float)
    flat = np.asarray(values, float).reshape(-1, values.shape[-1])
    out = np.stack([np.bincount(orbit, weights=r, minlength=n_orb) for r in flat]) / cnt
    return out.reshape(values.shape[:-1] + (n_orb,))


def orbit_representatives(orbit: np.ndarray) -> np.ndarray:
    r"""First coefficient index of every orbit.

    ness lemma (used by estimate_eps_inj and TLM.sigma_bar_linear).
    Let G ~ (Z_{2^jmin})^2 be the dyadic shift group, acting FREELY by
    permutations on input and output coordinates, and A a linear map that
    commutes with it (A g = g A).  For an input orbit O with representative
    k_O and any output index i,

        sum_{k in O} A_{ik}^2 = sum_{g in G} A_{g^{-1} i, k_O}^2
                              = sum_{i' in orb(i)} A_{i', k_O}^2 .

    Hence for Q0 = diag(q), q constant on orbits,
        diag(A Q0 A^T)_i = sum_O q_O  sum_{i' in orb(i)} A_{i', k_O}^2   (orbit-constant)
        Tr(M A Q0 A^T)   = sum_O q_O |O| (A e_{k_O})^T M (A e_{k_O})     (M G-invariant)
    i.e. ONE column per orbit gives both quantities:
    n_comp 4^(J-jmin) columns instead of n_comp 4^J (or Monte Carlo).
    """
    _, first = np.unique(np.asarray(orbit), return_index=True)
    return first



# 2.  Forcing specification

@dataclass
class ForcingSpec:
    """Level-diagonal forcing covariance  Q0 = sum_j q_j Pi_{B_j}.

    coords          : 'stream' (Q0 diagonal in psi-wavelet coordinates, n^2
                      inputs) or 'velocity' (u,v-wavelet coordinates, 2 n^2)
    q_per_level     : L2-normalised variance per coefficient, levels [jmin-1..J-1]
    q_diag          : variance of each DWT input coefficient, = 4^J q_{j(i)}
    level_per_coeff : level of each INPUT coefficient
    orbit_per_coeff : dyadic orbit of each INPUT coefficient
    diag_Q          : diag of Q in VELOCITY-wavelet coordinates ()
    eps_inj         : 1/2 dx^2 Tr(G2 Q), the energy injection rate ()
    eps_per_level   : energy injection by the inputs of each level (),
                      sum = eps_inj
    eta_per_level   : enstrophy injection by the inputs of each level (,
                      Z_gal of the forcing columns), sum = eta_inj
    k_f             : sqrt(eta_inj / eps_inj), ANGULAR forcing wavenumber
                      (Rayleigh quotient of the Galerkin dissipation form)
    """
    J: int
    jmin: int
    q0: float
    alpha_F: float
    levels: np.ndarray
    q_per_level: np.ndarray
    q_diag: np.ndarray
    level_per_coeff: np.ndarray
    orbit_per_coeff: np.ndarray
    coords: str = "stream"
    eps_inj: float = float("nan")
    eps_inj_se: float = float("nan")
    diag_Q: Optional[np.ndarray] = None
    eps_per_level: Optional[np.ndarray] = None
    eta_per_level: Optional[np.ndarray] = None

    @property
    def eta_inj(self) -> float:
        return float("nan") if self.eta_per_level is None else float(self.eta_per_level.sum())

    @property
    def k_f(self) -> float:
        """Angular forcing wavenumber sqrt(eta_inj / eps_inj) (scale-invariant)."""
        return float(np.sqrt(self.eta_inj / self.eps_inj))

    @property
    def n_in(self) -> int:
        return int(len(self.q_diag))

    def scaled(self, factor: float) -> "ForcingSpec":
        """Q0 -> factor * Q0.  eps_inj and diag_Q are linear in Q0."""
        return replace(
            self, q0=self.q0 * factor, q_per_level=self.q_per_level * factor,
            q_diag=self.q_diag * factor, eps_inj=self.eps_inj * factor,
            eps_inj_se=self.eps_inj_se * factor,
            diag_Q=None if self.diag_Q is None else self.diag_Q * factor,
            eps_per_level=None if self.eps_per_level is None else self.eps_per_level * factor,
            eta_per_level=None if self.eta_per_level is None else self.eta_per_level * factor)


def make_forcing_spec(J: int, jmin: int, q0: float = 1.0,
                      alpha_F: float = 1.5, coords: str = "stream") -> ForcingSpec:
    r"""Level-diagonal forcing in wavelet coordinates (L2-normalised units).

        velocity: q_j = q0 2^{-j(1+alpha_F)}   on (u_w, v_w), 2 n^2 inputs
        stream  : q_j = q0 2^{-j(3+alpha_F)}   on psi_w,      n^2 inputs

    Units.  With the l2-normalised pywt filters, the coefficient of a field
    on the L2-NORMALISED 2-D wavelet is dx * d (d the DWT coefficient of the
    scaling coefficients, dx = 2^-J), so the DWT coefficient carries variance
    q_diag = q_j / dx^2 = 4^J q_j and eps_inj is J-independent at fixed q0.
    Tr_{L2} of the velocity covariance ~ sum_j 2^{j(1-alpha_F)} in both
    parameterisations: finite uniformly in J iff alpha_F > 1.
    """
    if alpha_F <= 1.0:
        raise ValueError(
            f"alpha_F={alpha_F} <= 1: sum_j q_j |B_j| ~ sum_j 2^{{j(1-alpha_F)}} "
            "diverges and Tr Q is not bounded uniformly in J.")
    if coords not in ("stream", "velocity"):
        raise ValueError("coords must be 'stream' or 'velocity'")
    n_comp = 1 if coords == "stream" else 2
    expo = (3.0 if coords == "stream" else 1.0) + alpha_F
    levels = np.arange(jmin - 1, J)
    q_per_level = q0 * 2.0 ** (-levels * expo)
    lpc = build_level_map_2d(J, jmin, n_comp=n_comp)
    q_diag = (4.0 ** J) * q_per_level[lpc - (jmin - 1)]
    return ForcingSpec(J=int(J), jmin=int(jmin), q0=float(q0),
                       alpha_F=float(alpha_F), levels=levels,
                       q_per_level=q_per_level, q_diag=q_diag,
                       level_per_coeff=lpc,
                       orbit_per_coeff=build_orbit_map_2d(J, jmin, n_comp=n_comp),
                       coords=coords)



# 3.  Noise and the forced stepper

def remove_mean_flow(U, V):
    """Pi_0: remove the spatial mean of u_h, v_h.

    sum_k phi_{J,k} = 1, so subtracting a constant from the scaling
    coefficients subtracts the same constant from the field; the constant is
    an eigenvector of every circulant, so Pi_0 is G-orthogonal and commutes
    with P and with L.
    """
    return (U - jnp.mean(U, axis=(-2, -1), keepdims=True),
            V - jnp.mean(V, axis=(-2, -1), keepdims=True))


def linear_symbol(ops1d, projector, dt: float, nu: float, mu_d: float,
                  filter_nyquist=None, dtype=jnp.float32):
    """(z, modal) with z = dt (nu Spec - mu_d) and modal(R, f(z)) = f(dt L) R."""
    Spec, modal, _, _, _ = make_scaling_pieces(ops1d, projector, filter_nyquist,
                                               dtype=dtype)
    return dt * (nu * Spec - float(mu_d)), modal


def phi_noise_multiplier(z):
    """[(e^{2z} - 1)/(2z)]^{1/2}, with the removable singularity at z = 0."""
    y = 2.0 * z
    small = jnp.abs(y) < 1e-6
    ysafe = jnp.where(small, 1.0, y)
    return jnp.sqrt(jnp.where(small, 1.0 + 0.5 * y, jnp.expm1(ysafe) / ysafe))


def make_noise_fn(ops1d, project_fn, spec: ForcingSpec, dtype=jnp.float32,
                  modal=None, multiplier=None, filter_nyquist: bool = True):
    """key -> w = phi(dt L) B_F zeta  (SCALING-basis velocity pair), zeta ~ N(0, I).

        'stream'  : B_F zeta = F curl_h( W_inv (sqrt(q) zeta) W_inv^T )   (in V_F^0)
        'velocity': B_F zeta = P F Pi_0 W_inv (sqrt(q) zeta) W_inv^T

    Q0 is diagonal in WAVELET coordinates, so zeta is drawn there and
    synthesised with W_inv (white noise drawn in the scaling basis would
    give a different covariance).  With ``modal``/``multiplier`` the
    exponential-EM weight phi(dt L) is applied (it commutes with B_F's
    range structure).  Without them the function samples the CONTINUOUS
    covariance Q (used for eps_inj).

    Attributes (used by the  column estimators, single code path):
        noise.from_normals(zeta)  the linear map zeta -> w
        noise.n_in                dimension of zeta
        noise.orbit_in            dyadic orbit of each zeta coordinate
    """
    from STREAM import StreamMap
    n = 2 ** spec.J
    n2 = n * n
    w2s, _ = make_wavelet_maps(ops1d, dtype=dtype)
    sq = jnp.asarray(np.sqrt(spec.q_diag), dtype=dtype)
    Wi = jnp.asarray(ops1d.W_inv, dtype=dtype)

    if spec.coords == "stream":
        sm = StreamMap(ops1d, filter_nyquist=filter_nyquist, dtype=dtype)

        def base(zeta):
            Psi = sandwich(Wi, (zeta * sq).reshape(zeta.shape[:-1] + (n, n)), Wi.T)
            return sm.curl_s(Psi)
    else:
        def base(zeta):
            return project_fn(*remove_mean_flow(*w2s(zeta * sq)))

    def from_normals(zeta):
        U, V = base(zeta)
        if modal is not None:
            U, V = modal(U, multiplier), modal(V, multiplier)
        return U, V

    def noise(key):
        return from_normals(random.normal(key, (spec.n_in,), dtype=dtype))

    noise.from_normals = from_normals
    noise.n_in = spec.n_in
    noise.orbit_in = np.asarray(spec.orbit_per_coeff)
    noise.coords = spec.coords
    return noise


def make_forced_stepper(ops1d, projector, dt: float, nu: float,
                        spec: ForcingSpec, mu_d: float = 0.0,
                        advection_scheme: str = "if_rk4",
                        triad_rtol: float = 0.0, filter_nyquist=None,
                        dtype=np.float32, stoch_integrator: str = "exp",
                        advection: bool = True):
    """(step, det_step, noise): exponential Euler-Maruyama micro-step,

        step(UV, key) = det_step(UV) + sqrt(dt) * noise(key),

    noise = phi(dt L) B_F zeta ('exp', default) or B_F zeta ('em', plain
    Euler-Maruyama, comparison only).  det_step and noise are returned so
    that TLM linearises the maps the data were generated with.
    ``advection=False`` gives the linear (Stokes + drag) OU scheme, whose
    law is known in closed form (BENCH.py).
    """
    if stoch_integrator not in ("exp", "em"):
        raise ValueError("stoch_integrator must be 'exp' or 'em'")
    work = jnp.dtype(dtype)
    fn = _resolve_filter(projector, filter_nyquist)
    det_step = build_scaling_basis_stepper(
        ops1d, projector, dt, nu, advection_scheme=advection_scheme,
        triad_rtol=triad_rtol, filter_nyquist=filter_nyquist, dtype=work,
        mu_d=mu_d, advection=advection)
    _, _, _, project, _ = make_scaling_pieces(ops1d, projector, filter_nyquist,
                                              dtype=work)
    if stoch_integrator == "exp":
        z, modal = linear_symbol(ops1d, projector, dt, nu, mu_d,
                                 filter_nyquist, dtype=work)
        noise = make_noise_fn(ops1d, project, spec, dtype=work, modal=modal,
                              multiplier=phi_noise_multiplier(z).astype(work),
                              filter_nyquist=fn)
    else:
        noise = make_noise_fn(ops1d, project, spec, dtype=work, filter_nyquist=fn)
    sdt = jnp.asarray(np.sqrt(dt), dtype=work)

    def step(UV, key):
        U, V = det_step(UV)
        Nu, Nv = noise(key)
        return (U + sdt * Nu, V + sdt * Nv)

    return step, det_step, noise



# 4.  eps_inj and diag(Q)

def estimate_eps_inj(ops1d, projector, spec: ForcingSpec, filter_nyquist=None,
                     dtype=None, verbose: bool = True, **_ignored) -> ForcingSpec:
    r"""  eps_inj = 1/2 dx^2 Tr(G2 Q),  eta_inj = 1/2 Tr(H_Z Q)  and  diag(Q).

    eta_inj is the Ito injection rate of the Galerkin enstrophy Z_gal (H_Z its
    Hessian, ODE.scaling_dissipation_fn), i.e. sum over columns of Z_gal(col):
    the same ness lemma applies (the dissipation form is circulant, hence
    invariant under the dyadic shift group).  Both are also returned per
    INPUT level (``eps_per_level``, ``eta_per_level``), which is what
    CERT.forcing_regime_report needs -- no spectral model of the forcing.

    Both refer to the CONTINUOUS covariance (no phi(dt L) weight): eps_inj is
    the physical injection rate of the SDE, the one entering the stationary
    identity.  ONE column per dyadic-shift orbit of the forcing input
    (``orbit_representatives``): 4^(J-jmin) columns ('stream') or
    2 * 4^(J-jmin) ('velocity'); no Monte Carlo error at any J.
    """
    n = 2 ** spec.J
    dx = 1.0 / n
    orbit = np.asarray(spec.orbit_per_coeff)
    reps = orbit_representatives(orbit)
    size = np.bincount(orbit).astype(float)
    vel_orbit = build_orbit_map_2d(spec.J, spec.jmin, n_comp=2)
    n_vo = int(vel_orbit.max()) + 1
    work = jnp.result_type(float) if dtype is None else jnp.dtype(dtype)
    fn = _resolve_filter(projector, filter_nyquist)
    _, _, _, project, _ = make_scaling_pieces(ops1d, projector, filter_nyquist,
                                              dtype=work)
    noise = make_noise_fn(ops1d, project, spec, dtype=work, filter_nyquist=fn)
    energy = scaling_kinetic_energy_fn(ops1d, dx, dtype=work)
    enstrophy = scaling_dissipation_fn(ops1d, dx, dtype=work)
    _, s2w = make_wavelet_maps(ops1d, dtype=work)
    lev_in = np.asarray(spec.level_per_coeff) - (spec.jmin - 1)

    @jax.jit
    def column(zeta):
        U, V = noise.from_normals(zeta)
        return energy(U, V), enstrophy(U, V), s2w(U, V)

    eps_l = np.zeros(len(spec.levels))
    eta_l = np.zeros(len(spec.levels))
    dq = np.zeros(n_vo)
    for o_, k in enumerate(reps):
        e = np.zeros(spec.n_in)
        e[k] = 1.0
        Ek, Zk, w = column(jnp.asarray(e, work))
        eps_l[lev_in[k]] += size[o_] * float(Ek)    # = 1/2 dx^2 q_O c_O |O|
        eta_l[lev_in[k]] += size[o_] * float(Zk)
        dq += np.bincount(vel_orbit, weights=np.asarray(w, float) ** 2,
                          minlength=n_vo)
    eps = float(eps_l.sum())
    spec.eps_inj, spec.eps_inj_se = eps, 0.0
    spec.eps_per_level, spec.eta_per_level = eps_l, eta_l
    spec.diag_Q = dq[vel_orbit]
    if verbose:
        print(f"   [FORCING] coords={spec.coords}: eps_inj = {eps:.6e} (, "
              f"{len(reps)} orbit columns, {jnp.dtype(work).name}); "
              f"k_f = sqrt(eta_inj/eps_inj) = {spec.k_f:.2f} rad "
              f"({spec.k_f / (2 * np.pi):.2f} cycles)")
    return spec


def rescale_to_eps_inj(spec: ForcingSpec, eps_target: float) -> ForcingSpec:
    """Choose q0 so that eps_inj = eps_target (eps_inj is linear in q0).

    This is the physically meaningful knob: with drag-dominated large
    scales the stationary energy is <E> <= eps_inj / (2 mu_d).
    """
    if not np.isfinite(spec.eps_inj) or spec.eps_inj <= 0:
        raise RuntimeError("estimate eps_inj before rescaling")
    return spec.scaled(eps_target / spec.eps_inj)



# 5.  Energy budget

def check_energy_budget(E: np.ndarray, Z_gal: np.ndarray, dt_snap: float,
                        nu: float, mu_d: float, eps_inj: float,
                        window: int = 16, C: float = 4.0) -> dict:
    r"""Pathwise energy budget along ONE trajectory.

    Ito on E(x) = 1/2 dx^2 x^T G2 x (continuous SDE), with
    energy-neutral skew advection (Bubnov) and a G-orthogonal projector:

        dE = (-2 mu_d E - 2 nu Z_gal + eps_inj) dt + dM_t,

    Z_gal from ODE.scaling_dissipation_fn (the scheme's own dissipation), M
    a martingale with d<M>_t = dx^4 x^T G2 Q G2 x dt = O(E eps_inj) dt.  The
    window-averaged residual is therefore O(sqrt(Ebar eps_inj / T_window)).
    Returns the worst violation in units of C times that scale; >> 1 is a
    bug (forcing covariance, drag, projection or dissipation).  The
    snapshot/trapezoid evaluation of the window integrals adds an O(Delta)
    deterministic bias, largest for the stiff modes whose relaxation time is
    shorter than Delta; the STATIONARY identity below has no such bias for
    the linear part under the exponential integrator.
    """
    E = np.asarray(E, float)
    Z = np.asarray(Z_gal, float)
    m = int(window)
    if len(E) <= m:
        raise ValueError("series shorter than the averaging window")
    drift = (E[m:] - E[:-m]) / (m * dt_snap)
    # trapezoidal time average over each window, consistent with the drift
    cE = np.concatenate([[0.0], np.cumsum(0.5 * (E[1:] + E[:-1]))])
    cZ = np.concatenate([[0.0], np.cumsum(0.5 * (Z[1:] + Z[:-1]))])
    Ebar = (cE[m:] - cE[:-m]) / m
    Zbar = (cZ[m:] - cZ[:-m]) / m
    resid = drift + 2.0 * mu_d * Ebar + 2.0 * nu * Zbar - eps_inj
    bound = C * np.sqrt(np.maximum(Ebar * eps_inj / (m * dt_snap), 1e-300))
    ratio = np.abs(resid) / bound
    k = int(np.argmax(ratio))
    return {"max_ratio": float(ratio[k]), "argmax": k,
            "residual": float(resid[k]), "bound": float(bound[k]),
            "mean_residual": float(resid.mean()),
            "mean_ratio": float(ratio.mean()),
            "ok": bool(ratio.max() < 1.0)}      # C is already in the bound


def stationary_identity(E: np.ndarray, Z: np.ndarray, nu: float, mu_d: float,
                        eps_inj: float, n_eff: Optional[float] = None) -> dict:
    """2 mu_d <E> + 2 nu <Z_gal> = eps_inj under mu_infty (no free constant).

    Standard error.  If E, Z are (n_traj, n_t) arrays of INDEPENDENT
    trajectories, the SE is the spread of the per-trajectory time means
    divided by sqrt(n_traj): unbiased whatever the autocorrelation, no
    n_eff needed.  For a single series (1-D), the SE uses ``n_eff``
    (effective number of independent samples); for an OU-like energy with
    correlation time 1/(2 mu_d), n_eff ~ mu_d T.  Without n_eff the SE of a
    correlated series is UNDERSTATED and n_sigma is not meaningful.
    """
    E = np.asarray(E, float)
    Z = np.asarray(Z, float)
    lhs_s = 2.0 * mu_d * E + 2.0 * nu * Z
    lhs = float(lhs_s.mean())
    if lhs_s.ndim == 2 and lhs_s.shape[0] > 1:
        tm = lhs_s.mean(1)
        se = float(tm.std(ddof=1) / np.sqrt(len(tm)))
    else:
        v = lhs_s.ravel()
        ne = len(v) if n_eff is None else max(float(n_eff), 1.0)
        se = float(v.std(ddof=1) / np.sqrt(ne)) if len(v) > 1 else np.nan
    return {"lhs": lhs, "eps_inj": float(eps_inj),
            "rel_gap": (lhs - eps_inj) / eps_inj,
            "se": se, "n_sigma": abs(lhs - eps_inj) / max(se, 1e-300),
            "E_mean": float(E.mean()), "E_std": float(E.std()),
            "Z_mean": float(Z.mean())}