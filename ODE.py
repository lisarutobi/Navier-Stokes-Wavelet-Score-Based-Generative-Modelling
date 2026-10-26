"""ODE.py -- time integration of the wavelet-Galerkin 2-D Navier-Stokes system.

Spaces, bases, test-space conventions and the connection coefficients are
documented ONCE, in OPERATORS.py.  This module contains:
  * the scaling-basis building blocks (modal functions of L, Leray projection),
  * the banded skew-symmetric advection,
  * the single time integrator ``build_scaling_basis_stepper``,
  * energy / dissipation functionals consistent with the scheme,
  * the NumPy reference advection and the spatial validation tests.
"""
import os
if os.environ.get("NSWSGM_X64", "0") == "1":
    import jax
    jax.config.update("jax_enable_x64", True)
else:
    import jax

import numpy as np
from scipy.linalg import eigh

import jax.numpy as jnp
from jax import lax

from OPERATORS import (samples_to_coeffs, galerkin_multiplier,
                       mm, sandwich, build_2d_operators, HIGHEST)



#  Scaling-basis, banded time stepping

# Every operator here is CIRCULANT with 11-17 nonzeros per row in the SCALING
# basis, so it is applied as a sum of rolls, not as a dense n x n matmul.
# The DWT (for Petrov) is a similarity, so stepping in the scaling basis and
# applying W_fwd only at snapshot time.  This is the ONLY time
# integrator in the code base.

def _resolve_dtype(dtype):
    """Working floating-point type for the JAX steppers.

    ``None`` means "whatever JAX is configured for": float64 when
    NSWSGM_X64=1, float32 otherwise.  Pass an explicit dtype when the CALLER
    owns the precision -- the dataset generator runs in float32 on purpose
    even under NSWSGM_X64=1, because the Richardson validation needs float64
    but a 27 GB float64 rollout does not.

    Every precomputed constant inside a stepper is cast to this type.  If it
    were left to ambient promotion, a float32 state entering an operator whose
    constants are float64 comes back float64, and jax.lax.scan rejects the
    carry (`carry input and carry output must have equal types`).
    """
    if dtype is None:
        return jnp.result_type(float)
    return jnp.dtype(dtype)


def _circulant_band(C, tol=1e-14):
    """Signed offsets and values of a circulant built by circulant_from."""
    row = np.asarray(C, dtype=float)[0, :]
    n = len(row)
    nz = np.nonzero(np.abs(row) > tol * np.abs(row).max())[0]
    off = np.where(nz > n // 2, nz - n, nz).astype(int)
    order = np.argsort(off)
    return [int(p) for p in off[order]], row[nz][order]


def make_banded_circulant(C, dtype=None):
    """(C@X, C.T@X, X@C, X@C.T) at O(b n^2) instead of O(n^3)."""
    off, val = _circulant_band(C)
    val = jnp.asarray(val, dtype=_resolve_dtype(dtype))

    def left(X):    return sum(val[i] * jnp.roll(X, -p, axis=0) for i, p in enumerate(off))
    def left_T(X):  return sum(val[i] * jnp.roll(X,  p, axis=0) for i, p in enumerate(off))
    def right(X):   return sum(val[i] * jnp.roll(X,  p, axis=1) for i, p in enumerate(off))
    def right_T(X): return sum(val[i] * jnp.roll(X, -p, axis=1) for i, p in enumerate(off))
    return left, left_T, right, right_T


def _resolve_filter(projector, filter_nyquist):
    """Single source of truth for the velocity space.

    ``None`` inherits the projector's setting.  An EXPLICIT value that
    disagrees with the projector is an error: the initial conditions (made
    with the projector) and the time stepping would otherwise live in
    different discrete spaces.
    """
    proj_flag = None if projector is None else bool(getattr(projector, "filter_nyquist", False))
    if filter_nyquist is None:
        return bool(proj_flag) if proj_flag is not None else False
    if proj_flag is not None and bool(filter_nyquist) != proj_flag:
        raise ValueError(
            f"filter_nyquist={filter_nyquist} disagrees with the projector "
            f"(filter_nyquist={proj_flag}); pass filter_nyquist=None to inherit.")
    return bool(filter_nyquist)


def make_scaling_pieces(ops1d, projector=None, filter_nyquist=None, dtype=None):
    """Scaling-basis building blocks, for M = I (Petrov) or M = G (Bubnov).

    Delta_h := M^{-1} K (1-D weak Laplacian), with generalised eigenpairs
    K V = M V diag(lam_h), V^T M V = I, hence V^{-1} = V^T M:

    modal(R, f(Spec)) = f(Delta_h (+) Delta_h) R,   Spec = lam_h (+) lam_h;
    project(U, V)     = P (Pi_N (U, V)),  P = I - G2^{-1} C^T S^+ C,
                        C = [D (x) M, M (x) D],  S = Ap (x) Bp + Bp (x) Ap
                        (see OPERATORS.LerayProjector).  Similar (W_fwd) to
                        LerayProjector.project in the wavelet basis.
    """
    filter_nyquist = _resolve_filter(projector, filter_nyquist)
    dt_ = _resolve_dtype(dtype)
    n = ops1d.M_vj.shape[0]
    petrov = ops1d.test_space == "dual"
    Mv = np.eye(n) if petrov else np.asarray(ops1d.M_vj, float)
    Dv, Kv, Gv = (np.asarray(a, float) for a in (ops1d.D_vj, ops1d.K_vj, ops1d.G_vj))
    Gi_np = np.linalg.inv(Gv); Gi_np = 0.5 * (Gi_np + Gi_np.T)
    lam_h, V_mod = eigh(Kv, Mv, check_finite=False)
    Vj = jnp.asarray(V_mod, dtype=dt_)                   # modal synthesis V
    Vinvj = jnp.asarray(V_mod.T @ Mv, dtype=dt_)         # modal analysis V^{-1} = V^T M
    Spec = jnp.asarray(lam_h[:, None] + lam_h[None, :], dtype=dt_)
    Ap = Dv @ Gi_np @ Dv.T; Ap = 0.5 * (Ap + Ap.T)
    Bp = Mv @ Gi_np @ Mv.T; Bp = 0.5 * (Bp + Bp.T)
    mu, Y = eigh(Ap, Bp, check_finite=False)
    rcond = getattr(projector, "rcond", None)
    tol = (1e-10 if rcond is None else rcond) * (float(np.max(np.abs(mu))) + 1e-300)
    den = mu[:, None] + mu[None, :]; good = den > tol
    invd = jnp.asarray(np.where(good, 1.0 / np.where(good, den, 1.0), 0.0), dtype=dt_)
    Yj, Gi = jnp.asarray(Y, dtype=dt_), jnp.asarray(Gi_np, dtype=dt_)
    Dl, DlT, Dr, DrT = make_banded_circulant(Dv, dtype=dt_)
    if petrov:
        Ml = MlT = Mr = MrT = (lambda X: X); Mi = None
    else:
        Ml, MlT, Mr, MrT = make_banded_circulant(Mv, dtype=dt_)
        Mi_np = np.linalg.inv(Mv); Mi = jnp.asarray(0.5 * (Mi_np + Mi_np.T), dtype=dt_)
    eN = jnp.asarray(((-1.0) ** np.arange(n)) / np.sqrt(n), dtype=dt_)

    # Every dense contraction goes through mm/sandwich (HIGHEST precision):
    # a TF32 pass would break the ness of the projector.
    def denyq(X):
        X = X - jnp.outer(eN, mm(eN, X))
        return X - jnp.outer(mm(X, eN), eN)
    def modal(R, mult):
        return sandwich(Vj, sandwich(Vinvj, R, Vinvj.T) * mult, Vj.T)
    def modal2(R1, m1, R2, m2):
        return sandwich(Vj, sandwich(Vinvj, R1, Vinvj.T) * m1
                        + sandwich(Vinvj, R2, Vinvj.T) * m2, Vj.T)
    def project(U, V):
        if filter_nyquist:
            U, V = denyq(U), denyq(V)
        R = Dl(MrT(U)) + Ml(DrT(V))                       # D U M^T + M V D^T
        q = sandwich(Yj, sandwich(Yj.T, R, Yj) * invd, Yj.T)
        return (U - sandwich(Gi, DlT(Mr(q)), Gi),
                V - sandwich(Gi, MlT(Dr(q)), Gi))
    def mass_inv(R):
        return R if Mi is None else sandwich(Mi, R, Mi)
    return Spec, modal, modal2, project, mass_inv

def make_adv_scaling(Lam, Gam, triad_rtol=0.0, dtype=None):
    """Banded-circulant skew advection on SCALING-basis fields.

    Identical mathematics to compute_advection_ (NumPy reference):

        N_s = sc * sum_{m,k} sX_m sY_k (aX_m * A aY_k^T) o (bX_m * B bY_k^T),

    where (c * X) denotes the 17-tap circulant filter X_i -> sum_p c[p] X_{i+p}
    along one axis.  Cost per pair 2 Rx Ry b n^2 instead of 2 Rx Ry n^3.

    triad_rtol > 0 truncates the triad SVD.  sigma/sigma_0 for Lambda_skew is
    [1, .60, .093, .056, .020, 7.0e-3, 9.0e-4, 2.8e-4, 1.2e-5, 8.6e-6, 3.2e-8,
    1.9e-8], so rtol = 1e-6 costs ~5e-7 relative error in N -- far ABOVE the
    spatial consistency error with  Galerkin coefficients (~1e-11 at
    J=8).  Leave at 0.0 to keep the operator .
    """
    dt_ = _resolve_dtype(dtype)
    n = Lam.n
    sc = Lam.scale * Gam.scale
    off = [int(p) for p in Lam.offsets]
    if [int(p) for p in Gam.offsets] != off:
        raise ValueError("Lambda and Gamma stencils must share the offset set")

    def _tr(F):
        sel = F.sigma > triad_rtol * F.sigma[0] if triad_rtol > 0 else slice(None)
        return (jnp.asarray(F.sigma[sel], dtype=dt_),
                jnp.asarray(F.a[sel], dtype=dt_),
                jnp.asarray(F.b[sel], dtype=dt_))

    L = _tr(Lam)
    G = _tr(Gam)

    def filt(X, coef, axis):
        """out[r] = sum_p coef[r, p] roll(X, -p, axis)   -> (R, n, n)."""
        S = jnp.stack([jnp.roll(X, -p, axis=axis) for p in off])      # (P, n, n)
        return jnp.tensordot(coef, S, axes=(1, 0), precision=HIGHEST)

    def pair(Af, Bf, X, Y):
        sx, ax, bx = X
        sy, ay, by = Y
        PA = filt(Af, ax, 0)
        PB = filt(Bf, bx, 0)

        def body(acc, inp):
            pa, pb, s_m = inp
            prod = filt(pa, ay, 1) * filt(pb, by, 1)                  # (Ry, n, n)
            return acc + s_m * jnp.tensordot(sy, prod, axes=(0, 0),
                                             precision=HIGHEST), None

        out, _ = lax.scan(body, jnp.zeros((n, n), dt_), (PA, PB, sx))
        return out

    def adv(U, V):
        Nu = pair(U, U, L, G) + pair(V, U, G, L)
        Nv = pair(U, V, L, G) + pair(V, V, G, L)
        return sc * Nu, sc * Nv

    return adv


def _resolve_scheme(name):
    """The three time integrators; anything else is refused."""
    if name in ("imex_euler", "imex_cn", "if_rk4"):
        return name
    raise ValueError(f"unknown advection_scheme {name!r}; expected "
                     f"'if_rk4', 'imex_cn' or 'imex_euler'")


def build_scaling_basis_stepper(ops1d, projector, dt, nu,
                                advection_scheme="imex_cn",
                                triad_rtol=0.0, filter_nyquist=None,
                                dtype=None, mu_d=0.0, advection=True):
    r"""One time step on the SCALING-basis pair (U_s, V_s) (jit-compiled).

    Snapshots in the wavelet basis:  U_wav = W_fwd @ U_s @ W_fwd.T.

    Semi-discrete system (the DEFINITION of the scheme), with
    L = nu (Delta_h (+) Delta_h) - mu_d I,  Delta_h = M^{-1} K, and h the
    Leray-projected skew-symmetric advection:

        du/dt = L u + h(u),   h(u) = -P M2^{-1} N(u, u),   u(0) in range(P).

    Every 2-D operator here lies in the commutative algebra generated by
    (circulant (x) circulant), so P commutes with L and with every function of
    L (verified to 3e-13).  Hence range(P) is invariant under the  flow
    AND under each scheme below, and there is no projection splitting error.

    Schemes
    -------
    'imex_euler'  u* = (I - dt L)^{-1} (u^n + dt h^n)                  O(dt)
    'imex_cn'     predictor as above, then Crank-Nicolson/Heun corrector
                  u^{n+1} = (I - dt/2 L)^{-1}[(I + dt/2 L) u^n
                                              + dt/2 (h(u^n) + h(u*))]  O(dt^2)
    'if_rk4'      Lawson / integrating-factor RK4:  viscous propagator
                  exp(dt L), classical RK4 on h.  O(dt^4); the RK4 stability
                  region contains the imaginary interval |y| < 2 sqrt 2, which
                  is what a skew (energy-neutral) advection needs.  Heun's
                  does not: |R(iy)|^2 = 1 + y^4/4 > 1.

    Measured Richardson orders (float64): imex_euler 1.00, imex_cn 2.00,
    if_rk4 4.0 (run ``TG.validate_time_integrator_order``).

    ``filter_nyquist=None`` inherits the projector's setting; an explicit
    value that disagrees with it raises (see ``_resolve_filter``).

    Linear (Ekman) drag.  K is the weak (negative semi-definite) Laplacian
    stencil, so the modal symbol
    z = dt (nu * Spec - mu_d) <= 0 and every propagator below is a
    contraction.  -mu_d I commutes with P, so range(P) stays invariant.
    mu_d = 0 recovers the unforced scheme .

    ``advection=False`` drops h: the step is then the  propagator
    P e^{dt L} of the Stokes + drag system for every scheme (used by the
    closed-form stochastic benchmark, BENCH.py).
    """

    scheme = _resolve_scheme(advection_scheme)
    work = _resolve_dtype(dtype)
    Spec, modal, modal2, project, mass_inv = make_scaling_pieces(
        ops1d, projector, filter_nyquist, dtype=work)
    adv = make_adv_scaling(ops1d.Lam, ops1d.Gam, triad_rtol, dtype=work) if advection else None

    if mu_d < 0:
        raise ValueError("mu_d must be >= 0 (a negative drag is an energy source)")
    z = dt * (nu * Spec - float(mu_d))

    if not advection:
        Ef_lin = jnp.exp(z)

        def step_linear(UV):
            U, V = (jnp.asarray(UV[0], dtype=work), jnp.asarray(UV[1], dtype=work))
            return project(modal(U, Ef_lin), modal(V, Ef_lin))
        return jax.jit(step_linear)

    be = 1.0 / (1.0 - z)
    cn_prop = (1.0 + 0.5 * z) / (1.0 - 0.5 * z)
    cn_src = (0.5 * dt) / (1.0 - 0.5 * z)

    def rhs(U, V):
        au, av = adv(U, V)
        return project(-mass_inv(au), -mass_inv(av))

    if scheme == "if_rk4":
        Eh = jnp.exp(0.5 * z)                  # exp(dt/2 L), |Eh| <= 1
        Ef = jnp.exp(z)                        # exp(dt   L)

        def step_ifrk4(UV):
            U, V = (jnp.asarray(UV[0], dtype=work), jnp.asarray(UV[1], dtype=work))
            au, av = rhs(U, V)
            bu, bv = rhs(modal(U + 0.5 * dt * au, Eh), modal(V + 0.5 * dt * av, Eh))
            cu, cv = rhs(modal(U, Eh) + 0.5 * dt * bu, modal(V, Eh) + 0.5 * dt * bv)
            du, dv = rhs(modal2(U, Ef, cu, dt * Eh), modal2(V, Ef, cv, dt * Eh))
            Un = modal2(U + (dt / 6.0) * au, Ef, bu + cu, (dt / 3.0) * Eh) + (dt / 6.0) * du
            Vn = modal2(V + (dt / 6.0) * av, Ef, bv + cv, (dt / 3.0) * Eh) + (dt / 6.0) * dv
            return project(Un, Vn)
        return jax.jit(step_ifrk4)

    def step(UV):
        # Cast on entry so that the returned carry is bit-for-bit the same type
        # as the one declared by ``dtype``: jax.lax.scan requires carry-in and
        # carry-out to match , and a float32 state meeting float64
        # constants silently promotes.
        U, V = (jnp.asarray(UV[0], dtype=work), jnp.asarray(UV[1], dtype=work))
        hu, hv = rhs(U, V)
        Up = modal(U + dt * hu, be)
        Vp = modal(V + dt * hv, be)
        if scheme == "imex_euler":
            return project(Up, Vp)
        ku, kv = rhs(Up, Vp)
        return project(modal2(U, cn_prop, hu + ku, cn_src),
                       modal2(V, cn_prop, hv + kv, cn_src))

    return jax.jit(step)


def scaling_kinetic_energy_fn(ops1d, dx, dtype=None):
    """E = 1/2 int |u_h|^2 from SCALING-basis fields, G applied banded.

    ``dtype`` must match the state the stepper carries, otherwise the energy
    (and, inside a lax.scan, the emitted snapshot) promotes.
    """
    Gl, _, _, GrT = make_banded_circulant(ops1d.G_vj, dtype=dtype)

    def energy(U, V):
        return 0.5 * (jnp.sum(U * GrT(Gl(U))) + jnp.sum(V * GrT(Gl(V)))) * dx * dx
    return energy


def scaling_dissipation_fn(ops1d, dx, dtype=None):
    r"""Z_gal(u_h) such that the VISCOUS energy rate of the scheme is -2 nu Z_gal.

    With E = 1/2 dx^2 x^T G2 x and the semi-discrete viscous term
    xdot = nu M2^{-1} Lap x,  Lap = K (x) M + M (x) K,

        dE/dt|_visc = nu dx^2 x^T G2 (Delta_h (+) Delta_h) x,   Delta_h = M^{-1} K,
        Z_gal      := -1/2 dx^2 x^T G2 (Delta_h (+) Delta_h) x.

    Bubnov (M = G): Z_gal = -1/2 dx^2 x^T Lap x = 1/2 ||grad u_h||^2, which is
    the enstrophy of u_h for periodic divergence-free fields.  Petrov: the
    same formula is the  dissipation of the scheme (it is not a
    symmetric form, but it is the number the energy budget contains).
    Using a separately discretised vorticity instead would add an O(h^p)
    mismatch to a budget whose other terms are .
    """
    dt_ = _resolve_dtype(dtype)
    n = ops1d.M_vj.shape[0]
    Mv = np.eye(n) if ops1d.test_space == "dual" else np.asarray(ops1d.M_vj, float)
    Lh = jnp.asarray(np.linalg.solve(Mv, np.asarray(ops1d.K_vj, float)), dtype=dt_)
    Gj = jnp.asarray(ops1d.G_vj, dtype=dt_)

    def Z(U, V):
        out = 0.0
        for X in (U, V):
            Y = mm(Lh, X) + mm(X, Lh.T)
            out = out + jnp.sum(X * sandwich(Gj, Y, Gj.T))
        return -0.5 * out * dx * dx
    return Z


def l2_inner_s(ops1d, A, B):
    """<a_h, b_h>_{L2} for SCALING-basis velocity pairs A = (U, V), B = (U', V').

    NumPy, float64.  The single scaling-basis L2 inner product of the code
    base (BENCH.l2_inner_w is its wavelet-basis counterpart)."""
    G = np.asarray(ops1d.G_vj, float)
    dx = 1.0 / ops1d.M_vj.shape[0]
    return float((np.sum(np.asarray(A[0], float) * (G @ np.asarray(B[0], float) @ G.T))
                  + np.sum(np.asarray(A[1], float) * (G @ np.asarray(B[1], float) @ G.T)))
                 * dx * dx)


def make_wavelet_maps(ops1d, dtype=None):
    """(w2s, s2w): flattened WAVELET state <-> SCALING-basis pair (U_s, V_s).

    w2s(x) = (W_inv U_w W_inv^T, W_inv V_w W_inv^T),   s2w = its inverse.
    Single implementation for the whole code base (HIGHEST precision).
    """
    dt_ = _resolve_dtype(dtype)
    n = ops1d.M_vj.shape[0]
    n2 = n * n
    Wf = jnp.asarray(ops1d.W_fwd, dtype=dt_)
    Wi = jnp.asarray(ops1d.W_inv, dtype=dt_)

    def w2s(x):
        return (sandwich(Wi, x[..., :n2].reshape(x.shape[:-1] + (n, n)), Wi.T),
                sandwich(Wi, x[..., n2:].reshape(x.shape[:-1] + (n, n)), Wi.T))

    def s2w(U, V):
        return jnp.concatenate([sandwich(Wf, U, Wf.T).reshape(U.shape[:-2] + (n2,)),
                                sandwich(Wf, V, Wf.T).reshape(V.shape[:-2] + (n2,))],
                               axis=-1)
    return w2s, s2w


def dyadic_shift_permutation_1d(W_fwd, W_inv, J, jmin, k):
    """Index array p with (W_fwd roll_s W_inv) e_i = e_{p[i]}, s = k 2^(J-jmin).

    A translation of the fine grid by s = k 2^(J-jmin) cells shifts the
    level-j block by k 2^(j-jmin) coefficients and never mixes levels, so in
    wavelet coordinates it is an  permutation (asserted here).  Shifts
    by other amounts are not: the periodised DWT is not shift invariant.
    """
    n = 2 ** J
    s = (k * 2 ** (J - jmin)) % n
    R = W_fwd @ np.roll(np.eye(n), s, axis=0) @ W_inv
    p = np.argmax(np.abs(R), axis=0)
    if not (np.allclose(np.abs(R[p, np.arange(n)]), 1.0, atol=1e-9)
            and len(set(p.tolist())) == n
            and np.allclose(R[p, np.arange(n)], 1.0, atol=1e-9)):
        raise RuntimeError("dyadic shift is not a permutation in wavelet coordinates")
    return p


# Stability interval of the explicit part on the imaginary axis, |R(iy)| <= 1
# for |y| <= y_max:  RK4 (inside if_rk4): 2 sqrt 2;  Heun and forward Euler:
# none (|R(iy)|^2 = 1 + y^4/4 and 1 + y^2) -- only viscosity stabilises them.
IMAG_AXIS_STABILITY = {"if_rk4": 2.0 * np.sqrt(2.0), "imex_cn": 0.0,
                       "imex_euler": 0.0}


def advective_spectral_radius(ops1d):
    """r_adv = max_theta |d_hat(theta) / m_hat(theta)|, the spectral radius of
    the 1-D discrete derivative M^{-1} D (scaling basis, circulant), 1/length.

    Frozen-coefficient bound for the skew advection:  the eigenvalues of the
    linearised advection lie on the imaginary axis with modulus
    <= r_adv * max(|u| + |v|), so the explicit stage is linearly stable iff
        dt * r_adv * max(|u| + |v|) <= IMAG_AXIS_STABILITY[scheme].
    Measured at J=8: rho*dx = 1.867 (Bubnov), 1.710 (Petrov), vs pi for the
     derivative (the symbol vanishes at theta = pi).
    """
    n = ops1d.M_vj.shape[0]
    Mv = np.eye(n) if ops1d.test_space == "dual" else np.asarray(ops1d.M_vj)
    d = np.fft.fft(np.asarray(ops1d.D_vj)[:, 0])
    m = np.fft.fft(Mv[:, 0])
    return float(np.max(np.abs(d / m)))


# 9.  Skew-symmetric advection

# The weak nonlinear term is evaluated in the fine SCALING basis, where the
# triad connection coefficients are banded circulant 3-tensors, and only the
# n x n RESULT is mapped into the wavelet basis.  Conjugating the tensors
# themselves would produce a dense n^3 array (128 MiB at J=8, O(n^4) work).
#
# Skew-symmetric (Temam) form
#     N(u; w) = (u.grad) w + 1/2 (div u) w
#             = 1/2 [ (u.grad) w + div(u w) ]
#
# so that,  and for ANY u (divergence free or not),
#
#     <w, N(u; w)> = -1/2 \int (div u)|w|^2 + 1/2 \int (div u)|w|^2 = 0.
#
#  Write the test function Psit_{pq} = phit_p(x) phit_q(y).
# For the transported component W (= u or v):
#
#   <Psit, u d_x W>       -> Lambda in x, Gamma in y, slots (u, W)
#   <Psit, v d_y W>       -> Gamma  in x, Lambda in y, slots (v, W)
#   1/2 <Psit,(d_x u) W>  -> Lambda^T in x, Gamma  in y, slots (u, W)
#   1/2 <Psit,(d_y v) W>  -> Gamma  in x, Lambda^T in y, slots (v, W)
#
# because moving the derivative from the second field slot to the first is
#  a transposition of the 1-D triad stencil.  Since the pairing is
# LINEAR in each stencil, all four terms collapse into two by using
#
#     Lambda_skew = Lambda + 1/2 Lambda^T
#
# which is what ``compute_operators`` stores in ``ops1d.Lam``.  The skew form
# is therefore free: same number of contractions, same memory, only a
# slightly higher SVD rank of the 1-D stencil.

def _adv_pair_np(Af, Bf, LX, LY):
    """sum_{m,k} sX_m sY_k (CXa_m Af CYa_k^T) o (CXb_m Bf CYb_k^T)."""
    CXa, CXb = LX.circulants(); CYa, CYb = LY.circulants()
    out = np.zeros_like(Af)
    for m in range(LX.rank):
        PA = CXa[m] @ Af
        PB = CXb[m] @ Bf
        for k in range(LY.rank):
            out += (LX.sigma[m] * LY.sigma[k]) * (PA @ CYa[k].T) * (PB @ CYb[k].T)
    return out


def compute_advection_(u, v, Lam, Gam):
    """Weak SKEW-SYMMETRIC Galerkin advection.

        N_u[p,q] = < Psit_pq , u d_x u + v d_y u + 1/2 (div u) u >
        N_v[p,q] = < Psit_pq , u d_x v + v d_y v + 1/2 (div u) v >

    ``Lam`` must be the skew stencil Lambda + 1/2 Lambda^T (which is what
    ``compute_operators`` returns).  Passing ``ops1d.Lam_conv`` instead
    recovers the old plain convective form, for A/B comparison.

    Derivation of the factorisation.  With
        Lam_wav[p,i,k] = sum_abc Wt[a,p] Ws[b,i] Ws[c,k] Lam_vj[a,b,c]
    the contraction
        N[p,q] = sum Lam_wav[p,i,k] Gam_wav[q,j,l] U_wav[i,j] U_wav[k,l]
    factorises , because sum_i Ws[b,i] sum_j Ws[e,j] U_wav[i,j] is
    (W_syn U_wav W_syn^T)[b,e] = U_s[b,e], the SCALING-basis field.  Hence
        N_wav = W_test N_s W_test^T,
        N_s[a,d] = sum Lam_vj[a,b,c] Gam_vj[d,e,f] U_s[b,e] U_s[c,f],
    where W_test = W_inv^T (Bubnov) or W_fwd (Petrov).  Note that BOTH
    indices of N carry the test transform: N is a covector in each.
    """
    n = Lam.n
    Ws = Lam.W_syn
    Wt = Lam.W_test
    Us = Ws @ np.asarray(u).reshape(n, n) @ Ws.T
    Vs = Ws @ np.asarray(v).reshape(n, n) @ Ws.T
    sc = Lam.scale * Gam.scale
    N_u = _adv_pair_np(Us, Us, Lam, Gam) + _adv_pair_np(Vs, Us, Gam, Lam)
    N_v = _adv_pair_np(Us, Vs, Lam, Gam) + _adv_pair_np(Vs, Vs, Gam, Lam)
    return (sc * (Wt @ N_u @ Wt.T)).reshape(-1), (sc * (Wt @ N_v @ Wt.T)).reshape(-1)



# 11.  Diagnostics and validation


def _energy_pairing(ops, x):
    """y = G2 x per component: the covector such that dE/dt = y . xdot."""
    n2 = ops.n_var
    x = np.asarray(x)
    return [ops.apply_gram(x[:n2]), ops.apply_gram(x[n2:])]


def _energy_production(ops, x, N_u, N_v):
    """|cos| of the L2 angle between u_h and the Galerkin advection field.

    With n_h the V_J field whose coefficients are M2^{-1} N (for Petrov, N
    itself; for Bubnov, G2^{-1} N), the advective energy production is
    dE/dt|_adv = -<u_h, n_h>_{L2} = -(G2 x).(M2^{-1} N), and

        cos = |<u_h, n_h>| / (||u_h||_{L2} ||n_h||_{L2})  in [0, 1]

    is dimensionless and basis independent.
    """
    n2 = ops.n_var
    x = np.asarray(x)
    nu_ = ops.solve_mass(N_u); nv_ = ops.solve_mass(N_v)
    yu, yv = _energy_pairing(ops, x)
    num = float(yu @ nu_ + yv @ nv_)
    ex = float(yu @ x[:n2] + yv @ x[n2:])
    en = float(nu_ @ ops.apply_gram(nu_) + nv_ @ ops.apply_gram(nv_))
    return abs(num) / (np.sqrt(max(ex, 0.0) * max(en, 0.0)) + 1e-300)


def _smooth_divfree_field(J, k_max=3, seed=0):
    """Band-limited, resolved, analytically divergence-free test field.

    Contains every mode (m, q), 1 <= m, q <= k_max, with random amplitudes
    and phases, so wavevector triads k = p + q CLOSE inside the field.  This
    is required for any energy-transfer test: a field with a parity selection
    rule (e.g. only odd k_y), for which (u.grad)u has only even k_y, gives
    sum_k u_hat(k)* N_hat(k) = 0 mode by mode, whatever the discretisation.
    """
    rng = np.random.default_rng(seed)
    n = 2 ** J
    g = np.linspace(0.0, 1.0, n, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing='ij')
    u = np.zeros((n, n)); v = np.zeros((n, n))
    for m in range(1, k_max + 1):
        for q in range(1, k_max + 1):
            a = rng.standard_normal() / (m * m + q * q)
            px, py = rng.uniform(0, 2 * np.pi, 2)
            kx, ky = 2 * np.pi * m, 2 * np.pi * q
            u += a * ky * np.sin(kx * X + px) * np.cos(ky * Y + py)
            v -= a * kx * np.cos(kx * X + px) * np.sin(ky * Y + py)
    s = np.sqrt(np.mean(u ** 2 + v ** 2)) or 1.0
    return u / s, v / s


def _validation_setup(ops1d, projector):
    """Validators always run on the PRODUCTION operators and projector."""
    if ops1d is None or projector is None:
        raise ValueError("pass the production ops1d and projector")
    return ops1d, build_2d_operators(ops1d), projector


def _wavelet_state(u, v, ops1d):
    """ Galerkin coefficients of (u, v), flattened, in the WAVELET basis."""
    Wf = ops1d.W_fwd
    A = galerkin_multiplier(u.shape[0], ops1d.test_space)
    return np.concatenate([(Wf @ samples_to_coeffs(u, A) @ Wf.T).reshape(-1),
                           (Wf @ samples_to_coeffs(v, A) @ Wf.T).reshape(-1)])

def validate_energy_conservation_rough(ops1d, projector, slopes=(3.0, 5.0 / 3.0),
                                       n_trials=2, seed=0, verbose=True):
    """Advective energy production on ROUGH divergence-free fields whose
    spectrum E(k) ~ k^-slope extends to Nyquist (the regime the dataset
    reaches).  Bubnov: roundoff ( skew symmetry).  Petrov: the defect
    -<Pt* u_h - u_h, N(u_h)>, measured 2e-6..2e-4 with BOTH signs at J=6.
    The k_max=3 tests cannot see it (they have no grid-scale content)."""
    ops1d, ops, proj = _validation_setup(ops1d, projector)
    J, test_space = ops1d.J, ops1d.test_space
    n = 2 ** J; n2 = n * n
    Wf = ops1d.W_fwd
    k = np.fft.fftfreq(n, d=1.0 / n)
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2); kk[0, 0] = 1.0
    worst, signs = 0.0, []
    for slope in slopes:
        filt = kk ** (-(slope + 1.0) / 2.0); filt[0, 0] = 0.0   # E(k) ~ k^-slope
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)
            U = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt))
            V = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt))
            x = np.concatenate([(Wf @ U @ Wf.T).reshape(-1), (Wf @ V @ Wf.T).reshape(-1)])
            x, _ = proj.project(x)
            Nu, Nv = compute_advection_(x[:n2], x[n2:], ops1d.Lam, ops1d.Gam)
            worst = max(worst, _energy_production(ops, x, Nu, Nv))
            yu, yv = _energy_pairing(ops, x)
            signs.append(np.sign(-(yu @ ops.solve_mass(Nu) + yv @ ops.solve_mass(Nv))))
    if verbose:
        print(f"  [validate_energy_conservation_rough] J={J}, test_space={test_space}: "
              f"max L2 cosine = {worst:.3e}; production signs = {sorted(set(signs))}")
    return float(worst)

def validate_advection_operator(ops1d, projector, n_trials=3, seed=0, k_max=3,
                                verbose=True):
    r"""Spatial consistency and advective energy production of N.

    ``err_consistency``  max over trials of
        || M2^{-1} N(u_h) - c(N_) ||_2 / || c(N_) ||_2,
    c(.) =  Galerkin coefficients, N_ = the pointwise skew-symmetric
    nonlinearity of the analytic band-limited field (FFT products are 
    since 2 k_max < n/2).

    ``err_energy``  max over trials of the L2 cosine between u_h and the
    Galerkin advection field (see ``_energy_production``):
      * ROUNDOFF for test_space='primal' ( discrete skew symmetry);
      * the Petrov defect -<Pt* u_h - u_h, N>/(...) for 'dual': measured
        3.0e-8, 1.3e-10, 4.9e-13 at J = 5, 6, 7 (k_max = 3), i.e. ~h^8, and
        at roundoff from J = 8 on.

    The fields have closing triads (see ``_smooth_divfree_field``): a field
    with a parity selection rule passes the energy test for ANY scheme.
    """
    ops1d, ops, proj = _validation_setup(ops1d, projector)
    J, test_space = ops1d.J, ops1d.test_space
    n = 2 ** J; n2 = n * n
    Wi = ops1d.W_inv
    A = galerkin_multiplier(n, test_space)
    k = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = k[:, None], k[None, :]
    d = lambda f, Kk: np.real(np.fft.ifft2(1j * Kk * np.fft.fft2(f)))

    errs_c, errs_e, res = [], [], []
    for trial in range(n_trials):
        u, v = _smooth_divfree_field(J, k_max=k_max, seed=seed + trial)
        x, _ = proj.project(_wavelet_state(u, v, ops1d))
        res.append(proj.normalised_residual(x))
        N_u, N_v = compute_advection_(x[:n2], x[n2:], ops1d.Lam, ops1d.Gam)
        div = d(u, KX) + d(v, KY)
        ref_u = samples_to_coeffs(u * d(u, KX) + v * d(u, KY) + 0.5 * div * u, A)
        ref_v = samples_to_coeffs(u * d(v, KX) + v * d(v, KY) + 0.5 * div * v, A)
        got_u = Wi @ ops.solve_mass(N_u).reshape(n, n) @ Wi.T
        got_v = Wi @ ops.solve_mass(N_v).reshape(n, n) @ Wi.T
        errs_c.append(np.linalg.norm(np.stack([got_u - ref_u, got_v - ref_v]))
                      / (np.linalg.norm(np.stack([ref_u, ref_v])) + 1e-300))
        errs_e.append(_energy_production(ops, x, N_u, N_v))
    err_consistency, err_energy = float(max(errs_c)), float(max(errs_e))
    if verbose:
        print(f"  [validate_advection_operator] J={J}, test_space={test_space}, "
              f"{n_trials} fields with closing triads (k_max={k_max}): "
              f"||C x|| normalised = {max(res):.2e}; consistency = "
              f"{err_consistency:.3e}; energy production (L2 cosine) = "
              f"{err_energy:.3e}")
    return err_consistency, err_energy


def validate_skew_symmetry(ops1d, projector, n_trials=4, seed=0, k_max=3,
                           verbose=True):
    """Advective energy production, skew-symmetric vs plain convective form.

    Returns {'skew': max cos, 'convective': max cos, 'discriminating': bool}.
    For the convective form the production is proportional to the WEAK
    divergence defect of u_h, for the skew form it is  zero (Bubnov)
    or the O(h^8) Petrov defect.  Both are O(h^8) on resolved fields, so at
    large J they reach roundoff together and the comparison says nothing;
    'discriminating' is False in that case.
    """
    ops1d, ops, proj = _validation_setup(ops1d, projector)
    J, test_space = ops1d.J, ops1d.test_space
    n = 2 ** J; n2 = n * n
    out = {"skew": [], "convective": []}
    for trial in range(n_trials):
        u, v = _smooth_divfree_field(J, k_max=k_max, seed=seed + trial)
        x, _ = proj.project(_wavelet_state(u, v, ops1d))
        for name, L in (("skew", ops1d.Lam), ("convective", ops1d.Lam_conv)):
            Nu, Nv = compute_advection_(x[:n2], x[n2:], L, ops1d.Gam)
            out[name].append(_energy_production(ops, x, Nu, Nv))
    res = {k: float(np.max(v)) for k, v in out.items()}
    floor = 1e3 * np.finfo(float).eps
    res["discriminating"] = bool(res["convective"] > floor)
    if verbose:
        print(f"  [validate_skew_symmetry] J={J}, test_space={test_space}: "
              f"max L2 cosine  skew={res['skew']:.3e}  "
              f"convective={res['convective']:.3e}"
              + ("" if res["discriminating"] else
                 "  (both at roundoff: not discriminating at this J)"))
    return res