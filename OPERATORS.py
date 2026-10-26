"""OPERATORS.py: 1-D and 2-D wavelet operators for the NSWSGM solver.

Spaces and bases
----------------
V_J = span{phi_{J,k}} (primal CDF 9/7 scaling functions, periodic, n = 2^J).
Two coordinate systems on the SAME space V_J:

    scaling coefficients  U_s   (single-scale; every operator is a banded
                                  circulant here -> this is where we time-step)
    wavelet coefficients  U_w = W_fwd U_s W_fwd^T   (multi-scale; this is the
                                  coordinate system of the generative model,
                                  of the forcing covariance and of the OU
                                  diffusion schedule)

W_fwd (analysis, built from the DUAL filters) and W_inv = W_fwd^{-1}
(synthesis, PRIMAL filters) are  inverses, so the change of basis is a
similarity and the discrete dynamics are identical in both systems.  This is
why the solver "only uses scaling functions": the detail wavelets enter
through W_fwd / W_inv and nowhere else, and nothing is lost.

Test space
----------
test_space='primal'  Bubnov-Galerkin: test = trial = primal.  M = G (primal
                     Gram).  Only primal functions appear in the operators; the
                     dual functions appear only inside W_fwd.  
                     energy-neutral skew advection.  PRODUCTION.
test_space='dual'    Petrov-Galerkin: dual test, primal trial.  M = I by
                     biorthogonality; the L2 metric is still G != M.
"""
import os
if os.environ.get("NSWSGM_X64", "0") == "1":
    import jax
    jax.config.update("jax_enable_x64", True)
else:
    import jax

import numpy as np
import pywt
from scipy.linalg import eigh
import warnings

import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass

WAVELET_NAME = 'bior4.4'
wavelet = pywt.Wavelet(WAVELET_NAME)

# -----------------------------------------------------------------------------
# -arithmetic contract for JAX linear algebra (see module docstring).
# -----------------------------------------------------------------------------
HIGHEST = lax.Precision.HIGHEST


def mm(a, b):
    """a @ b in full working precision (never TF32/bf16 passes on GPU)."""
    return jnp.matmul(a, b, precision=HIGHEST)


def sandwich(L, X, Rt):
    """L @ X @ Rt in full working precision."""
    return mm(mm(L, X), Rt)



# 0.  Filters, with EXPLICIT tap indices

# The primal (synthesis) and dual (analysis) scaling functions of a
# biorthogonal pair satisfy
#
#     phi(x)  = sum_i  h[i]  phi(2x - i),        <phit(.), phi(. - l)> = d_{l0}
#     phit(x) = sum_i  ht[i] phit(2x - i),
#
# and biorthogonality holds only for ONE relative alignment of the two filter
# index sets.  PyWavelets stores dec_lo in the reversed (correlation)
# convention and pads both filters to a common length with zeros, so the
# naive "strip zeros from each and centre each on its own support" recipe
# misaligns the pair by one sample.  We therefore carry the tap indices
# explicitly and choose the offset that makes the mixed two-term stencil come
# out as the Kronecker delta -- which is then asserted at build time.

def _trim(a):
    a = np.asarray(a, dtype=float)
    nz = np.nonzero(np.abs(a) > 1e-14)[0]
    return a[nz[0]:nz[-1] + 1]


_H_PRIMAL = _trim(wavelet.rec_lo)
_H_PRIMAL = _H_PRIMAL * (2.0 / np.sum(_H_PRIMAL))
_I_PRIMAL = np.arange(len(_H_PRIMAL))

_H_DUAL = _trim(wavelet.dec_lo[::-1])
_H_DUAL = _H_DUAL * (2.0 / np.sum(_H_DUAL))
# common symmetry centre: both supports are centred on (len-1)/2 of the primal
_I_DUAL = np.arange(len(_H_DUAL)) - (len(_H_DUAL) - len(_H_PRIMAL)) // 2

_SUPPORT_PAD = int(max(_I_PRIMAL[-1] - _I_DUAL[0], _I_DUAL[-1] - _I_PRIMAL[0])) + 1


def _filters(test_space: str):
    """Return (h_test, i_test, h_trial, i_trial) for the requested test space."""
    if test_space == "dual":
        return _H_DUAL, _I_DUAL, _H_PRIMAL, _I_PRIMAL
    if test_space == "primal":
        return _H_PRIMAL, _I_PRIMAL, _H_PRIMAL, _I_PRIMAL
    raise ValueError("test_space must be 'dual' (Petrov-Galerkin) or "
                     "'primal' (Bubnov-Galerkin)")



# 1.  Two-term (mixed) connection coefficients


def build_2term_transition_matrix(test_space: str, Pmax: int = None):
    """Transition operator of the mixed two-scale relation.

        A[l, l'] = 1/2 sum_i  ht[i] h[i + l' - 2l]

    Its eigenvectors for eigenvalues 1, 1/2, 1/4 are respectively

        m[l] = <phit(.),  phi(. - l)>          ( = delta_{l0} when test=dual )
        d[l] = <phit(.),  phi'(. - l)>
        k[l] = <phit(.),  phi''(. - l)>
    """
    hT, iT, hR, iR = _filters(test_space)
    if Pmax is None:
        Pmax = _SUPPORT_PAD
    idxs = np.arange(-Pmax, Pmax + 1)
    L = len(idxs)
    lut = {int(v): hR[k] for k, v in enumerate(iR)}
    A = np.zeros((L, L))
    for a, l in enumerate(idxs):
        for b, lp in enumerate(idxs):
            val = 0.0
            for k, iv in enumerate(iT):
                j = int(iv) + lp - 2 * l
                if j in lut:
                    val += hT[k] * lut[j]
            A[a, b] = 0.5 * val
    return A, idxs


def compute__stencils(test_space: str, return_defects: bool = False):
    """ two-term connection coefficients.

    Normalisations (all derived from the polynomial reproduction
    x^r = sum_l <phit(.-l), x^r> phi(x-l) of the PRIMAL basis, so they are
    identical in the Bubnov and Petrov cases):

        sum_l      m[l] = 1        sum_l l   d[l] = 1        sum_l l^2 k[l] = 2

    Sign convention for K:  K_st[l] = <phit, phi''(.-l)>, i.e. the WEAK
    LAPLACIAN, negative semi-definite.  In the Bubnov case this coincides
    with -<phi', phi'(.-l)> after an integration by parts; in the Petrov case
    the derivative is deliberately kept on the (smooth) trial function,
    because the dual scaling function of CDF(9,7) is only barely in H^1 and
    <phit', phi'> would be a badly conditioned object.
    """
    A, idxs = build_2term_transition_matrix(test_space)
    w, v = np.linalg.eig(A)
    w, v = np.real(w), np.real(v)

    def pick(target):
        return v[:, np.argmin(np.abs(w - target))]

    def _asym(a, sign):              # relative (anti)symmetry defect of a stencil
        return float(np.max(np.abs(a - sign * a[::-1])) / max(np.max(np.abs(a)), 1e-300))

    M_raw = pick(1.0)
    M_st = M_raw / np.sum(M_raw)
    D_raw = pick(0.5)
    D_st = D_raw * (1.0 / np.sum(idxs * D_raw))
    K_raw = pick(0.25)
    K_st = K_raw * (2.0 / np.sum((idxs ** 2) * K_raw))
    defects = {"m stencil symmetric": _asym(M_st, +1),
               "d stencil antisymmetric": _asym(D_st, -1),
               "k stencil symmetric": _asym(K_st, +1)}
    M_st = 0.5 * (M_st + M_st[::-1])
    D_st = 0.5 * (D_st - D_st[::-1])
    K_st = 0.5 * (K_st + K_st[::-1])

    if test_space == "dual":
        # Biorthogonality is a *theorem*, so use it as a build-time assertion
        # on the filter alignment rather than trusting PyWavelets' padding.
        delta = (idxs == 0).astype(float)
        err = float(np.max(np.abs(M_st - delta)))
        if err > 1e-8:
            raise RuntimeError(
                f"Mixed two-term stencil is not the Kronecker delta (max error "
                f"{err:.2e}).  The primal/dual filter alignment is wrong -- "
                f"biorthogonality <phit(.-a), phi(.-b)> = delta_ab is violated."
            )
        M_st = delta

    if return_defects:
        return M_st, D_st, K_st, idxs, defects
    return M_st, D_st, K_st, idxs



# 2.  Three-term (triad) connection coefficients


def build_triad_transition_tensor(test_space: str, Pmax: int = None):
    """Transition operator for  T[a,b,c] = <phit(.-a) phi(.-b) phi^(r)(.-c)>.

    Eigenvalue 2 -> Gamma[p,q] = <phit, phi(.-p) phi(.-q)>
    Eigenvalue 1 -> Lambda[p,q] = <phit, phi(.-p) phi'(.-q)>
    """
    hT, iT, hR, iR = _filters(test_space)
    if Pmax is None:
        Pmax = _SUPPORT_PAD
    pidx = np.arange(-Pmax, Pmax + 1)
    Lp = len(pidx)
    lut = {int(v): hR[k] for k, v in enumerate(iR)}
    T = np.zeros((Lp * Lp, Lp * Lp))
    for a, p in enumerate(pidx):
        for b, q in enumerate(pidx):
            row = a * Lp + b
            for c, m in enumerate(pidx):
                for d, n in enumerate(pidx):
                    val = 0.0
                    for kk, iv in enumerate(iT):
                        j = int(iv) + m - 2 * p
                        k = int(iv) + n - 2 * q
                        if j in lut and k in lut:
                            val += hT[kk] * lut[j] * lut[k]
                    T[row, c * Lp + d] = val
    return T, pidx


def compute__triad_stencils(M_st, idxs, test_space: str):
    """Gamma and Lambda stencils, normalised by primal polynomial reproduction.

        sum_{p,q}   Gamma[p,q]     = 1
        sum_q  q *  Lambda[p,q]    = m[p]       (= delta_{p0} for test=dual)
    """
    T, pidx = build_triad_transition_tensor(test_space)
    Lp = len(pidx)
    w, v = np.linalg.eig(T)
    w, v = np.real(w), np.real(v)

    Gam_raw = v[:, np.argmin(np.abs(w - 2.0))].reshape(Lp, Lp)
    Gam_st = Gam_raw / np.sum(Gam_raw)
    Gam_st = 0.5 * (Gam_st + Gam_st.T)      #  symmetry in (p,q)

    mask = np.abs(w - 1.0) < 1e-6
    degvecs = v[:, mask]
    if degvecs.shape[1] < 1:
        raise RuntimeError("Lambda eigenspace not found near eigenvalue 1.0 "
                           "-- check filter normalisation.")
    rows, rhs = [], []
    for p in idxs:
        hit = np.where(pidx == p)[0]
        if hit.size == 0:
            continue
        pa = int(hit[0])
        rows.append([np.sum(pidx * degvecs[:, k].reshape(Lp, Lp)[pa, :])
                     for k in range(degvecs.shape[1])])
        rhs.append(M_st[int(np.where(idxs == p)[0][0])])
    coeffs, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    Lam_st = np.zeros((Lp, Lp), dtype=float)
    for k in range(degvecs.shape[1]):
        Lam_st += float(coeffs[k]) * degvecs[:, k].reshape(Lp, Lp)

    return np.asarray(Gam_st, dtype=float), Lam_st, np.asarray(pidx, dtype=int)


def _test_scaling_function(level=13):
    """(phi_test, phi_trial, x) sampled on a common grid."""
    phi_d, psi_d, phi_r, psi_r, x = wavelet.wavefun(level=level)
    return phi_d, phi_r, x


def _validate_stencils(M_st, D_st, K_st, idxs, Gam_st, Lam_st, pidx, *,
                       test_space, verbose=True, tol=1e-4):
    """Compare every stencil against direct numerical quadrature."""
    phi_d, phi_r, x = _test_scaling_function()
    phi_test = phi_d if test_space == "dual" else phi_r
    dx = x[1] - x[0]
    n_ = len(phi_r)
    d1 = np.gradient(phi_r, dx)
    d2 = np.gradient(d1, dx)
    lmax = int(np.floor((x[-1] - x[0]) / 4))

    def shift(f, k):
        pts = int(round(k / dx))
        fs = np.zeros_like(f)
        if pts >= 0:
            fs[pts:] = f[:n_ - pts]
        elif pts < 0:
            fs[:n_ + pts] = f[-pts:]
        return fs

    P2 = idxs[-1]
    ls = [l for l in idxs if abs(l) <= lmax]
    errM = max(abs(M_st[l + P2] - np.sum(phi_test * shift(phi_r, l)) * dx) for l in ls)
    errD = max(abs(D_st[l + P2] - np.sum(phi_test * shift(d1, l)) * dx) for l in ls)
    errK = max(abs(K_st[l + P2] - np.sum(phi_test * shift(d2, l)) * dx) for l in ls)

    P3 = pidx[-1]
    test_pairs = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 1), (2, -1), (3, 2), (-2, -3)]
    test_pairs = [(p, q) for p, q in test_pairs if abs(p) <= lmax and abs(q) <= lmax]
    errG = max(abs(Gam_st[p + P3, q + P3]
                   - np.sum(phi_test * shift(phi_r, p) * shift(phi_r, q)) * dx)
               for p, q in test_pairs)
    errL = max(abs(Lam_st[p + P3, q + P3]
                   - np.sum(phi_test * shift(phi_r, p) * shift(d1, q)) * dx)
               for p, q in test_pairs)

    if verbose:
        print(f"  [Validation] test space = {test_space}; "
              f"max|error| vs direct numerical integration:")
        print(f"    M: {errM:.2e}   D: {errD:.2e}   K: {errK:.2e}   "
              f"Gamma: {errG:.2e}   Lambda: {errL:.2e}")
    worst = max(errM, errD, errK, errG, errL)
    if worst > tol:
        raise RuntimeError(
            f"Connection-coefficient validation exceeded tolerance: "
            f"{worst:.2e} > {tol:.0e}.  Do not trust downstream operators.")
    return errM, errD, errK, errG, errL



# 3.  Discrete wavelet transform matrices


def pywt_unpack_1d(s, J, jmin):
    s = np.asarray(s, dtype=float)
    coeffs = pywt.wavedec(s, wavelet, level=J - jmin, mode='periodization')
    return coeffs[0], coeffs[1:]


def wavelet_reconstruct_from_coeffs(coeff_vec, J, jmin):
    n_coarse = 2 ** jmin
    c_coarse = coeff_vec[:n_coarse]
    lengths, L = [], 2 ** jmin
    for j in range(jmin, J):
        lengths.append(L)
        L *= 2
    details, pos = [], n_coarse
    for l in lengths:
        details.append(coeff_vec[pos:pos + l])
        pos += l
    s = pywt.waverec([c_coarse] + details, wavelet, mode='periodization')
    return s[:2 ** J]


def build_dwt_matrices(J, jmin):
    """W_fwd = analysis (dual synthesis^T), W_inv = synthesis.

    Conventions used throughout, with Phi the row vector of primal scaling
    functions and Psi the row vector of primal wavelets:

        Psi   = Phi  W_inv                (W_inv  = synthesis matrix S)
        Psit  = Phit W_fwd^T              (W_fwd^T = S^{-T}, dual synthesis)

    so that  W_fwd W_inv = I  and a Petrov-Galerkin operator maps as
    A_wav = W_fwd A_VJ W_inv while a Bubnov one maps as W_inv^T A_VJ W_inv.
    """
    n = 2 ** J
    n_coarse = 2 ** jmin
    if n_coarse < wavelet.dec_len:
        warnings.warn(
            f"jmin={jmin} gives a coarse block of only {n_coarse} samples, smaller "
            f"than PyWavelets' stored filter length ({wavelet.dec_len}). Expect "
            f"boundary wrap-around warnings from PyWavelets (expected for "
            f"periodization).")
    W_fwd = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        c, d = pywt_unpack_1d(e, J, jmin)
        W_fwd[:, i] = np.concatenate([c] + d)

    W_inv = np.zeros((n, n))
    for i in range(n):
        cu = np.zeros(n); cu[i] = 1.0
        W_inv[:, i] = wavelet_reconstruct_from_coeffs(cu, J, jmin)

    return W_fwd, W_inv          # W_fwd W_inv = I is asserted by validate_operators



# 3bis.   sample -> Galerkin-coefficient map (removes the variational crime)

# The Petrov-Galerkin coefficients of u are c_k = 2^J <phit_{J,k}, u>, NOT the
# point samples u(x_k).  For e^{i xi x/dx} the two differ by the multiplier
# A(xi) = FT(phit)(xi) e^{i c xi} = 1 - O(xi^2) (c = common symmetry centre;
# the resulting global shift by c*dx commutes with every circulant operator
# and is a pure relabeling).  Using samples as coefficients would be an O(h^2)
# variational crime; with the  multiplier the consistency error of
# ODE.validate_advection_operator is spectral (~h^7.7).

def _scaling_symbol(h, idx, xi, n_prod=40):
    c = _I_PRIMAL[-1] / 2.0
    out = np.ones(len(xi), dtype=complex)
    for j in range(1, n_prod + 1):
        w = xi / 2.0 ** j
        out *= 0.5 * np.sum(h[None, :] * np.exp(-1j * w[:, None] * (idx[None, :] - c)), axis=1)
    return out.real

def dual_scaling_multiplier(n, n_prod=40):
    """Petrov: coefficients of the OBLIQUE projection, c_k = 2^J <phit_k, u>."""
    return _scaling_symbol(_H_DUAL, _I_DUAL, 2.0 * np.pi * np.fft.fftfreq(n), n_prod)

def primal_projection_multiplier(n, n_prod=40):
    """Bubnov: coefficients of the L2 projection, c = G^{-1} 2^J <phi, u>."""
    xi = 2.0 * np.pi * np.fft.fftfreq(n)
    Gst, _, _, idx = compute__stencils("primal")
    Ghat = np.real(np.sum(Gst[None, :] * np.exp(-1j * np.outer(xi, idx)), axis=1))
    return _scaling_symbol(_H_PRIMAL, _I_PRIMAL, xi, n_prod) / Ghat

def galerkin_multiplier(n, test_space):
    if test_space == "dual":   return dual_scaling_multiplier(n)
    if test_space == "primal": return primal_projection_multiplier(n)
    raise ValueError("test_space must be 'dual' or 'primal'")

def samples_to_coeffs(f, A=None, test_space=None):
    """ Galerkin coefficients of the samples f; pass the multiplier A
    (galerkin_multiplier) or the test space it is built from."""
    f = np.asarray(f, dtype=float)
    if A is None:
        if test_space is None:
            raise ValueError("samples_to_coeffs needs A or test_space")
        A = galerkin_multiplier(f.shape[0], test_space)
    return np.real(np.fft.ifft2(np.fft.fft2(f) * (A[:, None] * A[None, :])))



# 4.  Triad stencils kept as O(1) objects (never as n^3 tensors)


@dataclass
class TriadFactor:
    """SVD factorisation S = sum_r sigma_r a_r b_r^T of a 1-D triad stencil.

    ``W_syn``  maps wavelet coefficients -> scaling coefficients (trial side).
    ``W_test`` maps a scaling-basis test functional -> the wavelet-basis one:
        Bubnov: W_test = W_inv^T ;  Petrov: W_test = W_fwd.
    """
    offsets: np.ndarray
    sigma:   np.ndarray
    a:       np.ndarray
    b:       np.ndarray
    scale:   float
    n:       int
    W_syn:   np.ndarray
    W_test:  np.ndarray

    @property
    def rank(self):
        return int(len(self.sigma))

    @property
    def nbytes(self):
        return int(self.sigma.nbytes + self.a.nbytes + self.b.nbytes)

    def circulants(self):
        """Dense (R,n,n) banded circulants C[r, i, (i+p) % n] = coeff_r[p]."""
        if getattr(self, "_circ", None) is None:
            n, R, i = self.n, self.rank, np.arange(self.n)
            CA = np.zeros((R, n, n)); CB = np.zeros((R, n, n))
            for pa, p in enumerate(self.offsets):
                CA[:, i, (i + int(p)) % n] += self.a[:, pa][:, None]
                CB[:, i, (i + int(p)) % n] += self.b[:, pa][:, None]
            self._circ = (CA, CB)
        return self._circ


def factor_triad_stencil(S, offsets, scale, n, W_syn, W_test, rtol=1e-10):
    """Rank-truncated SVD of a triad stencil (rtol=1e-10 is  to roundoff)."""
    u, s, vt = np.linalg.svd(np.asarray(S, dtype=float))
    R = max(1, int(np.sum(s > rtol * s[0])))
    return TriadFactor(np.asarray(offsets, dtype=int), s[:R].copy(),
                       u[:, :R].T.copy(), vt[:R, :].copy(),
                       float(scale), int(n),
                       np.asarray(W_syn, dtype=float),
                       np.asarray(W_test, dtype=float))



# 5.  The 1-D operator bundle


@dataclass
class WaveletOperators1D:
    """Every 1-D object the solver needs, in both bases.

    Wavelet-basis matrices (what the state vector sees):
        M   test-trial mass          ( = I  for test_space='dual' )
        D   first derivative
        K   weak Laplacian           (negative semi-definite spectrum)
        G   PRIMAL Gram <phi_a,phi_b>  -- the L2 metric, NOT the mass matrix

    Scaling-basis matrices (all circulant, all with  structure):
        M_vj (=I for Petrov), D_vj (skew), K_vj (symmetric, <=0), G_vj (SPD)

    Triad factors:
        Lam       skew-symmetric advection stencil  Lambda + 1/2 Lambda^T
        Lam_conv  plain convective stencil Lambda   (diagnostics only)
        Gam       <phit, phi phi>
    """
    M: np.ndarray
    D: np.ndarray
    K: np.ndarray
    G: np.ndarray
    Lam: TriadFactor
    Gam: TriadFactor
    Lam_conv: TriadFactor
    M_vj: np.ndarray
    D_vj: np.ndarray
    K_vj: np.ndarray
    G_vj: np.ndarray
    W_fwd: np.ndarray
    W_inv: np.ndarray
    W_test: np.ndarray
    W_test_inv: np.ndarray
    J: int
    jmin: int
    test_space: str
    raw_defects: dict = None      # structure defects BEFORE the roundoff cleanup


def compute_operators(J, jmin, *, test_space, verbose=True, triad_rtol=1e-10):
    """Build every 1-D operator for the chosen test space.

    test_space='dual'   -> Petrov-Galerkin (primal trial / dual test), M = I
    test_space='primal' -> Bubnov-Galerkin (test = trial),             M = G
    """
    if verbose:
        print(f"  -> Solving transition-matrix eigensystems "
              f"(test space: {test_space})...")

    M_st, D_st, K_st, idxs, st_def = compute__stencils(test_space, return_defects=True)
    Gam_st, Lam_st, pidx = compute__triad_stencils(M_st, idxs, test_space)
    _validate_stencils(M_st, D_st, K_st, idxs, Gam_st, Lam_st, pidx,
                       test_space=test_space, verbose=verbose)

    # The primal Gram stencil is always needed (it is the L2 metric).
    Gp_st, _, _, idxs_p, g_def = compute__stencils("primal", return_defects=True)

    n = 2 ** J

    def circulant_from(stencil, st_idxs, scale):
        row = np.zeros(n)
        for k in st_idxs:
            row[int(k) % n] += stencil[int(k) + st_idxs[-1]] * scale
        Cm = np.empty((n, n))
        for i in range(n):
            Cm[i] = np.roll(row, i)
        return Cm

    M_vj = circulant_from(M_st, idxs, 1.0)
    D_vj = circulant_from(D_st, idxs, 2.0 ** J)
    K_vj = circulant_from(K_st, idxs, 2.0 ** (2 * J))
    G_vj = circulant_from(Gp_st, idxs_p, 1.0)

    #  structure (circulants, so this is a roundoff cleanup only).  The
    # structural defects are recorded on the RAW stencils, before
    # symmetrisation; after it they would be 0 by construction.
    raw = {k: v for k, v in st_def.items() if not (test_space == "dual" and k.startswith("m "))}
    raw["g (primal Gram) stencil symmetric"] = g_def["m stencil symmetric"]
    M_vj = 0.5 * (M_vj + M_vj.T)
    D_vj = 0.5 * (D_vj - D_vj.T)
    K_vj = 0.5 * (K_vj + K_vj.T)
    G_vj = 0.5 * (G_vj + G_vj.T)

    W_fwd, W_inv = build_dwt_matrices(J, jmin)
    if test_space == "dual":
        W_test, W_test_inv = W_fwd, W_inv          # similarity
    else:
        W_test, W_test_inv = W_inv.T, W_fwd.T      # congruence

    # Wavelet-basis images.  They inherit the (anti)symmetry of the circulants
    #  (a congruence for Bubnov, a similarity for Petrov); only the
    # matmul roundoff is symmetrised away, so nothing is measured here.
    M_wav = W_test @ M_vj @ W_inv
    D_wav = W_test @ D_vj @ W_inv
    K_wav = W_test @ K_vj @ W_inv
    G_wav = W_inv.T @ G_vj @ W_inv                 # ALWAYS a congruence: SPD
    G_wav = 0.5 * (G_wav + G_wav.T)
    if test_space == "dual":
        M_wav = np.eye(n)                          # , by biorthogonality
    else:
        M_wav = 0.5 * (M_wav + M_wav.T)
        D_wav = 0.5 * (D_wav - D_wav.T)
        K_wav = 0.5 * (K_wav + K_wav.T)

    # --- triad factors -------------------------------------------------
    # Skew-symmetric (Temam) advection.  The correction 1/2 <psit,(div u)w>
    # differs from the convective term only by moving the derivative from the
    # second field slot to the first, i.e. by transposing the stencil.  Hence
    #     Lambda_skew := Lambda + 1/2 Lambda^T
    # reproduces  (u.grad)w + 1/2 (div u) w  , at zero extra cost.
    Lam_skew_st = Lam_st + 0.5 * Lam_st.T

    kw = dict(n=n, W_syn=W_inv, W_test=W_test, rtol=triad_rtol)
    Lam_f = factor_triad_stencil(Lam_skew_st, pidx, 2.0 ** J, **kw)
    Lam_conv_f = factor_triad_stencil(Lam_st, pidx, 2.0 ** J, **kw)
    Gam_f = factor_triad_stencil(Gam_st, pidx, 1.0, **kw)

    if verbose:
        print(f"  -> triad stencils factorised: ranks {Lam_f.rank} (skew), "
              f"{Gam_f.rank} (Gamma) "
              f"({(Lam_f.nbytes + Gam_f.nbytes) / 1024:.1f} KiB, vs "
              f"{2 * n ** 3 * 8 / 1024 ** 2:.0f} MiB for the dense tensors)")
        cond = np.linalg.cond(G_wav)
        print(f"  -> mass matrix M is {'IDENTITY' if test_space == 'dual' else 'the primal Gram'}"
              f"; cond(G_wav) = {cond:.3e}")

    return WaveletOperators1D(
        M=M_wav, D=D_wav, K=K_wav, G=G_wav,
        Lam=Lam_f, Gam=Gam_f, Lam_conv=Lam_conv_f,
        M_vj=M_vj, D_vj=D_vj, K_vj=K_vj, G_vj=G_vj,
        W_fwd=W_fwd, W_inv=W_inv, W_test=W_test, W_test_inv=W_test_inv,
        J=int(J), jmin=int(jmin), test_space=test_space, raw_defects=raw)



# 6.  Matrix-free 2-D tensor-product operators

# With C-order flattening,  (A kron B) vec_C(X) = vec_C(A X B^T), hence
#     M2 U  = M U M^T,   Dx U = D U M^T,   Dy U = M U D^T,
#     Lap U = K U M^T + M U K^T.
# No n^2 x n^2 array is ever formed.

@dataclass
class Kron2DOperators:
    """Matrix-free tensor-product operators for one 2-D scalar field.

    M is the test-trial mass matrix and G the L2 metric; they coincide only
    for Bubnov (under Petrov M = I and K is only SIMILAR to a symmetric
    matrix).  Mass/Gram/derivative applications and diagnostics only; time
    stepping lives in ODE.build_scaling_basis_stepper.  Built by
    ``build_2d_operators`` only.
    """
    M: np.ndarray
    D: np.ndarray
    K: np.ndarray
    G: np.ndarray
    M_vj: np.ndarray
    K_vj: np.ndarray
    W_fwd: np.ndarray
    W_test_inv: np.ndarray
    W_syn: np.ndarray                 # synthesis (wavelet -> scaling), = W_inv
    ops1d: object                     # back-reference to the 1-D bundle

    def __post_init__(self):
        self.M = np.asarray(self.M, dtype=float)
        self.D = np.asarray(self.D, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        if self.M.ndim != 2 or self.M.shape[0] != self.M.shape[1]:
            raise ValueError("M must be square")
        self.n = self.M.shape[0]
        self.n_var = self.n * self.n
        self.G = np.asarray(self.G, dtype=float)
        self.Minv = np.linalg.inv(self.M)
        self.Ginv = np.linalg.inv(self.G)

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _as_mat(x, n):
        a = np.asarray(x)
        return a.reshape(n, n) if a.ndim == 1 else a

    def apply_mass_mat(self, U):
        U = self._as_mat(U, self.n); return self.M @ U @ self.M.T

    def apply_gram_mat(self, U):
        U = self._as_mat(U, self.n); return self.G @ U @ self.G.T

    def apply_dx_mat(self, U):
        U = self._as_mat(U, self.n); return self.D @ U @ self.M.T

    def apply_dy_mat(self, U):
        U = self._as_mat(U, self.n); return self.M @ U @ self.D.T

    def apply_lap_mat(self, U):
        U = self._as_mat(U, self.n)
        return self.K @ U @ self.M.T + self.M @ U @ self.K.T

    def solve_mass_mat(self, R):
        R = self._as_mat(R, self.n); return self.Minv @ R @ self.Minv.T

    def apply_mass(self, u):  return self.apply_mass_mat(u).reshape(-1)
    def apply_gram(self, u):  return self.apply_gram_mat(u).reshape(-1)
    def apply_dx(self, u):    return self.apply_dx_mat(u).reshape(-1)
    def apply_dy(self, u):    return self.apply_dy_mat(u).reshape(-1)
    def solve_mass(self, r):  return self.solve_mass_mat(r).reshape(-1)

    def weak_vorticity_mat(self, u, v):
        r"""omega from  M2 omega = Dx v - Dy u, in WAVELET coordinates.

        This is the DENSE REFERENCE, not the production path.  It is the same
        operator as STREAM.StreamMap.vorticity_s -- both are  A v - u A^T  with
        A = M^{-1} D -- but expressed in the wavelet basis rather than the
        scaling basis, and at O(n^3) rather than O(n^2 log n).  The identity

            vorticity_s(U, V) = W_inv @ weak_vorticity_mat(u_w, v_w) @ W_inv.T

        is asserted by STREAM.validate_vorticity_consistency; keep this method
        so that assertion has something to compare against, and use
        ``vorticity_s`` everywhere else.  Note the argument basis: u, v here
        are WAVELET coefficients (Kron2DOperators is built from ops1d.M, .D,
        the wavelet-basis operators), while vorticity_s takes scaling-basis
        arrays.  Mixing the two silently gives a wrong field.
        """
        U = self._as_mat(u, self.n); V = self._as_mat(v, self.n)
        return self.solve_mass_mat(self.apply_dx_mat(V) - self.apply_dy_mat(U))



# 7.   full-resolution (equal-order) Leray projector


@dataclass
class LerayProjector:
    """ discrete Leray projection, constraint tested at level J.

    Constraint.  C = [Dx, Dy],  Dx = D (x) M,  Dy = M (x) D,  rows indexed by
    the TEST functions: (C x)_{pq} = <phit_p phit_q, div u_h> (Petrov) or
    <phi_p phi_q, div u_h> (Bubnov).  dim(constraint space) = n^2.

    Projection.  The G2-ORTHOGONAL projection onto ker C (G2 = G (x) G is the
    true L2 metric; under Petrov-Galerkin the mass matrix is I and is NOT the
    metric):

        P x* = x* - G2^{-1} C^T q,      S q = C x*,      S = C G2^{-1} C^T,

    i.e. argmin_{C x = 0} ||x - x*||_{G2}.  Consequences used elsewhere:
        ||P x||_{G2} <= ||x||_{G2},   G2 P = P^T G2,   P^2 = P.
    The correction G2^{-1} C^T q is the L2-Riesz representer in V_J^2 of the
    constraint functionals, Pi_{V_J} grad(pt) with pt in the test space; it is
    NOT the momentum-equation pressure gradient [Dx; Dy] q for Petrov (which
    would give an oblique, non-contractive projection).

    Sylvester structure.  S = Ap (x) Bp + Bp (x) Ap,  Ap = D G^{-1} D^T (SPSD),
    Bp = M G^{-1} M^T (SPD).  With Ap Z = Bp Z diag(mu), Z^T Bp Z = I,
        S^{-1} R = Z [ (Z^T R Z) / (mu_i + mu_j) ] Z^T
    on range(S).  Since C x* is in range(C) = range(S) = ker(S)^perp, the
    kernel entries of Z^T R Z vanish and zeroing them is  (any two
    solutions of S q = C x* differ by an element of ker S = ker C^T, which
    the correction annihilates).

    LBB.  ker(D) = span{constant, Nyquist} (in the scaling basis; W_fwd of it
    in the wavelet basis), so ker(S) = ker(Ap) (x) ker(Ap) is 4-dimensional:
    the physical constant pressure and three spurious checkerboard modes.
    Dually, velocity content on the Nyquist lines has ZERO discrete
    divergence and passes through P untouched.  Beyond this  kernel the
    pair is only weakly stable: the symbol d(theta) vanishes LINEARLY at
    theta = pi, so the discrete inf-sup constant restricted to the mode next
    to Nyquist is O(h).  P remains an  G-orthogonal projector (the scheme
    and its energy identity do not depend on inf-sup); what degrades is the
    discrete pressure (never used here) and the strength of the weak
    constraint on the near-Nyquist band.  Monitor the energy fraction in the
    finest level as the resolution criterion.  ``filter_nyquist=True``
    (production) removes it with the G2-orthogonal projector

        Pi_N = N (x) N,   N = W_fwd (I - e e^T) W_inv = I - (W_fwd e)(W_inv^T e)^T,

    (e = (-1)^k / sqrt(n), the Nyquist mode in SCALING coordinates; in
    wavelet coordinates it is W_fwd e, which lives in the finest detail
    block).  N is the similarity image of the scaling-basis projector
    I - e e^T, G-orthogonal because e is an eigenvector of the circulant G.
    Pi_N and P commute, so P Pi_N is the G2-orthogonal projector onto
    {C u = 0} cap {no Nyquist content}.
    """
    ops: object
    filter_nyquist: bool
    rcond: float = None

    def __post_init__(self):
        M = self.ops.M
        D = self.ops.D
        self.n = M.shape[0]
        self.n_pressure = self.n * self.n
        Ginv = self.ops.Ginv
        Ap = D @ Ginv @ D.T
        Bp = M @ Ginv @ M.T
        Ap = 0.5 * (Ap + Ap.T)
        Bp = 0.5 * (Bp + Bp.T)
        self.mu, self.Z = eigh(Ap, Bp, check_finite=False)
        scale = float(np.max(np.abs(self.mu))) + 1e-300
        tol = (1e-10 if self.rcond is None else self.rcond) * scale
        self.kernel_dim_1d = int(np.count_nonzero(self.mu <= tol))
        den = self.mu[:, None] + self.mu[None, :]
        good = den > tol
        self.inv_den = np.where(good, 1.0 / np.where(good, den, 1.0), 0.0)
        self.rank = int(np.count_nonzero(good))
        self.kernel_dim = self.n_pressure - self.rank

        # ||C||_2 <= sqrt(2) ||D||_2 ||M||_2 : normaliser for residuals.
        self.C_norm_bound = float(np.sqrt(2.0) * np.linalg.norm(D, 2)
                                  * np.linalg.norm(M, 2))

        # Nyquist filter IN THE BASIS THE OPERATORS ACT ON.
        if self.filter_nyquist:
            Wf = np.asarray(self.ops.W_fwd, dtype=float)
            Ws = np.asarray(self.ops.W_syn, dtype=float)
            e = ((-1.0) ** np.arange(self.n)) / np.sqrt(self.n)
            self._nyq = np.eye(self.n) - np.outer(Wf @ e, Ws.T @ e)
            # build-time proof that this is the right vector: it spans ker(D)
            # together with the constant, and N is a projector.
            if np.max(np.abs(D @ (Wf @ e))) > 1e-8 * (np.abs(D).max() + 1e-300):
                raise RuntimeError("Nyquist filter: W_fwd e is not in ker(D).")
            if np.max(np.abs(self._nyq @ self._nyq - self._nyq)) > 1e-10:
                raise RuntimeError("Nyquist filter is not idempotent.")
        else:
            self._nyq = None

        if self.kernel_dim > 1 and not self.filter_nyquist:
            warnings.warn(
                f"Equal-order constraint space: S has a {self.kernel_dim}-dimensional "
                f"kernel ({self.kernel_dim - 1} spurious checkerboard modes beyond "
                f"the physical constant mode); velocity content on the Nyquist "
                f"lines is invisible to the projector.  Set filter_nyquist=True "
                f"to remove it.")

    # ---- operator applications ----------------------------------------
    def _split(self, x):
        n = self.n; n2 = n * n
        x = np.asarray(x)
        return x[:n2].reshape(n, n), x[n2:].reshape(n, n)

    def divergence_mat(self, x):
        U, V = self._split(x)
        return self.ops.D @ U @ self.ops.M.T + self.ops.M @ V @ self.ops.D.T

    def divergence(self, x):
        return self.divergence_mat(x).reshape(-1)

    def pressure_mat(self, x):
        R = self.divergence_mat(x)
        return self.Z @ ((self.Z.T @ R @ self.Z) * self.inv_den) @ self.Z.T

    def correction(self, q):
        """G2^{-1} C^T q, i.e. the discrete gradient in the L2 metric."""
        q = np.asarray(q).reshape(self.n, self.n)
        Gi, M, D = self.ops.Ginv, self.ops.M, self.ops.D
        cu = Gi @ (D.T @ q @ M) @ Gi.T
        cv = Gi @ (M.T @ q @ D) @ Gi.T
        return np.concatenate([cu.reshape(-1), cv.reshape(-1)])

    def project(self, x):
        """(P x, q): projected state and the pressure-like multiplier q."""
        x = np.asarray(x)
        if self._nyq is not None:
            U, V = self._split(x)
            x = np.concatenate([(self._nyq @ U @ self._nyq.T).reshape(-1),
                                (self._nyq @ V @ self._nyq.T).reshape(-1)])
        q = self.pressure_mat(x)
        return x - self.correction(q), q.reshape(-1)

    def residual_norm(self, x):
        return float(np.linalg.norm(self.divergence(x)))

    def normalised_residual(self, x):
        """||C x||_2 / (||C||_2 ||x||_2), with ||C||_2 replaced by an upper bound.

        Scale invariant and bounded by 1; roundoff level is ~1e-16 in float64.
        This is the quantity to report: the ratio ||C x_after||/||C x_before||
        is meaningless when the input is already (almost) divergence free,
        because it then divides the roundoff floor by a tiny number.
        """
        x = np.asarray(x)
        return float(self.residual_norm(x)
                     / (self.C_norm_bound * np.linalg.norm(x) + 1e-300))


# 8.  Public construction helpers


def build_2d_operators(ops1d):
    """Matrix-free 2-D operator bundle from the WaveletOperators1D object."""
    if not isinstance(ops1d, WaveletOperators1D):
        raise TypeError("build_2d_operators expects the WaveletOperators1D bundle")
    o = ops1d
    return Kron2DOperators(M=o.M, D=o.D, K=o.K, G=o.G,
                           M_vj=o.M_vj, K_vj=o.K_vj,
                           W_fwd=o.W_fwd, W_test_inv=o.W_test_inv,
                           W_syn=o.W_inv, ops1d=o)


def build_leray_projector_full(ops, *, filter_nyquist):
    """ equal-order (level-J) Leray projector (production: filter_nyquist=True)."""
    if not isinstance(ops, Kron2DOperators):
        raise TypeError("matrix-free API expects a Kron2DOperators instance")
    return LerayProjector(ops, filter_nyquist=filter_nyquist)



# 9.  Structural validation of the 1-D operator bundle


def validate_operators(ops_1d, verbose=True, tol=1e-10):
    """Structural validation of a WaveletOperators1D bundle.

    Checks  structural properties of the discretisation, so any failure
    indicates a build bug, not a tolerance issue:

        1. m, d, k, g stencils (anti)symmetric, measured on the RAW stencils
           (compute_operators symmetrises them afterwards);
        2. K_vj negative semi-definite (weak Laplacian);
        3. G_vj and G_wav positive definite (L2 metric; G_wav is a congruence
           of G_vj, so its symmetry is structural, not a test);
        4. W_fwd @ W_inv = I to roundoff (the only check of the DWT pair).
    """
    o = ops_1d
    if not isinstance(o, WaveletOperators1D):
        raise TypeError(f"validate_operators expects WaveletOperators1D, got {type(o)}")
    n = 2 ** o.J
    if o.raw_defects is None:
        raise RuntimeError("ops_1d has no raw_defects: rebuild it with compute_operators")

    # ---- (anti)symmetry: measured BEFORE the cleanup in compute_operators --
    checks = dict(o.raw_defects)
    # ---- spectra, RELATIVE to the operator scale (K_vj ~ 4^J) -------------
    kev = np.linalg.eigvalsh(0.5 * (o.K_vj + o.K_vj.T))
    checks['K_vj max eigenvalue / |K| (<= 0)'] = float(kev.max() / np.abs(kev).max())
    checks['G_vj min eigenvalue (> 0)'] = float(np.min(np.linalg.eigvalsh(o.G_vj)))
    checks['G_wav min eigenvalue (> 0)'] = float(np.min(np.linalg.eigvalsh(o.G)))
    # ----  change of basis --------------------------------------------
    checks['W_fwd @ W_inv = I'] = float(np.max(np.abs(o.W_fwd @ o.W_inv - np.eye(n))))

    # ---- verdict -----------------------------------------------------------
    failed = []
    for name, val in checks.items():
        if '(> 0)' in name:
            ok = val > 0.0
        elif '(<= 0)' in name:
            ok = val <= tol
        elif name.startswith('W_fwd'):
            ok = val < 1e-8
        else:
            ok = val < max(tol, 1e-8)      # stencils come from an eigensolver
        if verbose:
            print(f"    [validate_operators] {name:<38s} {val:.3e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)

    if failed:
        raise RuntimeError(
            f"Operator structural validation FAILED for: {failed}. "
            "Do not trust downstream operators.")
    return o