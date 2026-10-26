r"""GROUP.py --  Gaussian algebra of the dyadic shift group.

Why
---
Every linear object of the LINEAR (Stokes + drag) problem -- Sigma_lin, the
dequantised covariance Sz, the OU coefficients a, s, Sg, the surrogate D --
commutes with the dyadic shift group  G = (Z_m)^2,  m = 2^jmin,  which acts on
psi-wavelet coordinates by  permutations (ODE.dyadic_shift_permutation_1d)
and FREELY (every orbit has |G| = 4^jmin elements).  Relabel coordinate i as
(O, g): orbit O and group element g, i = T[O, g].  A G-invariant matrix M
then satisfies  M[(O,g),(O',g')] = c_{OO'}(g' - g)  and the 2-D DFT over g
block-diagonalises it:

    (M v)^(k) = C(k) v^(k),   v^(k) = sum_g v(g) e^{-i theta_k . g},
    C(k)      = sum_h c(h) e^{+i theta_k . h},     theta_k = 2 pi k / m,

with |G| Hermitian blocks C(k) of size n_orb x n_orb (n_orb = 4^(J-jmin)).
At J = 7, jmin = 4 that is 256 blocks of 64 x 64 instead of one dense
16384 x 16384 matrix: the linear-Gaussian oracle, and a FULL-covariance
Gaussian surrogate, become  and cheap at every resolution.

Conventions: blocks are stored as (m, m, n_orb, n_orb) complex arrays.
"""
from __future__ import annotations

import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from ODE import dyadic_shift_permutation_1d
from FORCING import build_orbit_map_1d, build_orbit_map_2d



# 1.  Coordinates (O, g)

def dyadic_orbit_table(ops1d) -> np.ndarray:
    """T (n_orb, m, m) with T[O, gx, gy] = psi-wavelet index of g . k_O.

    Orbit ids agree with FORCING.build_orbit_map_2d(J, jmin, n_comp=1).
    """
    J, jmin = ops1d.J, ops1d.jmin
    n, m = 2 ** J, 2 ** jmin
    p = dyadic_shift_permutation_1d(ops1d.W_fwd, ops1d.W_inv, J, jmin, 1)
    o1 = build_orbit_map_1d(J, jmin)
    no = int(o1.max()) + 1
    T1 = np.empty((no, m), int)
    for o in range(no):
        i = int(np.flatnonzero(o1 == o)[0])
        for g in range(m):
            T1[o, g] = i
            i = int(p[i])
    T = (T1[:, None, :, None] * n + T1[None, :, None, :]).reshape(no * no, m, m)
    orbit = build_orbit_map_2d(J, jmin, n_comp=1)
    if not all(np.all(orbit[T[O]] == O) for O in range(T.shape[0])):
        raise RuntimeError("orbit table disagrees with build_orbit_map_2d")
    if len(np.unique(T)) != n * n:
        raise RuntimeError("orbit table is not a bijection: the action is not free")
    return T


def to_group(v: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(..., n^2) -> (..., n_orb, m, m)."""
    return np.asarray(v)[..., T]


def from_group(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    out = np.empty(V.shape[:-3] + (T.size,), V.dtype)
    out[..., T.reshape(-1)] = V.reshape(V.shape[:-3] + (-1,))
    return out



# 2.  Blocks of a G-invariant matrix

def blocks_from_columns(cols: np.ndarray, T: np.ndarray) -> np.ndarray:
    """cols (n_orb_in, n^2): column k_{O'} of M for every input orbit O'.
    Returns C (m, m, n_orb, n_orb_in)."""
    X = to_group(cols, T)                                # (O', O, gx, gy)
    Xh = np.fft.fft2(X, axes=(-2, -1))                   # sum_g x(g) e^{-ikg}
    return np.transpose(Xh, (2, 3, 1, 0))


def apply_blocks(C: np.ndarray, v: np.ndarray, T: np.ndarray) -> np.ndarray:
    """M v for a G-invariant M given by its blocks; v (..., n^2) real."""
    Vh = np.fft.fft2(to_group(v, T), axes=(-2, -1))      # (..., O, kx, ky)
    Wh = np.einsum('xyab,...bxy->...axy', C, Vh)
    return np.real(from_group(np.fft.ifft2(Wh, axes=(-2, -1)), T))


def dense_from_blocks(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Dense M (validation only, small J)."""
    n2 = T.size
    return apply_blocks(C, np.eye(n2), T).T


def orbit_diagonal(C: np.ndarray) -> np.ndarray:
    """diag M, one value per orbit: mean over k of Re diag C(k)."""
    return np.real(np.einsum('xyaa->a', C)) / (C.shape[0] * C.shape[1])



# 3.  Sigma_lin and the dequantised covariance, blockwise and 

def sigma_lin_blocks(response_fn, noise, T: np.ndarray, dt: float) -> np.ndarray:
    """Blocks of Sigma_lin = dt sum_s A_s A_s^T from ONE response per input orbit.

    ``response_fn`` = TLM._linear_response_fn(...) (zeta -> (S, n^2)),
    run on e_{k_O'} for each input-orbit representative.  A_s is
    G-equivariant, so its column at k_O' determines the O'-th column of every
    block A_s^(k), and
        Sigma^(k) = dt sum_s sum_O' a_{s,O'}(k) a_{s,O'}(k)^H
    is accumulated one input orbit at a time (memory O(|G| n_orb^2)).
    'stream' forcing only: input and output share the orbit table T.
    """
    if getattr(noise, "coords", "stream") != "stream":
        raise NotImplementedError("blockwise Sigma_lin implemented for 'stream' forcing")
    n_orb, m = T.shape[0], T.shape[1]
    resp = jax.jit(response_fn)
    C = np.zeros((m, m, n_orb, n_orb), complex)
    for O in range(n_orb):
        e = np.zeros(noise.n_in)
        e[T[O, 0, 0]] = 1.0
        R = np.asarray(resp(jnp.asarray(e)), np.float64)                    # (S, n^2)
        a = np.transpose(np.fft.fft2(to_group(R, T), axes=(-2, -1)), (0, 2, 3, 1))
        C += np.einsum('sxya,sxyb->xyab', a, np.conj(a))                    # (m, m, O, O)
    return dt * C


def dequantised_blocks(C_sig: np.ndarray, B: np.ndarray, D: np.ndarray,
                       T: np.ndarray) -> np.ndarray:
    """Blocks of Sz = T_D Sigma T_D^T + B A^{-1} B^T (TLM.linear_gaussian_oracle),
    T_D = I - B A^{-1} B^T D^{-1},  A = B^T D^{-1} B.  Needs only n_orb columns."""
    reps = T[:, 0, 0]
    n2 = T.size
    A = B.T @ (B / D[:, None])
    Ai = np.linalg.inv(A)
    E = np.zeros((n2, len(reps)))
    E[reps, np.arange(len(reps))] = 1.0
    Tt = E - (B @ (Ai @ (B.T @ E))) / D[:, None]         # T_D^T e_k
    S1 = apply_blocks(C_sig, Tt.T, T).T                  # Sigma T_D^T e_k
    cols = S1 - B @ (Ai @ (B.T @ (S1 / D[:, None]))) + B @ (Ai @ (B.T @ E))
    return blocks_from_columns(cols.T, T)


def dequantised_fixed_point(C_sig: np.ndarray, B: np.ndarray, T: np.ndarray,
                            tol: float = 1e-10, max_it: int = 100):
    """ fixed point D = orbit-diag Cov(z_D) of the dequantiser.

    Replaces the Monte-Carlo control-variate loop of
    TLM.sigma_bar_linear: diag(Sz(D)) is computed in closed form from
    the blocks, so sigma_bar_z carries no sampling error.
    Returns (D per coefficient, D per orbit, n_iterations).
    """
    orbit_of = np.empty(T.size, int)
    for O in range(T.shape[0]):
        orbit_of[T[O].ravel()] = O
    d_orb = orbit_diagonal(C_sig)                     # start: Sigma_bar_r
    for it in range(1, max_it + 1):
        Cz = dequantised_blocks(C_sig, B, d_orb[orbit_of], T)
        new = orbit_diagonal(Cz)
        change = float(np.max(np.abs(new / d_orb - 1.0)))
        d_orb = new
        if change < tol:
            break
    # Cz was built with the previous iterate; at convergence (change < tol)
    # the difference is O(tol), so it is returned rather than recomputed.
    return d_orb[orbit_of], d_orb, it, Cz



# 4.  Linear-Gaussian oracle at any J

def _oracle_mse_blocks(Cu, cu_diag, D, sig2, lam, tau):
    """(base, mse) per orbit at diffusion time tau (see linear_gaussian_oracle_blocks)."""
    a = np.exp(-lam * tau)
    s = sig2 * (-np.expm1(-2.0 * lam * tau))
    Sg = a * a * D + s
    al, be = a * np.sqrt(D / Sg), np.sqrt(s / Sg)
    CFR = np.diag(al * be)[None, None] - be[None, None, :, None] * Cu * al[None, None, None, :]
    CR = al[None, None, :, None] * Cu * al[None, None, None, :] + np.diag(be ** 2)[None, None]
    H = lambda X: np.swapaxes(X, -1, -2).conj()
    K = H(np.linalg.solve(H(CR), H(CFR)))                # CFR CR^{-1}
    base = al ** 2 + be ** 2 * cu_diag
    return base, base - orbit_diagonal(K @ H(CFR))


def linear_gaussian_oracle_blocks(C_z: np.ndarray, D_orb: np.ndarray,
                                  sig2_orb: np.ndarray, lam_orb: np.ndarray,
                                  level_orb: np.ndarray, levels, taus) -> dict:
    """Same quantity as TLM.linear_gaussian_oracle (unit weighting,
    per-level mean, tau on the grid ``taus``), from the blocks of Sz.

    For whitened u = z / sqrt(D), Cu = Cov(u), alpha = a sqrt(D/Sg),
    beta = sqrt(s/Sg):  F_tgt = alpha eps - beta u,  Rc = alpha u + beta eps,
        MSE* = diag Cov F - diag[ Cov(F,Rc) Cov(Rc)^{-1} Cov(Rc,F) ],
    every factor G-invariant, hence evaluated block by block.
    D_orb, sig2_orb, lam_orb, level_orb: one value per orbit.
    Output keys as the dense oracle: loss, base_loss, gain, levels (MSE at
    SNR_j = 1), plus gain_per_level (share of the gain carried by level j).
    """
    d = 1.0 / np.sqrt(D_orb)
    Cu = C_z * d[None, None, :, None] * d[None, None, None, :]
    cu_diag = orbit_diagonal(Cu)
    levels = list(levels)

    def level_mean(v):
        return np.array([v[level_orb == j].mean() for j in levels])

    L, Bs = [], []
    per = np.zeros((len(levels), 2))
    for tau in taus:
        b, m_ = _oracle_mse_blocks(Cu, cu_diag, D_orb, sig2_orb, lam_orb, tau)
        lb, lm = level_mean(b), level_mean(m_)
        L.append(lm.mean())
        Bs.append(lb.mean())
        per += np.stack([lb, lm], 1) / len(taus)
    out = {"loss": float(np.mean(L)), "base_loss": float(np.mean(Bs))}
    out["gain"] = out["base_loss"] - out["loss"]
    out["gain_per_level"] = {int(j): float((per[i, 0] - per[i, 1]) / len(levels))
                             for i, j in enumerate(levels)}
    kap = D_orb / sig2_orb
    out["levels"] = {}
    for j in levels:
        mk = level_orb == j
        tau_j = np.log1p(kap[mk].mean()) / (2.0 * lam_orb[mk][0])      # SNR_j = 1
        b, m_ = _oracle_mse_blocks(Cu, cu_diag, D_orb, sig2_orb, lam_orb, tau_j)
        out["levels"][int(j)] = (float(b[mk].mean()), float(m_[mk].mean()))
    return out



# 5.  'stream' forcing

def sigma_bar_linear_(ops1d, projector, stream, noise, dt: float, nu: float,
                           mu_d: float, stride: int, orbit: np.ndarray,
                           dtype=None, verbose: bool = True):
    """(Sigma_bar_z, Sigma_bar_r, blocks) with NO Monte Carlo anywhere.

    Sigma_bar_r : orbit diagonal of Sigma_lin            (identical to TLM)
    Sigma_bar_z :  dequantiser fixed point          (TLM: MC, ~1e-2 error)
    blocks      : dict(T, sig=blocks of Sigma_lin, z=blocks of Sz at the fixed
                  point, B=kernel basis), reused by the oracle.
    """
    from TLM import _linear_response_fn
    from STREAM import curl_kernel_basis_w
    dtype = jnp.result_type(float) if dtype is None else dtype
    T = dyadic_orbit_table(ops1d)
    orbit = np.asarray(orbit)
    if not all(np.all(orbit[T[O]] == O) for O in range(T.shape[0])):
        raise ValueError("orbit labels differ from the dyadic orbit table")
    C_sig = sigma_lin_blocks(_linear_response_fn(ops1d, projector, stream, noise, dt,
                                                 nu, mu_d, stride, dtype),
                             noise, T, dt)
    B = curl_kernel_basis_w(stream)
    sbz, sbz_orb, it, C_z = dequantised_fixed_point(C_sig, B, T)
    sbr = orbit_diagonal(C_sig)[orbit]
    if verbose:
        k = sbr / sbz
        print(f"   [GROUP] Sigma_lin  ({T.shape[0]} orbit columns, {T.shape[1] ** 2} "
              f"blocks of {T.shape[0]}x{T.shape[0]}); range [{sbr.min():.2e}, {sbr.max():.2e}]; "
              f"dequantised fixed point  after {it} it.: Sigma_bar_r / Sigma_bar_z in "
              f"[{k.min():.3f}, {k.max():.3f}]")
    return sbz, sbr, {"T": T, "sig": C_sig, "z": C_z, "B": B}


def oracle_from_diffusion(blocks: dict, diffusion, sigma_bar_z: np.ndarray,
                          n_tau: int = 48, taus=None, verbose: bool = True) -> dict:
    """TLM.linear_gaussian_oracle, blockwise (any J): same output keys,
        plus 'gain_per_level' (the share of the gain carried by each level).
    
        SCOPE.  This is an expectation under the LINEAR (Stokes + drag) law with
        the B0 surrogate.  It is the correct floor ONLY when the data are that
        law (BENCH.run_tg_diffusion_benchmark, advection off).  For nonlinear
        data use ``empirical_linear_floor``, which is paired with the network. ``taus``: pass floor['taus'] to compare
    with GROUP.empirical_linear_floor on IDENTICAL quadrature nodes."""
    T = blocks["T"]
    reps = T[:, 0, 0]
    if taus is None:                       # same rule as empirical_linear_floor
        edges = np.linspace(np.log(diffusion.t_min), np.log(diffusion.T_diff), n_tau + 1)
        taus = np.exp(0.5 * (edges[1:] + edges[:-1]))
    out = linear_gaussian_oracle_blocks(
        blocks["z"], np.asarray(sigma_bar_z, float)[reps],
        np.asarray(diffusion.sigma_sq_diag, float)[reps],
        np.asarray(diffusion.lambda_diag, float)[reps],
        np.asarray(diffusion.level_per_coeff)[reps], diffusion.levels, np.asarray(taus))
    if verbose:
        print(f"   [oracle, blockwise] linear-Gaussian gain {out['gain']:.2e}; per level "
              + ", ".join(f"j={j}: {g:.2e}" for j, g in out["gain_per_level"].items()))
    return out


# 6.  Paired EMPIRICAL linear floor (the reference for a trained network)

def empirical_linear_floor(dataset, diffusion, sigma_bar_z, idx_fit, idx_eval, *,
                           dequant, g=None, apply_F=None, params=None,
                           n_tau: int = 16, chunk: int = 64, ridge: float = 1e-8,
                           seed: int = 0, verbose: bool = True) -> dict:
    r"""Best x-INDEPENDENT linear predictor of the training target, fitted on
    the DATA and scored in the network's own loss, paired with the network.

    Definition.  For each tau on a quadrature grid of tau ~ logU[t_min, T],
        K(tau) = argmin_K  E_data || F_tgt - K Rc ||^2,
    K restricted to G-invariant matrices (the dyadic-shift augmentation makes
    the Bayes-optimal linear map G-invariant, so this is no restriction):
        K^(k) = E[F^(k) Rc^(k)H] E[Rc^(k) Rc^(k)H]^{-1},   256 blocks 64 x 64.
    Fitted on ``idx_fit`` (training pairs), scored on ``idx_eval`` with the
    SAME (tau, eps, dequantiser) draws as F = 0 and as the network, so
        excess = loss_lin - loss_net
    is a PAIRED per-sample difference with a cluster (trajectory) SE.

    Why not GROUP.oracle_from_diffusion.  That oracle is an expectation under
    the LINEAR law N(0, Sz) with the B0 surrogate; the network is scored on
    the nonlinear data with the rung's surrogate.  On the J=7 run the data
    variance exceeds the linear one by up to ~80x at j = 6 (kappa_min 0.013),
    so the two numbers are not in the same units: the "floor" was identical
    for rungs whose F = 0 losses were 3.36 / 1.004 / 0.9998.

    What '+' means.  The floor contains everything a linear, x-independent
    map can extract (full marginal covariance, cross-scale included).  A
    network beating it has learned x-DEPENDENT structure: mean shift,
    x-dependent (TLM, cross-scale) covariance, or non-Gaussianity.  It does
    NOT separate the three; non-Gaussianity is decided at fixed x (third
    cumulant, GRID.mean_shift_diagnostic), never from a loss.

    Also returned: 'xmse', the denoising error sum_i (rhat_i - z_i)^2 / sigma_i^2
    (sigma^2 = diffusion.sigma_sq_diag).  It is invariant under a change of
    surrogate (rung) PROVIDED the forward process (sigma_sq_diag, lambda,
    [t_min, T]) is shared by the rungs; the F-loss is not.
    Unit weighting only.
    """
    from DIFF import make_batch_loader, make_sbar_fn, cluster_se
    from TLM import (ou_coefficients, baseline_coefficients, whiten,
                           f_target, conditional_log_snr, denoise)
    T = dyadic_orbit_table(diffusion.ops_1d)
    n_orb, m = T.shape[0], T.shape[1]
    f32 = jnp.float32
    load = make_batch_loader(dataset, diffusion, g=g)
    sbar_of = make_sbar_fn(diffusion, sigma_bar_z)
    lam = jnp.asarray(diffusion.lambda_diag, f32)[None]
    sig = jnp.asarray(diffusion.sigma_sq_diag, f32)[None]
    mk = jnp.asarray(diffusion.level_masks)
    cnt = jnp.sum(mk, axis=1)
    masks = np.asarray(diffusion.level_masks, np.float64)          # (L, n^2)
    msum = masks.sum(1)
    sig_np = np.asarray(diffusion.sigma_sq_diag, np.float64)
    mscale = jnp.asarray(diffusion.m_scale_diag, f32)
    xscale = jnp.asarray(diffusion.x_scale_diag, f32)
    edges = np.linspace(np.log(diffusion.t_min), np.log(diffusion.T_diff), n_tau + 1)
    taus = np.exp(0.5 * (edges[1:] + edges[:-1]))       # midpoint rule, tau ~ logU
    H = lambda X: np.conj(X).transpose(0, 2, 1)
    idx_fit, idx_eval = np.asarray(idx_fit), np.asarray(idx_eval)

    def draw(batch, tau, key):
        r, pm, px, g = batch
        k1, k2 = random.split(key)
        z = dequant(r, g, k1)
        sbar = sbar_of(g)
        a, s = ou_coefficients(lam, sig, f32(tau))
        Sg, al, be = baseline_coefficients(a, s, sbar)   # al = c_out, be = c_skip
        eps = random.normal(k2, z.shape, f32)
        R = a * z + jnp.sqrt(s) * eps
        lsnr = (conditional_log_snr(a, s, sbar) @ mk.T) / cnt
        return dict(z=z, R=R, a=a, s=s, sbar=sbar, Rc=whiten(R, Sg),
                    u=z / jnp.sqrt(sbar), al=al, be=be,
                    tgt=f_target(z, eps, a, s, sbar), cm=pm / mscale,
                    cx=px / xscale, lsnr=lsnr)

    def hat(v):                        # (N, n^2) real -> (m*m, n_orb, N)
        X = np.fft.fft2(to_group(np.asarray(v, np.float64), T), axes=(-2, -1))
        return X.transpose(2, 3, 1, 0).reshape(m * m, n_orb, -1)

    def unhat(Y, N):                   # inverse of hat
        X = Y.reshape(m, m, n_orb, N).transpose(3, 2, 0, 1)
        return np.real(from_group(np.fft.ifft2(X, axes=(-2, -1)), T))

    def per_level(E2):                 # (N, n^2) -> (N, L)
        return E2 @ masks.T / msum

    # ---- pass 1: fit K(tau), data loaded once --------------------------------
    # PLUG-IN moments, NOT a regression of F_tgt on Rc.  With u = z/sqrt(Sigma_bar),
    # F_tgt = al eps - be u,  Rc = al u + be eps  (al, be orbit-constant per sample)
    # and eps ~ N(0, I) independent of (u, x), LY:
    #     E[Rc Rc^T] = E[al u u^T al] + E[be^2] I,
    #     E[F  Rc^T] = E[al be] I     - E[be u u^T al].
    # The eps terms are known in closed form, so only the u-moments are
    # sampled.  Regressing F on Rc instead re-estimates the eps covariance
    # from N samples and pays the OLS out-of-sample inflation 1 + p/(N-p-1)
    # (p = n_orb): 1.7 at the smoke size (floor WORSE than F = 0), ~1 % at
    # J = 7 -- the size of 2 SE of the network gain, and anti-conservative.
    # The plug-in K is  at both ends of the tau range (u-free at large
    # tau, F = -be u + known noise at small tau); estimation error enters
    # only at intermediate tau, through the u-moments.
    n_fit = len(idx_fit)
    if n_fit < 5 * n_orb:
        warnings.warn(f"empirical_linear_floor: {n_fit} fitting pairs for {n_orb}x{n_orb} "
                      "blocks (< 5 per parameter row): the floor is not estimable; "
                      "returning NaN")
    key = random.PRNGKey(seed)
    Saa = np.zeros((n_tau, m * m, n_orb, n_orb), complex)     # sum hat(al u) hat(al u)^H
    Sba = np.zeros_like(Saa)                                   # sum hat(be u) hat(al u)^H
    mbe2 = np.zeros((n_tau, n_orb))          # sum_{g} c_skip_{O,g}^2 , summed over pairs
    malbe = np.zeros((n_tau, n_orb))         # sum_{g} (c_out c_skip)_{O,g}
    orb = np.asarray(diffusion.orbit_per_coeff)   # same labels as T (asserted in dyadic_orbit_table)
    for c, s0 in enumerate(range(0, n_fit, chunk)):
        batch = load(idx_fit[s0:s0 + chunk])
        for it, tau in enumerate(taus):
            d = draw(batch, tau, random.fold_in(random.fold_in(key, 0), c * n_tau + it))
            al, be = np.asarray(d["al"], np.float64), np.asarray(d["be"], np.float64)
            u = np.asarray(d["u"], np.float64)
            Ua, Ub = hat(al * u), hat(be * u)
            Saa[it] += Ua @ H(Ua)
            Sba[it] += Ub @ H(Ua)
            mbe2[it] += np.bincount(orb, weights=(be ** 2).sum(0), minlength=n_orb)
            malbe[it] += np.bincount(orb, weights=(al * be).sum(0), minlength=n_orb)
    # For every k: E[hat(c eps)^(k) hat(c' eps)^(k)H] = diag_O( sum_g c_{O,g} c'_{O,g} ).
    # c_out, c_skip are per coefficient (orbit-constant only for rungs L1, L1.5).
    K = []
    for it in range(n_tau):
        SRR = Saa[it] / n_fit + np.diag(mbe2[it] / n_fit)[None]
        SFR = np.diag(malbe[it] / n_fit)[None] - Sba[it] / n_fit
        reg = ridge * float(np.mean(np.real(np.trace(SRR, axis1=1, axis2=2)))) / n_orb
        K.append(H(np.linalg.solve(SRR + reg * np.eye(n_orb)[None], H(SFR))))
    del Saa, Sba

    # ---- pass 2: paired scoring on idx_eval ----------------------------------
    names = ("F0", "lin") + (("net",) if apply_F is not None else ())
    L = masks.shape[0]
    loss = {k: np.zeros((len(idx_eval), L)) for k in names}
    xmse = {k: np.zeros((len(idx_eval), L)) for k in names}
    for c, s0 in enumerate(range(0, len(idx_eval), chunk)):
        sl = slice(s0, s0 + chunk)
        batch = load(idx_eval[sl])
        for it, tau in enumerate(taus):
            d = draw(batch, tau, random.fold_in(random.fold_in(key, 1), c * n_tau + it))
            N = d["Rc"].shape[0]
            tgt = np.asarray(d["tgt"], np.float64)
            F = {"F0": np.zeros_like(tgt), "lin": unhat(K[it] @ hat(d["Rc"]), N)}
            if apply_F is not None:
                F["net"] = np.asarray(apply_F(params, d["Rc"], d["cm"], d["cx"],
                                              d["lsnr"]), np.float64)
            z = np.asarray(d["z"], np.float64)
            for k in names:
                loss[k][sl] += per_level((F[k] - tgt) ** 2) / n_tau
                rh = np.asarray(denoise(d["R"], d["a"], d["s"], d["sbar"],
                                        jnp.asarray(F[k], f32)), np.float64)
                xmse[k][sl] += per_level((rh - z) ** 2 / sig_np) / n_tau

    if n_fit < 5 * n_orb:
        for dct in (loss, xmse):
            dct["lin"][:] = np.nan
    traj = np.asarray(dataset["traj_id"])[idx_eval]
    tot = {k: v.mean(1) for k, v in loss.items()}         # unit weighting: level mean
    out = {"taus": taus, "loss": {k: float(v.mean()) for k, v in tot.items()},
           "xmse": {k: float(v.mean(1).mean()) for k, v in xmse.items()},
           "per_level": {int(j): {k: float(loss[k][:, i].mean()) for k in names}
                         for i, j in enumerate(diffusion.levels)}}
    out["gain_lin"] = out["loss"]["F0"] - out["loss"]["lin"]
    out["gain_lin_se"] = cluster_se(tot["F0"] - tot["lin"], traj)
    # A cluster SE from G clusters has G - 1 degrees of freedom: below 5 it is
    # not a usable scale for a 3-sigma rule (DIFF warns; here we refuse).
    out["n_clusters"] = int(len(np.unique(traj)))
    if out["n_clusters"] < 5:
        out["gain_lin_se"] = float("nan")
    if apply_F is not None:
        dlt = tot["lin"] - tot["net"]
        out["gain_net"] = out["loss"]["F0"] - out["loss"]["net"]
        out["excess"] = float(dlt.mean())
        out["excess_se"] = (cluster_se(dlt, traj) if out["n_clusters"] >= 5
                            else float("nan"))
    if verbose:
        msg = (f"   [floor, empirical+paired] F-loss: F=0 {out['loss']['F0']:.5f} | "
               f"linear {out['loss']['lin']:.5f}")
        if apply_F is not None:
            msg += (f" | network {out['loss']['net']:.5f};  network - linear = "
                    f"{-out['excess']:+.5f} +- {out['excess_se']:.5f}")
        print(msg)
        print("      per level (tau-averaged F-loss): " + "  ".join(
            f"j={j}: " + "/".join(f"{v:.3f}" for v in d_.values())
            for j, d_ in out["per_level"].items()) + f"   [{'/'.join(names)}]")
    return out