r"""STREAM.py --  discrete stream function: unconstrained coordinates of V.

Why
---
The discrete divergence-free space V = ker C (C = [D (x) M, M (x) D]) is a
linear subspace of dimension ~ n^2 inside the 2 n^2-dimensional velocity
space.  A diffusion model run in velocity coordinates would have to learn a
law supported on a subspace: the Gaussian surrogate N(0, diag Sigma_bar) is
not supported on V, the level-dependent OU drift does not preserve V, and
projecting the sample at the end removes about half of its variance
(measured: 0.54 of the target energy at J = 5 with F = 0).  Parameterising
V removes the constraint instead of enforcing it.

Construction
------------
With the discrete derivative  d_h := M^{-1} D  (circulant; symbol
d_hat / m_hat),

    curl_h(psi) := ( psi d_h^T ,  -d_h psi )          [U = psi d_h^T, V = -d_h psi]

satisfies C curl_h(psi) = D psi d_h^T M^T - M d_h psi D^T = D psi D^T - D psi D^T = 0
for EVERY psi (all 1-D operators are circulants, hence commute).  For the
Bubnov scheme (M = G) this is 

    u_h = Pi_{V_J}( d_y psi_h ),   v_h = -Pi_{V_J}( d_x psi_h ),

Pi_{V_J} the L2 projection, psi_h = sum psi_s[a,b] phi_a(x) phi_b(y): psi_h
is a genuine discrete stream function, not an ad-hoc parameterisation.

Range.  In Fourier, U^ = s(t2) psi^, V^ = -s(t1) psi^, s the symbol of d_h.
s vanishes only at t = 0 and t = pi, so with the Nyquist filter Pi_N (which
removes the t = pi lines from u, v) and zero-mean forcing:

    Pi_N curl_h : Psi -> V_F  is ONTO  V_F cap {zero mean flow},
    ker(Pi_N curl_h) = {constants} + {Nyquist lines of psi}      (2n dims).

The only part of V_F outside the range is the spatial mean flow, which is
 conserved by the Bubnov advection + projection and is absent from the
forcing (FORCING), so r never has any.

Pseudo-inverse.  psi^ = (conj(s2) U^ - conj(s1) V^) / (|s1|^2 + |s2|^2),
set to 0 where the denominator vanishes or on Nyquist lines: the
Moore-Penrose inverse of Pi_N curl_h (the FFT is unitary),  on its
range.  Everything below is O(n^2 log n), float-safe, vmappable.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from OPERATORS import sandwich, build_2d_operators
from ODE import _resolve_dtype


def _symbol(C):
    """s(theta_k) with (C X)^ = s(theta_1) X^ and (X C^T)^ = s(theta_2) X^.

    For the circulants of OPERATORS.compute_operators, C[i, (i+p) % n] = row[p],
    so (C x)[i] = sum_p row[p] x[i+p] and s(theta) = sum_p row[p] e^{i p theta}.
    """
    row = np.asarray(C, dtype=float)[0]
    return np.conj(np.fft.fft(row))


class StreamMap:
    """psi (scaling or wavelet coords, n x n)  <->  velocity (U, V).

    Coordinates:
        *_s : scaling-basis n x n arrays (physical grid)
        *_w : flattened wavelet-basis vectors (n^2 for psi, 2 n^2 for (u, v))
    All maps accept leading batch dimensions.  ``filter_nyquist`` must be the
    projector's (one velocity space for the whole pipeline).
    """

    def __init__(self, ops1d, filter_nyquist: bool, dtype=None):
        self.n = n = int(ops1d.M_vj.shape[0])
        self.n2 = n * n
        self.filter_nyquist = bool(filter_nyquist)
        dt_ = _resolve_dtype(dtype)
        cdt = jnp.complex128 if dt_ == jnp.float64 else jnp.complex64
        Mv = np.eye(n) if ops1d.test_space == "dual" else np.asarray(ops1d.M_vj)
        sym = _symbol(ops1d.D_vj) / _symbol(Mv)          # symbol of d_h = M^{-1} D
        keep = np.ones(n)
        if self.filter_nyquist:
            keep[n // 2] = 0.0                           # theta = pi
        k2 = keep[:, None] * keep[None, :]               # Pi_N = N (x) N, both lines
        s1 = k2 * sym[:, None]                           # symbol on axis 0 (x)
        s2 = k2 * sym[None, :]                           # symbol on axis 1 (y)
        den = np.abs(s1) ** 2 + np.abs(s2) ** 2
        good = den > 1e-12 * den.max()
        inv = np.where(good, 1.0 / np.where(good, den, 1.0), 0.0)
        self._s1 = jnp.asarray(s1, cdt)
        self._s2 = jnp.asarray(s2, cdt)
        self._inv = jnp.asarray(inv, dt_)
        self._Wf = jnp.asarray(ops1d.W_fwd, dt_)
        self._Wi = jnp.asarray(ops1d.W_inv, dt_)
        self.dtype = dt_

    # ---- scaling basis -------------------------------------------------
    def curl_s(self, Psi):
        """(U, V) = Pi_N curl_h(Psi), Psi (..., n, n)."""
        P = jnp.fft.fft2(Psi, axes=(-2, -1))
        U = jnp.real(jnp.fft.ifft2(self._s2 * P, axes=(-2, -1)))
        V = jnp.real(jnp.fft.ifft2(-self._s1 * P, axes=(-2, -1)))
        return U.astype(self.dtype), V.astype(self.dtype)

    def psi_s(self, U, V):
        """Moore-Penrose inverse of Pi_N curl_h ( on V_F, zero-mean flow)."""
        Uh = jnp.fft.fft2(U, axes=(-2, -1))
        Vh = jnp.fft.fft2(V, axes=(-2, -1))
        P = (jnp.conj(self._s2) * Uh - jnp.conj(self._s1) * Vh) * self._inv
        return jnp.real(jnp.fft.ifft2(P, axes=(-2, -1))).astype(self.dtype)

    def vorticity_s(self, U, V):
        """omega_h = d_x v - d_y u with the SAME discrete derivative d_h as
        curl_h (so vorticity_s(curl_s(psi)) = -(d_h^2 (+) d_h^2) psi).
        Scaling-basis arrays; visualisation and one-point statistics."""
        Uh = jnp.fft.fft2(U, axes=(-2, -1))
        Vh = jnp.fft.fft2(V, axes=(-2, -1))
        return jnp.real(jnp.fft.ifft2(self._s1 * Vh - self._s2 * Uh,
                                      axes=(-2, -1))).astype(self.dtype)

    # ---- wavelet basis -------------------------------------------------
    def _s2w1(self, X):
        return sandwich(self._Wf, X, self._Wf.T).reshape(X.shape[:-2] + (self.n2,))

    def _w2s1(self, x):
        return sandwich(self._Wi, x.reshape(x.shape[:-1] + (self.n, self.n)), self._Wi.T)

    def psi_w_from_velocity_s(self, U, V):
        return self._s2w1(self.psi_s(U, V))

    def psi_w_from_state(self, x):
        """flattened wavelet velocity state (..., 2 n^2) -> psi_w (..., n^2)."""
        U = self._w2s1(x[..., :self.n2])
        V = self._w2s1(x[..., self.n2:])
        return self.psi_w_from_velocity_s(U, V)

    def state_from_psi_w(self, p):
        """psi_w (..., n^2) -> flattened wavelet velocity state (..., 2 n^2)."""
        U, V = self.curl_s(self._w2s1(p))
        return jnp.concatenate([self._s2w1(U), self._s2w1(V)], axis=-1)


def validate_stream_map(ops1d, projector, n_trials: int = 3, seed: int = 0,
                        verbose: bool = True) -> dict:
    """(i) C curl_h = 0; (ii) psi o curl = id off the null space;
    (iii) curl o psi = id on zero-mean V_F (tested on projected random fields)."""
    from ODE import make_scaling_pieces
    sm = StreamMap(ops1d, projector.filter_nyquist, dtype=None)   # ambient precision
    _, _, _, project, _ = make_scaling_pieces(ops1d, projector, dtype=jnp.float64)
    rng = np.random.default_rng(seed)
    n = sm.n
    e1 = e2 = e3 = 0.0
    for _ in range(n_trials):
        Psi = jnp.asarray(rng.standard_normal((n, n)))
        U, V = sm.curl_s(Psi)
        Pu, Pv = project(U, V)
        e1 = max(e1, float(jnp.max(jnp.abs(Pu - U)) / jnp.max(jnp.abs(U))))
        U2, V2 = sm.curl_s(sm.psi_s(U, V))
        e2 = max(e2, float(jnp.max(jnp.abs(U2 - U)) / jnp.max(jnp.abs(U))))
        X, Y = project(jnp.asarray(rng.standard_normal((n, n))),
                       jnp.asarray(rng.standard_normal((n, n))))
        X, Y = X - jnp.mean(X), Y - jnp.mean(Y)
        X3, Y3 = sm.curl_s(sm.psi_s(X, Y))
        e3 = max(e3, float(jnp.max(jnp.abs(X3 - X)) / jnp.max(jnp.abs(X))))
    if verbose:
        print(f"  [validate_stream_map] ||P curl - curl|| = {e1:.1e}; "
              f"||curl psi curl - curl|| = {e2:.1e}; "
              f"||curl psi - I|| on zero-mean V_F = {e3:.1e}")
    worst = max(e1, e2, e3)
    tol = 1e-10 if jax.config.jax_enable_x64 else 1e-5
    if worst > tol:
        raise RuntimeError(f"stream-function map is not  ({worst:.2e})")
    return {"P_curl": e1, "curl_psi_curl": e2, "onto_V": e3}



# Kernel of curl_h and its  Gaussian dequantisation

def curl_kernel_basis_w(stream: StreamMap) -> np.ndarray:
    r"""(n^2, k) basis B_ker of ker(Pi_N curl_h) in psi WAVELET coordinates.

    In scaling coordinates the kernel is spanned by the constant and the
    two Nyquist lines (theta_1 = pi or theta_2 = pi); B_s is an orthonormal
    basis of it (k = 2n with the Nyquist filter, 4 without), and
    B_ker = (W_fwd (x) W_fwd) B_s.  psi_s(.) returns fields Euclidean-orthogonal
    to span(B_s), so for every data residual r = S(y - m):

        r  in  Ran := (W_fwd (x) W_fwd) span(B_s)^perp,   R^{n^2} = Ran (+) span(B_ker).

    The decomposition is oblique in wavelet coordinates (W_fwd is not
    orthogonal) but that is irrelevant: only curl_h o W_inv, which kills
    span(B_ker) , ever acts on r.  Dense reference (validation and
    GROUP's closed-form algebra); the dequantiser uses a structured basis of
    the SAME span.
    """
    n = stream.n
    e0 = np.ones(n) / np.sqrt(n)
    eN = ((-1.0) ** np.arange(n)) / np.sqrt(n)
    if stream.filter_nyquist:
        I = np.eye(n)
        cols = [np.full((n, n), 1.0 / n).reshape(-1)]
        cols += [np.outer(eN, I[j]).reshape(-1) for j in range(n)]
        cols += [np.outer(I[i], eN).reshape(-1) for i in range(n)]
    else:
        # d_h has d_hat(0) = d_hat(pi) = 0, so curl_h annihilates the FOUR
        # modes (theta_1, theta_2) in {0, pi}^2, not the constant alone.
        cols = [np.outer(p, q).reshape(-1) for p in (e0, eN) for q in (e0, eN)]
    U, s, _ = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
    k = int(np.sum(s > 1e-10 * s[0]))
    Bs = U[:, :k]                                                # orthonormal
    Wf = np.asarray(stream._Wf, float)
    return np.einsum('ab,bdk,ed->aek', Wf, Bs.reshape(n, n, k), Wf).reshape(n * n, k)


def _kernel_ops(stream: StreamMap, dtype):
    r"""(k, gram, adjoint, apply) of a STRUCTURED basis of ker(Pi_N curl_h).

    Wavelet images of the scaling-basis kernel vectors, with
    u = W_fwd e_0, v = W_fwd e_N (constant / Nyquist mode), w_j = W_fwd e_j:

        filter_nyquist:  {v w_j^T}_{j=0..n-1}  U  {w_i v^T}_{i=1..n-1}  U  {u u^T}
                         (e_N (x) e_N lies in both line families; dropping
                         i = 0 from the second removes the one dependency,
                         k = 2n);
        otherwise:       {u u^T, u v^T, v u^T, v v^T}                  (k = 4).

    The dequantised law depends on span(B_ker) only (c -> R c reparametrises
    c | r without changing B c), so this basis is equivalent to
    curl_kernel_basis_w; STREAM.validate_kernel_dequantizer checks the span
    and the Gram matrix against it.  Per sample, with Dm = D^{-1} as an
    (n, n) array,

        gram(Dm)   H = B^T D^{-1} B        O(n^3)   (vs O(n^2 k^2) dense),
        adjoint(X) B^T X                    O(n^2),
        apply(c)   B c                      O(n^2).
    """
    n = stream.n
    Wf = np.asarray(stream._Wf, float)
    e0 = np.ones(n) / np.sqrt(n)
    eN = ((-1.0) ** np.arange(n)) / np.sqrt(n)
    u, v = Wf @ e0, Wf @ eN
    W = jnp.asarray(Wf, dtype)
    W1 = W[:, 1:]
    u_, v_ = jnp.asarray(u, dtype), jnp.asarray(v, dtype)

    if not stream.filter_nyquist:
        B = np.stack([np.outer(p, q).reshape(-1) for p in (u, v) for q in (u, v)], 1)
        Bj = jnp.asarray(B, dtype)

        def gram(Dm):
            d = Dm.reshape(Dm.shape[0], -1)
            return jnp.einsum('ik,bi,il->bkl', Bj, d, Bj)

        def adjoint(X):
            return X.reshape(X.shape[0], -1) @ Bj

        def apply(c):
            return (c @ Bj.T).reshape(c.shape[0], n, n)

        return 4, gram, adjoint, apply

    v2, u2, uv = v_ * v_, u_ * u_, u_ * v_

    def gram(Dm):                                   # Dm (Bt, n, n)
        t = jnp.einsum('a,bac->bc', v2, Dm)                         # sum_a v_a^2 Dm_ab
        s = jnp.einsum('bac,c->ba', Dm, v2)                         # sum_b Dm_ab v_b^2
        Hxx = jnp.einsum('aj,ba,al->bjl', W, t, W)
        Hyy = jnp.einsum('ai,ba,al->bil', W1, s, W1)
        E = Dm * (v_[:, None] * v_[None, :])[None]
        Hyx = jnp.einsum('ai,bac,cj->bij', W1, E, W)
        p = jnp.einsum('a,bac->bc', uv, Dm) * u_[None]              # (u v)^T Dm, times u
        q = jnp.einsum('bac,c->ba', Dm, uv) * u_[None]
        H0x = p @ W
        H0y = q @ W1
        H00 = jnp.einsum('a,bac,c->b', u2, Dm, u2)
        top = jnp.concatenate([Hxx, jnp.swapaxes(Hyx, -1, -2), H0x[:, :, None]], -1)
        mid = jnp.concatenate([Hyx, Hyy, H0y[:, :, None]], -1)
        bot = jnp.concatenate([H0x[:, None, :], H0y[:, None, :], H00[:, None, None]], -1)
        return jnp.concatenate([top, mid, bot], -2)

    def adjoint(X):                                 # X (Bt, n, n)
        bx = jnp.einsum('a,bac->bc', v_, X) @ W
        by = jnp.einsum('bac,c->ba', X, v_) @ W1
        b0 = jnp.einsum('a,bac,c->b', u_, X, u_)
        return jnp.concatenate([bx, by, b0[:, None]], -1)

    def apply(c):                                   # c (Bt, 2n)
        cx, cy, c0 = c[:, :n], c[:, n:2 * n - 1], c[:, -1]
        return (v_[None, :, None] * (cx @ W.T)[:, None, :]
                + (cy @ W1.T)[:, :, None] * v_[None, None, :]
                + c0[:, None, None] * (u_[:, None] * u_[None, :])[None])

    return 2 * n, gram, adjoint, apply


def make_kernel_dequantizer(stream: StreamMap, sigma_bar_z: np.ndarray,
                            orbit: np.ndarray, dtype=jnp.float32):
    r"""deq(r, g, key) -> z = r + B_ker c,  c | r ~  Gaussian conditional.

    Why.  The data r live on the codimension-k subspace Ran (see
    ``curl_kernel_basis_w``), whereas the surrogate q = N(0, Sigma_bar) has
    full support.  Trained on r directly, the optimal network output is NOT
    zero even for Gaussian data: E[F_tgt | R] has a nonzero, known, linear
    component in span(B_ker).  That breaks the proposition "F* = 0 under the
    surrogate" on which the residual construction rests.

    Construction.  z <-> (r, c) is a linear bijection with constant Jacobian,
    so if z ~ N(0, Sigma_bar) then, with the kernel precision
    H := B^T Sigma_bar^{-1} B (k x k),

        c | r ~ N( -H^{-1} B^T Sigma_bar^{-1} r,  H^{-1} ).

    Sampling c from this conditional and setting z = r + B c gives:
      (i)  curl_h(z) = curl_h(r)            (y is unchanged, pathwise);
      (ii) if r is the Ran-component of an N(0, Sigma_bar) vector,
           z ~ N(0, Sigma_bar) , hence F* = 0 under the surrogate;
      (iii) the data law of z has a density on R^{n^2}.

    Sigma_bar(x) = sigma_bar_z * g(x).  The law of z depends on span(B) only,
    so the basis is an implementation choice, made by the width of g:

      g (Bt, n_orb)  rungs L1, L1.5: Sigma_bar^{-1} is orbit-wise affine in
                     1/g, so H = sum_o H_o / g_o with H_o precomputed on the
                     dense orthonormal basis (curl_kernel_basis_w);
      g (Bt, n^2)    rung L2: H assembled per sample on the STRUCTURED basis
                     of ``_kernel_ops`` at O(n^3) instead of O(n^2 k^2).

    The two paths draw from the same law (validate_kernel_dequantizer checks
    the span and the Gram matrix), not the same realisation; the orbit path
    is kept so that existing checkpoints stay reproducible.
    """
    orbit = np.asarray(orbit)
    n, n2 = stream.n, stream.n2
    n_orb = int(orbit.max()) + 1
    if n_orb == n2:
        raise ValueError("orbit width equals n^2: rung width is ambiguous (jmin = 0)")
    w = 1.0 / np.asarray(sigma_bar_z, float)
    wj = jnp.asarray(w, dtype)
    oj = jnp.asarray(orbit)

    # orbit-wide g: dense orthonormal basis, precomputed orbit Gram blocks
    B = curl_kernel_basis_w(stream)
    k_dense = B.shape[1]
    Bj = jnp.asarray(B, dtype)
    H_o = np.empty((n_orb, k_dense, k_dense))
    for o in range(n_orb):
        Bo = B[orbit == o]
        H_o[o] = Bo.T @ (Bo * w[orbit == o, None])
    Hj = jnp.asarray(H_o, dtype)

    # per-coefficient g: structured basis
    k_str, gram, adjoint, apply = _kernel_ops(stream, dtype)

    def _draw(H, b, key):
        L = jnp.linalg.cholesky(H)
        mean = -jax.scipy.linalg.cho_solve((L, True), b[..., None])[..., 0]
        xi = jax.random.normal(key, mean.shape, dtype)
        return mean + jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(L, -1, -2), xi[..., None], lower=False)[..., 0]

    def deq(r, g, key):
        """r (Bt, n^2); g (Bt, n_orb) or (Bt, n^2):  Sigma_bar = sigma_bar_z * g."""
        g = jnp.asarray(g, dtype)
        if g.shape[-1] == n2:
            dinv = (wj[None, :] / g).reshape(-1, n, n)
            c = _draw(gram(dinv), adjoint(r.reshape(-1, n, n) * dinv), key)
            return r + apply(c).reshape(r.shape)
        inv_g = 1.0 / g
        H = jnp.einsum('bo,oij->bij', inv_g, Hj)
        dinv = wj[None, :] * jnp.take(inv_g, oj, axis=1)
        c = _draw(H, (r * dinv) @ Bj, key)
        return r + c @ Bj.T

    deq.k = k_str
    deq.gram, deq.adjoint, deq.apply = gram, adjoint, apply
    return deq


def validate_kernel_dequantizer(stream: StreamMap, sigma_bar_z, orbit,
                                n_samples: int = 4096, seed: int = 0,
                                verbose: bool = True) -> dict:
    """(i) the structured basis spans ker(Pi_N curl_h) and its Gram matrix is
    B^T Sigma_bar^{-1} B;  (ii) curl_h kills it;  (iii) Ran-projection of
    N(0, Sigma_bar) + dequantisation = N(0, Sigma_bar), for an orbit-wise AND
    a per-coefficient multiplier g."""
    from FORCING import orbit_pool
    wdt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    deq = make_kernel_dequantizer(stream, sigma_bar_z, orbit, dtype=wdt)
    n, n2, k = stream.n, stream.n2, deq.k
    rng = np.random.default_rng(seed)
    orbit = np.asarray(orbit)
    n_orb = int(orbit.max()) + 1
    sbz = np.asarray(sigma_bar_z, float)

    # (i) span and Gram against the dense orthonormal reference
    B_ref = curl_kernel_basis_w(stream)
    B_str = np.asarray(deq.apply(jnp.eye(k, dtype=wdt)), float).reshape(k, n2).T
    coef, *_ = np.linalg.lstsq(B_ref, B_str, rcond=None)
    span_err = float(np.linalg.norm(B_ref @ coef - B_str) / np.linalg.norm(B_str))
    rank = int(np.linalg.matrix_rank(B_str))
    g_test = rng.uniform(0.5, 2.0, n2)
    dinv = 1.0 / (sbz * g_test)
    H_str = np.asarray(deq.gram(jnp.asarray(dinv.reshape(1, n, n), wdt))[0], float)
    H_ref = B_str.T @ (B_str * dinv[:, None])
    gram_err = float(np.max(np.abs(H_str - H_ref)) / np.max(np.abs(H_ref)))

    # (ii) curl_h annihilates the kernel
    Wi = np.asarray(stream._Wi, float)
    Zs = np.einsum('ab,kbd,ed->kae', Wi, B_str.T.reshape(k, n, n), Wi)
    U, V = stream.curl_s(jnp.asarray(Zs, stream.dtype))
    e_curl = float(jnp.max(jnp.abs(U)) + jnp.max(jnp.abs(V))) / max(float(np.abs(Zs).max()), 1e-300)

    # (iii) ness of the conditional law, orbit-wise and per-coefficient g
    Bs = np.einsum('ab,bdk,ed->aek', Wi, B_ref.reshape(n, n, -1), Wi).reshape(n2, -1)
    tol = 4.0 * np.sqrt(2.0 / (n_samples * np.bincount(orbit).min()))
    devs = {}
    for name, g in (("orbit", rng.uniform(0.5, 2.0, n_orb)),
                    ("per-coefficient", rng.uniform(0.5, 2.0, n2))):
        gc = g if g.size == n2 else g[orbit]
        D = sbz * gc
        z = rng.standard_normal((n_samples, n2)) * np.sqrt(D)
        zs = np.einsum('ab,kbd,ed->kae', Wi, z.reshape(-1, n, n), Wi).reshape(n_samples, -1)
        r = z - (zs @ Bs) @ B_ref.T                 #  oblique projection onto Ran
        gj = jnp.asarray(g[None, :], wdt)
        keys = jax.random.split(jax.random.PRNGKey(seed), -(-n_samples // 256))
        z2 = np.concatenate([np.asarray(deq(jnp.asarray(r[i:i + 256], wdt),
                                            jnp.broadcast_to(gj, (len(r[i:i + 256]), g.size)),
                                            keys[i // 256]), float)
                             for i in range(0, n_samples, 256)])
        ratio = orbit_pool((z2.var(0) / D)[None, :], orbit).ravel()
        devs[name] = float(np.max(np.abs(ratio - 1.0)))
    alg_tol = 1e-8 if wdt == jnp.float64 else 1e-3
    ok = (span_err < alg_tol and rank == k and gram_err < alg_tol
          and e_curl < (1e-10 if stream.dtype == jnp.float64 else 1e-4)
          and all(d < tol for d in devs.values()))
    if verbose:
        print(f"  [validate_kernel_dequantizer] dim ker = {k} (rank {rank}); span err "
              f"{span_err:.1e}, Gram err {gram_err:.1e}, max|curl B| {e_curl:.1e}; "
              f"orbit Var(z)/Sigma_bar - 1: max {devs['orbit']:.3f} (orbit g), "
              f"{devs['per-coefficient']:.3f} (per-coefficient g) (tol {tol:.3f})")
    if not ok:
        raise RuntimeError(f"kernel dequantiser failed: span {span_err:.1e}, rank {rank}/{k}, "
                           f"Gram {gram_err:.1e}, curl {e_curl:.1e}, law {devs} (tol {tol:.3e})")
    return {"span_err": span_err, "gram_err": gram_err, "curl": e_curl,
            "max_dev": devs, "tol": tol, "ok": ok}



#  Vorticity: one operator, two bases -- the certificate

def _nyquist_free(X, filter_nyquist: bool):
    """Zero the theta = pi rows/columns of X, so Pi_N acts as the identity on it."""
    if not filter_nyquist:
        return np.asarray(X, float)
    n = X.shape[-1]
    H = np.fft.fft2(np.asarray(X, float))
    H[n // 2, :] = 0.0
    H[:, n // 2] = 0.0
    return np.real(np.fft.ifft2(H))


def validate_vorticity_consistency(ops1d, projector, seed: int = 0,
                                   tol: float = 1e-5, verbose: bool = True) -> dict:
    r"""StreamMap.vorticity_s == Kron2DOperators.weak_vorticity_mat, conjugated.

    There is  ONE discrete vorticity in this code base,
    omega = d_h v - u d_h^T with d_h = M^{-1} D, and two ways to evaluate it:
    the FFT one on scaling-basis arrays (vorticity_s, the production path)
    and the dense one on wavelet coefficients (weak_vorticity_mat, the
    reference).  After the change of basis they agree to float32 roundoff:

        vorticity_s(U, V) = W_inv (weak_vorticity_mat(W U W^T, W V W^T)) W_inv^T.

    Nyquist caveat.  With ``filter_nyquist=True`` StreamMap carries Pi_N
    inside its symbols, so vorticity_s computes Pi_N (d_h v - u d_h^T) Pi_N
    while weak_vorticity_mat does not.  On a field with Nyquist content the
    two differ by O(1) and neither is wrong; the test field is therefore made
    Nyquist-free first (as the production data are).  A failure here is a
    sign/transpose/axis drift between ``_symbol`` and ops1d.

    Also checks vorticity_s(curl_s(psi)) = -(d_h^2 (+) d_h^2) psi.  The code
    base has TWO discrete Laplacians:

      d_h^2-Laplacian  (d_h (x) I)^2 + (I (x) d_h)^2,  d_h = M^{-1} D: the
                       discrete first derivative applied twice; what the
                       stream-function construction inverts, and the operator
                       in the identity above.
      Delta_h          M^{-1} K,  K_ij = <phi_i, phi_j''> = -<phi_i', phi_j'>
                       (Bubnov; negative semidefinite); what the VISCOUS term
                       uses.

    Both are negative semidefinite; they coincide iff K = D M^{-1} D.  In
    symbols, k_hat/m_hat vs (d_hat/m_hat)^2: equal to high order as
    theta -> 0, but d_hat(pi) = 0 while k_hat(pi) != 0, so they differ by
    O(1) near Nyquist.  ||K - D M^{-1} D|| / ||K|| is returned as
    'stiffness_vs_d2'.  Whenever a formula mixes curl_h and viscosity, say
    which Laplacian is meant.
    """
    ops = build_2d_operators(ops1d)
    n = 2 ** ops1d.J
    fn = bool(projector.filter_nyquist)
    Wf = np.asarray(ops1d.W_fwd, float)
    Wi = np.asarray(ops1d.W_inv, float)
    st = StreamMap(ops1d, fn, dtype=jnp.float32)
    rng = np.random.default_rng(seed)
    U = _nyquist_free(rng.standard_normal((n, n)), fn)
    V = _nyquist_free(rng.standard_normal((n, n)), fn)

    w_fft = np.asarray(st.vorticity_s(jnp.asarray(U, jnp.float32),
                                      jnp.asarray(V, jnp.float32)), float)
    w_den = np.asarray(ops.weak_vorticity_mat((Wf @ U @ Wf.T).ravel(),
                                              (Wf @ V @ Wf.T).ravel())).reshape(n, n)
    w_den = Wi @ w_den @ Wi.T
    err_basis = float(np.max(np.abs(w_fft - w_den))
                      / max(np.max(np.abs(w_fft)), 1e-300))

    Psi = _nyquist_free(rng.standard_normal((n, n)), fn)
    Uc, Vc = st.curl_s(jnp.asarray(Psi, jnp.float32))
    lhs = np.asarray(st.vorticity_s(Uc, Vc), float)
    Pw = Wf @ Psi @ Wf.T
    Mw, Dw, Kw = (np.asarray(a, float) for a in (ops.M, ops.D, ops.K))
    dh = np.linalg.solve(Mw, Dw)
    dh2 = dh @ dh
    rhs = -(Wi @ (dh2 @ Pw + Pw @ dh2.T) @ Wi.T)
    err_lap = float(np.max(np.abs(lhs - rhs)) / max(np.max(np.abs(rhs)), 1e-300))

    out = {"err_basis": err_basis, "err_laplacian": err_lap, "tol": float(tol),
           "filter_nyquist": fn,
           "stiffness_vs_d2": float(np.max(np.abs(Kw - Dw @ np.linalg.solve(Mw, Dw)))
                                    / max(np.max(np.abs(Kw)), 1e-300)),
           "ok": bool(err_basis < tol and err_lap < tol)}
    if verbose:
        print(f"   [STREAM] vorticity_s vs weak_vorticity_mat: {err_basis:.2e}; "
              f"curl vs d_h^2-Laplacian: {err_lap:.2e} (tol {tol:.0e}) -> "
              f"{'OK' if out['ok'] else 'FAIL'};  ||K - D M^-1 D||/||K|| = "
              f"{out['stiffness_vs_d2']:.2f} (viscous vs stream Laplacian)")
    return out
