r"""DIFF.py -- network, residual denoising score matching, diagnostics.

What is learned
---------------
Only F_theta, the correction of the conditional score BEYOND THE DIAGONAL
GAUSSIAN SURROGATE q = N(0, diag Sigma_bar(x)).  It
contains the off-diagonal Gaussian covariance as well as x-dependent and
non-Gaussian structure; GROUP.empirical_linear_floor measures what an
x-independent linear map already gets:

    s(R, tau; x) = -R/Sg  -  alpha F_theta(Rc, psi_m, psi_x, logSNR) / sqrt(s),

trained by regression on the unit-variance, Gaussian-orthogonal target

    F_tgt = (a sqrt(Sigma_bar) eps - sqrt(s) r / sqrt(Sigma_bar)) / sqrt(Sg),
    R = a r + sqrt(s) eps,  r = S(y - m(x)),  tau ~ logU[t_min, T_diff],

S the stream-function map (STREAM.py).  r lies on a codimension-2n
subspace; it is replaced, inside the loss, by its  Gaussian
dequantisation z = r + B_w c (STREAM.make_kernel_dequantizer), fresh c at
every step.  Without it the proposition below fails in the 2n kernel
directions (F* != 0 even for Gaussian data).

Proposition (what the loss minimises).  For every tau, the L2 minimiser is
F* = E[F_tgt | R, x], and by the pathwise identity
E[r | R, x] = a Sigma_bar R/Sg - sqrt(s Sigma_bar/Sg) F*, i.e. denoising
score matching (Tweedie).  Under the Gaussian surrogate F* = 0, so the
network output measures the departure of p(z | x) from the
DIAGONAL Gaussian N(0, diag Sigma_bar(x)) -- correlations included.  The
final convolution is zero-initialised: training STARTS at the surrogate.

F = 0 loss.  With alpha = a sqrt(Sigma_bar/Sg), beta = sqrt(s/Sg),
alpha^2 + beta^2 = 1 identically, and eps independent of (z, x):

    E[F_tgt,i^2] = 1 + E[ beta_i^2(tau, x) (E[z_i^2 | x] / Sigma_bar_i(x) - 1) ]

(tau and eps independent of (z, x); beta depends on x through Sigma_bar(x)
and on sigma^2 through s).  So the F = 0 loss measures a beta^2-weighted
CALIBRATION OF THE SURROGATE: it equals 1 if E[z_i^2 | x] = Sigma_bar_i(x)
for all x (conditional calibration), and < 1 means Sigma_bar over-estimates
the conditional variance on average in that weighting.  Only for an
x-independent Sigma_bar (rung L1) does it factorise,
1 + E_tau[beta_i^2] (E[z_i^2] / Sigma_bar_i - 1), with "= 1 iff calibrated".

Weighting.  With eps_hat - eps = alpha (F - F_tgt) and alpha^2 = SNR/(1+SNR)
(SNR = a^2 Sigma_bar / s, the CONDITIONAL SNR):

    ||F - F_tgt||^2 = (1 + 1/SNR) ||eps_hat - eps||^2
                    = (1 + SNR)   ||rhat - r||^2 / Sigma_bar.

'unit' (default, required by the linear floor) is therefore the
v-prediction loss: eps-MSE weighted by (1 + 1/SNR), equivalently x-MSE
weighted by (1 + SNR).  'min_snr' multiplies the F-loss by
min(SNR, snr_cap)/(1 + SNR), Min-SNR on the eps-loss,
min(SNR, snr_cap)/SNR.

Inputs
------
* Rc = R / sqrt(Sg) (unit variance under the surrogate at every tau);
* S m / sqrt(E[(S m)^2]) and S x / sqrt(E[(S x)^2]) (orbit-pooled): the
  conditioners.  x itself is given, not only m = Phi_Delta(x): Sigma(x)
  depends on the whole path x -> m, and backward NS being ill-posed, the
  fine scales of x are not recoverable from m;
* per-level mean conditional log-SNR  log(a^2 Sigma_bar / s): the noise
  level of what x does not determine, which already contains g(x).
All images are synthesised to the scaling (physical) grid with W_inv, where
a periodic U-Net is translation equivariant; the output is analysed back with
W_fwd.  A per-level output gain (FiLM on the conditioning) lets each level's
unit-variance target be matched independently.

Augmentation: dyadic shifts by k 2^(J-jmin) cells, which are 
permutations of wavelet coordinates and symmetries of the forced law.  They
leave the orbit maps invariant, hence an orbit-wide g; a per-coefficient g
(rung L2) is a field and is shifted with r.
"""

from __future__ import annotations

import os
import pickle
import resource
from typing import Any, Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax

from ODE import HIGHEST, dyadic_shift_permutation_1d
from WSBG import WaveletDiffusionModel
from TLM import (ou_coefficients, baseline_coefficients, whiten,
                       f_target, conditional_log_snr)



# 1.  Network

class CondEmbedding(nn.Module):
    """Per-level conditional log-SNR vector (B, L) -> embedding (B, D)."""
    dim: int = 128

    @nn.compact
    def __call__(self, lsnr):
        s = jnp.clip(lsnr.astype(jnp.float32), -25.0, 25.0) / 8.0
        h = jnp.concatenate([s, jnp.tanh(s), jnp.sin(s), jnp.cos(s)], axis=-1)
        h = nn.gelu(nn.Dense(self.dim)(h))
        return nn.Dense(self.dim)(h)


class _FiLMResBlock(nn.Module):
    """Pre-norm residual block, periodic 3x3 convolutions, FiLM conditioning."""
    ch: int

    @nn.compact
    def __call__(self, h, cond):
        g = min(8, self.ch)
        y = nn.GroupNorm(num_groups=g)(h)
        if y.shape[-1] != self.ch:
            y = nn.Conv(self.ch, (1, 1))(y)
        ss = nn.Dense(2 * self.ch, kernel_init=nn.initializers.zeros)(cond)
        scale, shift = jnp.split(ss[:, None, None, :], 2, axis=-1)
        y = nn.gelu(y * (1.0 + scale) + shift)
        y = nn.Conv(self.ch, (3, 3), padding="CIRCULAR")(y)
        y = nn.gelu(nn.GroupNorm(num_groups=g)(y))
        y = nn.Conv(self.ch, (3, 3), padding="CIRCULAR",
                    kernel_init=nn.initializers.zeros)(y)
        skip = h if h.shape[-1] == self.ch else nn.Conv(self.ch, (1, 1))(h)
        return skip + y


class PeriodicUNet(nn.Module):
    """U-Net with circular padding.  Equivariant to shifts by multiples of
    2^(len(channels)-1); len(channels)-1 <= J - jmin is ENFORCED in
    make_apply_F so that the augmentation group is a symmetry of the
    architecture as well."""
    channels: Tuple[int, ...] = (32, 64, 96, 128)
    out_ch: int = 1
    remat: bool = False

    @nn.compact
    def __call__(self, x, cond):
        Block = nn.remat(_FiLMResBlock) if self.remat else _FiLMResBlock
        h = nn.Conv(self.channels[0], (3, 3), padding="CIRCULAR")(x)
        skips = []
        for i, ch in enumerate(self.channels):
            h = Block(ch)(h, cond)
            skips.append(h)
            if i < len(self.channels) - 1:
                h = nn.Conv(ch, (3, 3), strides=(2, 2), padding="CIRCULAR")(h)
        h = Block(self.channels[-1])(h, cond)
        for i in reversed(range(len(self.channels))):
            h = Block(self.channels[i])(jnp.concatenate([h, skips[i]], axis=-1), cond)
            if i > 0:
                h = jnp.repeat(jnp.repeat(h, 2, axis=1), 2, axis=2)
        h = nn.gelu(nn.GroupNorm(num_groups=min(8, self.channels[0]))(h))
        return nn.Conv(self.out_ch, (3, 3), padding="CIRCULAR",
                       kernel_init=nn.initializers.zeros)(h)


class ResidualScoreNet(nn.Module):
    """F_theta on scaling-domain images; per-level output gain."""
    channels: Tuple[int, ...] = (32, 64, 96, 128)
    n_levels: int = 4
    remat: bool = False

    @nn.compact
    def __call__(self, img, lsnr):
        e = CondEmbedding(128)(lsnr)
        cond = nn.gelu(nn.Dense(256)(e))
        out = PeriodicUNet(self.channels, 1, self.remat)(img, cond)
        gain = 1.0 + nn.Dense(self.n_levels, kernel_init=nn.initializers.zeros)(cond)
        return out, gain


def make_apply_F(model: ResidualScoreNet, diffusion: WaveletDiffusionModel):
    """apply_F(params, Rc, cond_m, cond_x, lsnr) -> F  (psi wavelet coordinates)."""
    n = diffusion.n
    depth = len(model.channels) - 1
    if depth > diffusion.J - diffusion.jmin:
        raise ValueError(f"U-Net depth {depth} > J - jmin = {diffusion.J - diffusion.jmin}: "
                         "the network is not equivariant to the dyadic shifts "
                         "used for augmentation.")
    Wf = jnp.asarray(diffusion.W_fwd, jnp.float32)
    Wi = jnp.asarray(diffusion.W_inv, jnp.float32)
    lvl = jnp.asarray(diffusion.level_index_per_coeff)

    def to_img(Z):                      # (B, n^2) -> (B, n, n)
        X = Z.reshape(-1, n, n)
        return jnp.einsum('ab,kbd,ed->kae', Wi, X, Wi, precision=HIGHEST)

    def from_img(Y):                    # (B, n, n) -> (B, n^2)
        Y = jnp.einsum('ab,kbd,ed->kae', Wf, Y, Wf, precision=HIGHEST)
        return Y.reshape(Y.shape[0], -1)

    def apply_F(params, Rc, cond_m, cond_x, lsnr):
        B = Rc.shape[0]
        img = jnp.stack([to_img(jnp.asarray(Rc, jnp.float32)),
                         jnp.broadcast_to(to_img(jnp.asarray(cond_m, jnp.float32)), (B, n, n)),
                         jnp.broadcast_to(to_img(jnp.asarray(cond_x, jnp.float32)), (B, n, n))],
                        axis=-1)
        out, gain = model.apply(params, img, jnp.asarray(lsnr, jnp.float32))
        return from_img(out[..., 0]) * jnp.take(gain, lvl, axis=1)

    def init(key):
        return model.init(key, jnp.zeros((1, n, n, 3), jnp.float32),
                          jnp.zeros((1, len(diffusion.levels)), jnp.float32))

    return apply_F, init



# 2.  Data access and augmentation

def split_by_trajectory(traj_id: np.ndarray, val_split=0.2, seed=42):
    """Pair indices split by TRAJECTORY (consecutive pairs are correlated)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(traj_id)
    rng.shuffle(uniq)
    n_val = max(1, int(round(val_split * len(uniq))))
    val = np.isin(traj_id, uniq[:n_val])
    idx_val, idx_tr = np.flatnonzero(val), np.flatnonzero(~val)
    rng.shuffle(idx_tr)
    return idx_tr, idx_val


def make_batch_loader(dataset, diffusion: WaveletDiffusionModel, g=None):
    """idx -> (r, S m, S x, g), all in psi wavelet coordinates.

    ``g`` is the surrogate multiplier: Sigma_bar = sigma_bar_z * g.  Width
    n_orbits (rungs L1, L1.5) or n^2 (rung L2); default dataset['rho'].
    """
    X, M = dataset["x"], dataset["m"]
    pi = np.asarray(dataset["pair_index"])
    g_all = np.asarray(dataset["rho"] if g is None else g, np.float32)
    to_psi = diffusion.to_psi

    def load(idx):
        idx = np.asarray(idx)
        order = np.argsort(idx)
        s = idx[order]
        d = X.shape[1]
        y = np.empty((len(idx), d), np.float32)
        x = np.empty_like(y)
        m = np.empty_like(y)
        y[order] = X[pi[s, 1]]
        x[order] = X[pi[s, 0]]
        m[order] = M[s]
        return to_psi(y - m), to_psi(m), to_psi(x), jnp.asarray(g_all[idx])
    return load


def make_shift_augmentation(diffusion: WaveletDiffusionModel):
    """(B, n^2) -> (B, n^2) under a dyadic shift (kx, ky);  permutation.

    S commutes with every translation (it is a Fourier multiplier), so the
    shift acts on psi as on each velocity component.
    """
    J, jmin, n = diffusion.J, diffusion.jmin, diffusion.n
    Kn = 2 ** jmin
    inv = []
    for k in range(Kn):
        p = dyadic_shift_permutation_1d(diffusion.W_fwd, diffusion.W_inv, J, jmin, k)
        q = np.empty_like(p)
        q[p] = np.arange(n)                   # x'[p[i]] = x[i]  ->  x' = x[q]
        inv.append(q)
    inv = jnp.asarray(np.stack(inv))

    def shift(z, kx, ky):
        Z = z.reshape(z.shape[0], n, n)
        Z = jnp.take(jnp.take(Z, inv[kx], axis=1), inv[ky], axis=2)
        return Z.reshape(z.shape)
    return shift, Kn



# 3.  Training

class TrainState(train_state.TrainState):
    loss: float = 0.0


def _save(path, payload):
    tmp = path + ".tmp"
    if os.path.dirname(tmp):
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def cluster_se(d: np.ndarray, cluster: np.ndarray) -> float:
    """Cluster-robust SE of mean(d) (clusters = independent units, e.g.
    trajectories):  sqrt(G/(G-1)) sqrt(sum_c (S_c - n_c dbar)^2) / N.
    NaN with fewer than 2 clusters."""
    d = np.asarray(d, float)
    _, inv = np.unique(np.asarray(cluster), return_inverse=True)
    G = int(inv.max()) + 1
    if G < 2:
        return float("nan")
    S = np.bincount(inv, weights=d)
    nc = np.bincount(inv).astype(float)
    return float(np.sqrt(G / (G - 1.0) * np.sum((S - nc * d.mean()) ** 2)) / d.size)


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3

def make_sbar_fn(diffusion: WaveletDiffusionModel, sigma_bar_z):
    """g -> Sigma_bar, accepting an orbit-wide (n_orb) or per-coefficient
    (n^2) multiplier.  The width is what distinguishes the rungs; nothing
    else in the loss changes between them."""
    n_psi = int(diffusion.n_psi)
    orbit = jnp.asarray(diffusion.orbit_per_coeff)
    sbar_z = jnp.asarray(sigma_bar_z, jnp.float32)

    def sbar_of(g):
        g = jnp.asarray(g, jnp.float32)
        gg = g if g.shape[-1] == n_psi else jnp.take(g, orbit, axis=1)
        return sbar_z[None, :] * gg
    return sbar_of


def train_conditional_score_model(
    dataset: Dict[str, Any], diffusion: WaveletDiffusionModel,
    sigma_bar_z: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray, *,
    n_epochs=100, batch_size=16, lr=2e-4, seed=42,
    channels=(32, 64, 96, 128), weighting="unit", snr_cap=5.0,
    ema_decay=0.999, patience=20, augment=True, checkpoint_path=None,
    verbose=True, dequant=None, g=None,
):
    """Residual denoising score matching (module docstring).  ``g`` overrides
    dataset['rho'] as the surrogate multiplier (GRID rungs)."""
    if weighting not in ("unit", "min_snr"):
        raise ValueError(f"weighting must be 'unit' or 'min_snr', got {weighting!r}")
    if diffusion.t_min is None:
        raise RuntimeError("diffusion.set_time_range(...) first")
    f32 = jnp.float32
    L = len(diffusion.levels)
    model = ResidualScoreNet(channels=tuple(channels), n_levels=L)
    apply_F, init = make_apply_F(model, diffusion)
    load = make_batch_loader(dataset, diffusion, g=g)
    sbar_of = make_sbar_fn(diffusion, sigma_bar_z)
    shift, Kn = make_shift_augmentation(diffusion)

    lam = jnp.asarray(diffusion.lambda_diag, f32)
    sig = jnp.asarray(diffusion.sigma_sq_diag, f32)
    mscale = jnp.asarray(diffusion.m_scale_diag, f32)
    xscale = jnp.asarray(diffusion.x_scale_diag, f32)
    masks = jnp.asarray(diffusion.level_masks)
    counts = jnp.sum(masks, axis=1)
    log_t0, log_T = float(np.log(diffusion.t_min)), float(np.log(diffusion.T_diff))

    def loss_fn(params, r, pm, px, gb, key, zero=False, per_sample=False):
        B = r.shape[0]
        k1, k2, k3 = random.split(key, 3)
        if dequant is not None:
            r = dequant(r, gb, k3)
        tau = jnp.exp(random.uniform(k1, (B, 1), f32, log_t0, log_T))
        eps = random.normal(k2, r.shape, f32)
        sbar = sbar_of(gb)
        a, s = ou_coefficients(lam[None, :], sig[None, :], tau)
        Sg, _, _ = baseline_coefficients(a, s, sbar)
        R = a * r + jnp.sqrt(s) * eps
        lsnr_c = conditional_log_snr(a, s, sbar)
        lsnr = (lsnr_c @ masks.T) / counts
        F = (jnp.zeros_like(R) if zero else
             apply_F(params, whiten(R, Sg), pm / mscale, px / xscale, lsnr))
        tgt = f_target(r, eps, a, s, sbar)
        if weighting == "unit":
            w = 1.0
        else:
            snr = jnp.exp(lsnr_c)
            w = jnp.minimum(snr, snr_cap) / (1.0 + snr)
        per_level = ((w * (F - tgt) ** 2) @ masks.T) / counts        # (B, L)
        return per_level.mean(1) if per_sample else jnp.mean(per_level)

    n_psi = int(diffusion.n_psi)        # n^2: the per-coefficient surrogate width

    @jax.jit
    def train_step(state, ema, r, pm, px, gb, kx, ky, key):
        if augment:
            r, pm, px = shift(r, kx, ky), shift(pm, kx, ky), shift(px, kx, ky)
            # g of width n_orb is shift-invariant (orbit labels are); g of
            # width n^2 is a FIELD g_i(x) and must be translated with r.
            if gb.shape[-1] == n_psi:
                gb = shift(gb, kx, ky)
        loss, grads = jax.value_and_grad(loss_fn)(state.params, r, pm, px, gb, key)
        state = state.apply_gradients(grads=grads).replace(loss=loss)
        k = state.step.astype(f32)
        d = jnp.minimum(ema_decay, (1.0 + k) / (10.0 + k))
        ema = jax.tree_util.tree_map(lambda e, p: d * e + (1 - d) * p, ema, state.params)
        return state, ema

    val_loss_fn = jax.jit(loss_fn, static_argnames=("zero", "per_sample"))


    key = random.PRNGKey(seed)
    key, ik = random.split(key)
    params = init(ik)
    steps_per_epoch = max(1, len(idx_train) // batch_size)
    total = n_epochs * steps_per_epoch
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, total // 20),
                                               max(total, 2), lr * 1e-2)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(sched, weight_decay=1e-4))
    state = TrainState.create(apply_fn=None, params=params, tx=tx, loss=0.0)
    ema = params
    fp = diffusion.fingerprint()
    fp["sigma_bar_z"] = np.asarray(sigma_bar_z)
    fp["channels"] = np.asarray(channels)
    # a checkpoint trained at one rung must not be reloaded at another
    fp["surrogate_width"] = np.asarray([float(load(idx_train[:1])[3].shape[-1])])
    start_epoch, best_val, bad, best_ema = 0, np.inf, 0, None
    best_gain_se = float("nan")

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ck = pickle.load(f)
        same = all(k in ck["fingerprint"] and np.shape(ck["fingerprint"][k]) == np.shape(v)
                   and np.allclose(ck["fingerprint"][k], v, rtol=1e-6)
                   for k, v in fp.items())
        if not same:
            raise RuntimeError(f"{checkpoint_path} was trained with a different "
                               "calibration / Sigma_bar / architecture; delete it.")
        state = state.replace(params=ck["params"], opt_state=ck["opt_state"],
                              step=ck["step"])
        ema, best_val = ck["ema"], ck["best_val"]
        best_gain_se = ck.get("best_gain_se", float("nan"))
        start_epoch = ck["epoch"]
        best_ema = ck.get("best_ema", ema)
        if verbose:
            print(f"  [train] resumed at epoch {start_epoch} (best val {best_val:.5f})")

    n_par = sum(p.size for p in jax.tree_util.tree_leaves(params))
    if verbose:
        print(f"  [train] {n_par/1e6:.2f}M parameters; {len(idx_train)} train / "
              f"{len(idx_val)} val pairs; tau ~ logU[{diffusion.t_min:.2e}, "
              f"{diffusion.T_diff:.2e}]; weighting={weighting}")

    nvb = max(1, len(idx_val) // batch_size)
    val_keys = random.split(random.PRNGKey(seed + 1), nvb)
    history = dict(ck.get("history", {"train": [], "val": []})) if (
        checkpoint_path and os.path.exists(checkpoint_path)) else {"train": [], "val": []}
    rng = np.random.default_rng(seed + start_epoch)
    # PAIRED F = 0 baseline on the same validation batches and keys.
    # E[F_tgt^2] = 1 + E[beta^2](E[z^2/Sigma_bar] - 1) != 1 in general (module
    # docstring), so "1 - val" is NOT the network's gain; "F=0 - val" is.
    val_batches = [idx_val[bi * batch_size:(bi + 1) * batch_size] for bi in range(nvb)]
    # Per-SAMPLE paired losses; the independent units are TRAJECTORIES
    # (pairs of one trajectory share their conditioning history), so the SE
    # of the gain is cluster-robust over trajectories, never over batches.
    traj_val = np.concatenate([np.asarray(dataset["traj_id"])[vb] for vb in val_batches])
    n_traj_val = int(len(np.unique(traj_val)))
    if n_traj_val < 5 and verbose:
        print(f"  [train] WARNING: {n_traj_val} validation trajectories: the gain SE is "
              "unreliable (< 5 clusters) or undefined (< 2)")

    def _val_losses(p, zero):
        return np.concatenate([np.asarray(val_loss_fn(p, *load(vb), val_keys[bi],
                                                      zero=zero, per_sample=True))
                               for bi, vb in enumerate(val_batches)])

    val_base_s = _val_losses(params, True)
    val_base_mean = float(val_base_s.mean())
    if verbose:
        print(f"  [train] F = 0 loss on val: {val_base_mean:.5f}  "
              f"(1 if E[z^2|x] = Sigma_bar(x); < 1: Sigma_bar over-estimates, "
              f"beta^2-weighted)")
    for epoch in range(start_epoch, n_epochs):
        acc = jnp.zeros((), f32)
        for _ in range(steps_per_epoch):
            b = rng.choice(idx_train, size=batch_size, replace=False)
            r, pm, px, gb = load(b)
            key, sk = random.split(key)
            state, ema = train_step(state, ema, r, pm, px, gb,
                                    int(rng.integers(Kn)), int(rng.integers(Kn)), sk)
            acc = acc + state.loss
        tr_loss = float(acc) / steps_per_epoch
        vls_s = _val_losses(ema, False)
        vl = float(vls_s.mean())
        vl_se = cluster_se(val_base_s - vls_s, traj_val)
        history["train"].append(tr_loss)
        history["val"].append(vl)
        if verbose:
            print(f"  [epoch {epoch+1:4d}/{n_epochs}] train {tr_loss:.5f}  "
                  f"val(EMA) {vl:.5f}  gain vs F=0 {val_base_mean - vl:+.5f}  "
                  f"RSS {_peak_rss_mb():.0f} MB", flush=True)
        if vl < best_val:
            best_val, best_gain_se, bad = vl, vl_se, 0
            best_ema = jax.tree_util.tree_map(np.asarray, ema)
        else:
            bad += 1
        if checkpoint_path:
            _save(checkpoint_path, {
                "params": jax.tree_util.tree_map(np.asarray, state.params),
                "opt_state": jax.tree_util.tree_map(np.asarray, state.opt_state),
                "step": int(state.step), "epoch": epoch + 1,
                "ema": jax.tree_util.tree_map(np.asarray, ema),
                "best_ema": best_ema, "best_val": best_val, "fingerprint": fp,
                "history": history, "best_gain_se": best_gain_se})
        if bad > patience:
            if verbose:
                print(f"  [train] early stop at epoch {epoch+1}")
            break

    best = ema if best_ema is None else jax.tree_util.tree_map(jnp.asarray, best_ema)
    return {"params": best, "apply_F": apply_F, "model": model,
            "best_val": best_val, "history": history,
            "val_base": val_base_mean, "best_gain": val_base_mean - best_val,
            "best_gain_se": best_gain_se}