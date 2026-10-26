r"""GRID.py -- the headline result: (baseline rung) x (lag) grid.

Why a grid
----------
A single "training gain vs F = 0" number confounds three things:

  (a) the OFF-DIAGONAL covariance of Sigma(x), present already for LINEAR
      dynamics (: GROUP.oracle_from_diffusion);
  (b) the WITHIN-ORBIT spatial inhomogeneity of diag Sigma_TLM(x): real
      x-dependence that the TLM covariance supplies but that the
      orbit-pooled surrogate throws away;
  (c) the x-dependent conditional mean shift and the genuinely non-Gaussian
      part of p(z | x).

Only (c) is what the learned component is for.  The grid separates them:

  ROWS  = baseline rung, i.e. how much of the TLM diagonal is put in
          the surrogate Sigma_bar_i(x) = Sigma_bar_z,i g_i(x):

            L1    g == 1                    x-independent (sigma_mode 'B0')
            L1.5  g_i = rho_{o(i)}(x)       orbit-pooled TLM ratio
            L2    g_i = rho_i(x)            per-coefficient TLM ratio, shrunk
                                            to its orbit mean (empirical Bayes)

          Going down a row removes (b) from what the network can claim.
  COLS  = lag Delta.  The nonlinear content grows with Delta (third
          cumulant ~ Delta^{3/2}, x-dependent mean shift ~ Delta^{5/2}:
          CERT.delta_regime).  Every column is its own experiment (dataset,
          Sigma_lin, forward process).
  FLOOR = the EMPIRICAL linear floor, one per CELL (GROUP.empirical_linear_floor):
          the best x-independent linear predictor of F_tgt in the rung's own
          loss units, fitted on the cell's training pairs and scored on its
          validation pairs with the network's (tau, eps) draws.

Verdict per cell (CERT.gain_verdict),  z = (loss_lin - loss_net) / SE, paired:
    '+'  z >  3   x-dependent structure learned (decided by the shift panel)
    '0'  |z| <= 3 nothing beyond an x-independent linear predictor
    '-'  z < -3   worse than a closed-form linear map: under-trained
    '?'  no SE (fewer than 5 validation trajectories)

Forward process.  ONE OU forward process per column (sigma^2, scales,
[tau_min, tau_max] = union of the per-rung ranges), independent of the rung
order (shared_forward_process).  Across rows compare 'xmse' (denoising
error in units of that shared sigma^2), not the F-loss.

A cell is INTERESTING only when it reads '+' and stays '+' going DOWN the
rows.  A gain that collapses from L1.5 to L2 was (b); a gain that survives L2
and grows with Delta is (c), the only result worth reporting.

Second panel: the conditional MEAN SHIFT, measured at fixed x from a K-member
solver ensemble (CERT.mean_shift_diagnostic), split into the x-independent
lattice term of the wavelet-diagonal forcing and the x-dependent part
(x_dependent_shift).  Per cell the emulator's own shift, its bias beyond
Monte Carlo (CERT.conditional_mean_bias) and corr(shift_solver,
shift_emulator) with a sign-flip null (CERT.shift_correlation).  Every null
resamples or sign-flips MEMBERS; none assumes independent orbits.

Cost.  The TLM probes dominate everything: n_probe * stride tangent steps
per stored pair (grid_cost_report, printed and gated by MAIN before
stage 1).  Probes are generated only if a rung needs them (L1.5: 'TLM',
L2: 'TLM_FULL', L1 only: none); a column can reuse the stage-4 dataset.
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax.numpy as jnp
from jax import random

from FORCING import (make_forced_stepper, make_forcing_spec, estimate_eps_inj,
                     rescale_to_eps_inj)
from STREAM import StreamMap, make_kernel_dequantizer
from TLM import shrink_rho_full, solver_conditional_ensemble
from TG import generate_transition_pair_dataset
from WSBG import WaveletDiffusionModel, make_conditional_sampler
from DIFF import train_conditional_score_model, split_by_trajectory
from ODE import make_wavelet_maps
from BENCH import make_phi_delta
from GROUP import sigma_bar_linear_, empirical_linear_floor
from CERT import (gain_verdict, mean_shift_diagnostic, conditional_mean_bias,
                  shift_correlation, forcing_regime_report)

RUNGS = ("L1", "L1.5", "L2")
C_TANGENT = 2.0      # cost of one tangent-linear (JVP) step in nonlinear-step units


def grid_sigma_mode(rungs) -> str:
    """Cheapest dataset mode that serves every requested rung."""
    rungs = set(rungs)
    if not rungs <= set(RUNGS):
        raise ValueError(f"unknown rungs {rungs - set(RUNGS)}")
    return "TLM_FULL" if "L2" in rungs else ("TLM" if "L1.5" in rungs else "B0")


def grid_strides(cfg):
    strides = [int(round(L / cfg["dt"])) for L in cfg["grid_lags"]]
    for L, s in zip(cfg["grid_lags"], strides):
        if abs(s * cfg["dt"] - L) > 1e-12:
            raise ValueError(f"lag {L} is not an integer multiple of dt = {cfg['dt']}")
    return strides


def main_column_reusable(cfg) -> bool:
    """True iff the stage-4 dataset can serve as the grid column at cfg['stride']."""
    need = grid_sigma_mode(cfg["grid_rungs"])
    have = cfg["sigma_mode"]
    order = {"B0": 0, "TLM": 1, "TLM_FULL": 2}
    return (cfg.get("grid_reuse_main_dataset", True)
            and cfg["stride"] in grid_strides(cfg)
            and order[have] >= order[need]
            and (need == "B0" or cfg["sigma_probes"] == cfg["grid_sigma_probes"]))


def grid_cost_report(cfg, verbose=True) -> dict:
    r"""A-priori cost of the grid, in nonlinear-step equivalents per dataset,

        n_traj [ n_burn + n_pairs ( (pair_gap + 1) stride + C n_probe stride ) ],

    (pair_gap forced lags + one noiseless lag for m per pair; C ~ 2 the cost
    of a tangent step) relative to the stage-4 dataset, whose wall time is
    known.  Both are probe-dominated, so the ratio is insensitive to C.
    Training: optimizer steps (upper bound, before early stopping) relative
    to stage 6.
    """
    def data_cost(n_traj, n_pairs, stride, n_probe):
        return n_traj * (cfg["n_burn"] + n_pairs * ((cfg["pair_gap"] + 1) * stride
                                                    + C_TANGENT * n_probe * stride))

    strides = grid_strides(cfg)
    rungs = list(cfg["grid_rungs"])
    mode = grid_sigma_mode(rungs)
    main_probe = 0 if cfg["sigma_mode"] == "B0" else cfg["sigma_probes"]
    main = data_cost(cfg["n_trajectories"], cfg["n_pairs_per_traj"], cfg["stride"], main_probe)
    reuse = main_column_reusable(cfg)
    g_probe = 0 if mode == "B0" else cfg["grid_sigma_probes"]
    cols = {s: (0.0 if (reuse and s == cfg["stride"]) else
                data_cost(cfg["grid_n_trajectories"], cfg["grid_n_pairs_per_traj"], s, g_probe))
            for s in strides}
    probe_share = {s: (0.0 if c == 0 else
                       cfg["grid_n_trajectories"] * cfg["grid_n_pairs_per_traj"]
                       * C_TANGENT * g_probe * s / c) for s, c in cols.items()}
    total = float(sum(cols.values()))
    main_steps = cfg["n_epochs"] * (1 - cfg["val_split"]) * cfg["n_trajectories"] \
        * cfg["n_pairs_per_traj"] / cfg["batch_size"]
    grid_steps = len(rungs) * len(strides) * cfg["grid_n_epochs"] * (1 - cfg["val_split"]) \
        * cfg["grid_n_trajectories"] * cfg["grid_n_pairs_per_traj"] / cfg["batch_size"]
    out = {"strides": strides, "mode": mode, "per_column": cols, "total": total,
           "stage4": float(main), "ratio_data": total / main, "reuse_main": reuse,
           "probe_share": probe_share, "ratio_train": grid_steps / main_steps}
    if verbose:
        print(f"  [grid cost] {len(rungs)} rungs {rungs} x {len(strides)} lags "
              f"{list(cfg['grid_lags'])}; dataset mode {mode}, "
              f"{g_probe} probes; {cfg['grid_n_trajectories']} x "
              f"{cfg['grid_n_pairs_per_traj']} pairs per column")
        for s, c in cols.items():
            tag = ("  (reuses the stage-4 dataset)" if c == 0 else
                   f"  ({probe_share[s]:.0%} of it TLM probes)")
            print(f"      Delta = {s * cfg['dt']:g}: {c / main:5.2f} x stage-4 data{tag}")
        print(f"      total data {out['ratio_data']:.2f} x stage 4; trainings "
              f"<= {out['ratio_train']:.2f} x stage 6 (optimizer steps)")
    return out



# 1.  Measured conditional mean shift (solver side)

def x_dependent_shift(samples, psi_ms, orbit, n_boot=200, seed=0) -> dict:
    r"""Split the measured shift  d(x) = E[S y | x] - S m(x)  into
        d(x) = ell + delta(x),
    ell the x-INDEPENDENT lattice term of the wavelet-diagonal forcing
    (CERT.delta_regime (b)) and delta(x) the x-dependent part, the only one
    the learned residual is for.  With n_s states and K members,
        ell_hat = mean_s d_s,   chi_x^2 = mean_{i,s} ((d_s - ell_hat)/sd_s)_i^2,
        E[chi_x^2 | delta = 0] = (1 - 1/n_s)/K,
    z_x = (chi_x^2 - (1 - 1/n_s)/K) / SE_boot, SE by resampling the members
    of every state.  Needs n_s >= 2 (with one state the parts are confounded).
    """
    n_s = len(samples)
    nan = float("nan")
    if n_s < 2:
        return {"chi_lattice": nan, "chi_x": nan, "chi_x_ng": nan, "z_x": nan}
    orb = np.asarray(orbit)
    n_orb = int(orb.max()) + 1
    cnt = np.bincount(orb, minlength=n_orb)
    K = samples[0].shape[0]

    def parts(Ys):
        raw = np.stack([Y.mean(0) - pm for Y, pm in zip(Ys, psi_ms)])
        sd = np.stack([np.sqrt(np.maximum((np.bincount(orb, weights=Y.var(0, ddof=1),
                                                       minlength=n_orb) / cnt)[orb], 1e-300))
                       for Y in Ys])
        ell = raw.mean(0)
        return float(np.mean(((raw - ell) / sd) ** 2)), float(np.mean((ell / sd.mean(0)) ** 2))

    chi2_x, lat2 = parts(samples)
    h0 = (1.0 - 1.0 / n_s) / K
    rng = np.random.default_rng(seed)
    boot = [parts([Y[rng.integers(0, K, K)] for Y in samples])[0] for _ in range(n_boot)]
    se = float(np.std(boot, ddof=1))
    return {"chi_lattice": float(np.sqrt(max(lat2 - 1.0 / (K * n_s), 0.0))),
            "chi_x": float(np.sqrt(chi2_x)),
            "chi_x_ng": float(np.sqrt(max(chi2_x - h0, 0.0))),
            "z_x": float((chi2_x - h0) / max(se, 1e-300))}


def solver_mean_shift(forced_step, det_step, stream, ops1d, states, stride,
                      orbit, n_rep=128, seed=0, chunk=16, verbose=True) -> dict:
    """Measured shift over several conditioning states x; keeps the members
    (the emulator comparison of every rung is paired with them)."""
    w2s, _ = make_wavelet_maps(ops1d, dtype=np.float32)
    phi = make_phi_delta(ops1d, det_step, stride)
    recs, samples, psi_ms = [], [], []
    for i, x in enumerate(np.asarray(states, np.float32)):
        U, V = w2s(jnp.asarray(x)[None])
        ys = solver_conditional_ensemble(forced_step, stream, U[0], V[0], stride,
                                         n_rep=n_rep, seed=seed + i, chunk=chunk)
        psi_m = np.asarray(stream.psi_w_from_state(jnp.asarray(phi(x))), np.float64)
        recs.append(mean_shift_diagnostic(ys, psi_m, orbit, seed=seed + i))
        samples.append(ys)
        psi_ms.append(psi_m)
    out = {k: float(np.mean([r[k] for r in recs])) for k in ("chi", "chi_mc", "chi_ng")}
    out["z"] = float(np.sum([r["z"] for r in recs]) / np.sqrt(len(recs)))  # Stouffer
    out.update(x_dependent_shift(samples, psi_ms, orbit, seed=seed))
    out["per_state"], out["samples"], out["psi_m"] = recs, samples, psi_ms
    out["n_states"] = len(recs)
    if verbose:
        print(f"    [mean shift] chi = {out['chi']:.4f} (MC floor {out['chi_mc']:.4f}); "
              f"total beyond MC chi_NG = {out['chi_ng']:.4f} (z = {out['z']:+.1f}); "
              f"x-INDEPENDENT lattice part chi_lat = {out['chi_lattice']:.4f}; "
              f"x-DEPENDENT part chi_x = {out['chi_x_ng']:.4f} (z = {out['z_x']:+.1f}) "
              f"over {len(recs)} conditioning states")
    return out


def emulator_samples(generate, params, stream, m_x, x, g_x, n_rep=128, n_steps=64,
                     guidance=1.0, chunk=16, seed=0) -> np.ndarray:
    """(n_rep, n^2) emulator draws of S y at one x."""
    out = []
    for i in range(0, n_rep, chunk):
        B = min(chunk, n_rep - i)
        o = generate(params, random.PRNGKey(seed + i),
                     np.broadcast_to(m_x, (B, m_x.size)), x, g_x,
                     n_steps=n_steps, guidance=guidance)
        out.append(np.asarray(stream.psi_w_from_state(o["y"]), np.float64))
    return np.concatenate(out)



# 2.  One column: everything that depends on Delta but not on the rung

def build_column(o, proj, orbit, spec, cfg, stride, *, E0, main=None,
                 verbose=True) -> dict:
    """Dataset, Sigma_lin, L2 shrinkage and solver mean shift at one lag.

    main = dict(ds, sigma_bar_z, sigma_bar_r) of the stage-4 dataset, used
    instead of a new one when this column's stride is the stage-4 stride
    (main_column_reusable); the column then has the stage-4 sample size.
    """
    dt, nu, mu_d = cfg["dt"], cfg["nu"], cfg["mu_d"]
    Delta = dt * stride
    rungs = list(cfg["grid_rungs"])
    t0 = time.time()
    if verbose:
        print(f"\n  === column Delta = {Delta:g} (stride {stride}) "
              f"===============================")

    forced_step, det_step, noise = make_forced_stepper(
        o, proj, dt, nu, spec, mu_d=mu_d,
        advection_scheme=cfg["advection_scheme"], triad_rtol=cfg["triad_rtol"],
        stoch_integrator=cfg["stoch_integrator"])
    stream = StreamMap(o, proj.filter_nyquist, dtype=jnp.float32)

    if main is not None:
        ds, sigma_bar_z, sigma_bar_r = main["ds"], main["sigma_bar_z"], main["sigma_bar_r"]
        if verbose:
            print(f"    [dataset] reusing the stage-4 dataset "
                  f"({len(np.unique(ds['traj_id']))} trajectories, {len(ds['pair_index'])} pairs)")
    else:
        amb = jnp.result_type(float)
        stream_hp = StreamMap(o, proj.filter_nyquist, dtype=amb)
        noise_hp = make_forced_stepper(
            o, proj, dt, nu, spec, mu_d=mu_d, advection_scheme=cfg["advection_scheme"],
            dtype=amb, stoch_integrator=cfg["stoch_integrator"])[2]
        sigma_bar_z, sigma_bar_r, _ = sigma_bar_linear_(
            o, proj, stream_hp, noise_hp, dt, nu, mu_d, stride, orbit, dtype=amb,
            verbose=verbose)
        ds = generate_transition_pair_dataset(
            o, spec, proj, dt, nu, mu_d, stride,
            n_trajectories=cfg["grid_n_trajectories"], n_burn=cfg["n_burn"],
            n_pairs_per_traj=cfg["grid_n_pairs_per_traj"], pair_gap=cfg["pair_gap"],
            seed=cfg["seed"], k_max=cfg["k_max"], spec_alpha=cfg["spec_alpha"], E0=E0,
            batch_size=cfg["dataset_gen_batch_size"],
            path=f"{cfg['grid_data_path']}_S{stride}",
            advection_scheme=cfg["advection_scheme"], triad_rtol=cfg["triad_rtol"],
            sigma_mode=grid_sigma_mode(rungs), sigma_probes=cfg["grid_sigma_probes"],
            sigma_lin=sigma_bar_r, stream=stream,
            stoch_integrator=cfg["stoch_integrator"], verbose=verbose)

    # Empirical-Bayes shrinkage of the per-coefficient ratio toward its orbit
    # mean.  Un-shrunk, the L2 surrogate carries
    # the probe noise straight into Sigma_bar and is a WORSE baseline than
    # L1.5.  snr_within is the validity flag of the L2 row.
    rho_shrunk, w_orb, snr_within = None, None, float("nan")
    if "L2" in rungs:
        rho_shrunk, w_orb, snr_within = shrink_rho_full(
            np.asarray(ds["rho_full"], np.float64), orbit, cfg["grid_sigma_probes"])
        if verbose:
            print(f"    [L2] within-orbit signal / probe noise = {snr_within:.2f}; "
                  f"shrinkage weight in [{w_orb.min():.2f}, {w_orb.max():.2f}]"
                  + ("" if snr_within > 1.0 else
                     "   <-- L2 not resolved at this probe count: read the L2 row as L1.5"))

    idx_tr, idx_val = split_by_trajectory(ds["traj_id"], cfg["val_split"], cfg["seed"])

    # Solver mean shift at a few validation states, paired with every rung.
    rng = np.random.default_rng(cfg["seed"] + stride)
    p_states = rng.choice(idx_val, size=min(cfg["grid_shift_states"], len(idx_val)),
                          replace=False)
    x_states = np.asarray(ds["x"][ds["pair_index"][p_states, 0]], np.float32)
    shift_sol = solver_mean_shift(forced_step, det_step, stream, o, x_states,
                                  stride, orbit, n_rep=cfg["grid_shift_members"],
                                  seed=cfg["seed"] + 77, chunk=cfg["compare_chunk"],
                                  verbose=verbose)

    return dict(stride=int(stride), Delta=float(Delta), ds=ds,
                idx_tr=idx_tr, idx_val=idx_val, reused_main=main is not None,
                sigma_bar_z=sigma_bar_z, sigma_bar_r=sigma_bar_r,
                stream=stream, forced_step=forced_step, det_step=det_step,
                rho_shrunk=rho_shrunk, w_orb=w_orb, snr_within=float(snr_within),
                shift_solver=shift_sol, p_states=p_states, x_states=x_states,
                wall=time.time() - t0)


def _surrogate_g(col, rung, n_orb):
    """Per-pair surrogate multiplier g (Np, width) for a rung."""
    ds = col["ds"]
    if rung == "L1":
        return np.ones((len(ds["pair_index"]), n_orb), np.float32)
    if rung == "L1.5":
        return np.asarray(ds["rho"], np.float32)
    if rung == "L2":
        return np.asarray(col["rho_shrunk"], np.float32)
    raise ValueError(f"unknown rung {rung!r}")


def shared_forward_process(o, proj, orbit, cfg, col, rungs, verbose=True) -> dict:
    """ONE OU forward process per column, independent of the rung order.

    sigma^2 = E[z^2] with the dequantiser at g = 1 (a fixed reference: the
    rungs differ from it only in the 2n kernel directions).  [t_min, T_diff]
    is the UNION of the per-rung ranges, so every rung has SNR >= snr_hi at
    t_min and SNR <= snr_lo at T_diff.  xmse is then comparable across rungs.
    """
    n_orb = int(orbit.max()) + 1
    ds, sbz = col["ds"], col["sigma_bar_z"]
    ref = WaveletDiffusionModel(o, proj, beta=cfg["beta"], lambda0=cfg["lambda0"])
    deq1 = make_kernel_dequantizer(col["stream"], sbz, orbit, dtype=jnp.float32)
    ref.calibrate(ds, col["idx_tr"], dequant=deq1, seed=cfg["seed"],
                  g=np.ones((len(ds["pair_index"]), n_orb), np.float32), verbose=verbose)
    t0s, t1s = [], []
    for rung in rungs:
        g_tr = _surrogate_g(col, rung, n_orb)[col["idx_tr"]]
        t0, t1 = ref.set_time_range(sbz, np.percentile(g_tr, 0.5, axis=0),
                                    np.percentile(g_tr, 99.5, axis=0),
                                    cfg["snr_hi"], cfg["snr_lo"], verbose=False)
        t0s.append(t0)
        t1s.append(t1)
    if verbose:
        print(f"    [forward process] tau in [{min(t0s):.3e}, {max(t1s):.3e}] "
              f"(union over rungs {list(rungs)})")
    return {"sigma_sq_diag": ref.sigma_sq_diag, "m_scale_diag": ref.m_scale_diag,
            "x_scale_diag": ref.x_scale_diag, "t_min": min(t0s), "T_diff": max(t1s)}



# 3.  One cell

def run_cell(o, proj, orbit, cfg, col, rung, *, verbose=True) -> dict:
    """Train + diagnose one (rung, Delta) cell on the column's forward process."""
    J = o.J
    n_orb = int(orbit.max()) + 1
    ds, stream = col["ds"], col["stream"]
    sigma_bar_z = col["sigma_bar_z"]
    stride = col["stride"]
    t0 = time.time()
    if verbose:
        print(f"\n    --- rung {rung} at Delta = {col['Delta']:g} ---")

    g_all = _surrogate_g(col, rung, n_orb)
    dequant = make_kernel_dequantizer(stream, sigma_bar_z, orbit, dtype=jnp.float32)
    diffusion = WaveletDiffusionModel(o, proj, beta=cfg["beta"], lambda0=cfg["lambda0"])
    for k, v in col["fwd"].items():              # shared by every rung of the column
        setattr(diffusion, k, v)

    ckpt = os.path.join(cfg["checkpoint_dir"],
                        f"grid_J{J}_S{stride}_{rung.replace('.', 'p')}.pkl")
    tr = train_conditional_score_model(
        ds, diffusion, sigma_bar_z, col["idx_tr"], col["idx_val"],
        n_epochs=cfg["grid_n_epochs"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], seed=cfg["seed"], channels=cfg["channels"],
        weighting=cfg["weighting"], snr_cap=cfg["snr_cap"],
        ema_decay=cfg["ema_decay"], patience=cfg["patience"],
        checkpoint_path=ckpt, dequant=dequant, g=g_all, verbose=verbose)
    floor = empirical_linear_floor(ds, diffusion, sigma_bar_z, col["idx_tr"],
                                   col["idx_val"], dequant=dequant, g=g_all,
                                   apply_F=tr["apply_F"], params=tr["params"],
                                   n_tau=cfg["floor_n_tau"], seed=cfg["seed"],
                                   verbose=verbose)

    # Emulator at the solver's states: shift, mean bias, shift correlation.
    generate = make_conditional_sampler(diffusion, tr["apply_F"], sigma_bar_z)
    phi = make_phi_delta(o, col["det_step"], stride)
    sh = col["shift_solver"]
    chis, chis_ng, biases, corrs = [], [], [], []
    for i, (p, x) in enumerate(zip(col["p_states"], col["x_states"])):
        m_x = np.asarray(phi(x))
        psi_m = sh["psi_m"][i]
        g_x = None if rung == "L1" else jnp.asarray(g_all[p])[None]
        ye = emulator_samples(generate, tr["params"], stream, m_x, x, g_x,
                              n_rep=cfg["grid_shift_members"], n_steps=cfg["n_gen_steps"],
                              guidance=cfg["guidance"], chunk=cfg["compare_chunk"],
                              seed=cfg["seed"] + 5000 + i)
        ys = sh["samples"][i]
        e = mean_shift_diagnostic(ye, psi_m, orbit, seed=cfg["seed"] + i)
        chis.append(e["chi"])
        chis_ng.append(e["chi_ng"])
        biases.append(conditional_mean_bias(ys, ye, psi_m, orbit, seed=cfg["seed"] + i))
        corrs.append(shift_correlation(ys, ye, psi_m, orbit, seed=cfg["seed"] + i))

    n_s = len(biases)
    b2 = float(np.mean([b["b2"] for b in biases]))
    b2_se = float(np.sqrt(np.sum([b["se_boot"] ** 2 for b in biases])) / n_s)
    corr = float(np.mean([c["corr"] for c in corrs]))
    corr_null = float(np.sqrt(np.sum([c["null_sd"] ** 2 for c in corrs])) / n_s)
    cell = dict(
        rung=rung, stride=int(stride), Delta=float(col["Delta"]),
        gain=float(tr["best_gain"]), gain_se=float(tr["best_gain_se"]),
        val_base=float(tr["val_base"]), best_val=float(tr["best_val"]),
        levels=floor["per_level"],
        floor_gain=float(floor["gain_lin"]), floor_loss=float(floor["loss"]["lin"]),
        excess=float(floor["excess"]),
        excess_z=float(floor["excess"] / max(floor["excess_se"], 1e-12)),
        xmse=floor["xmse"],
        mean_bias_emulator=float(np.sqrt(max(b2, 0.0))),
        mean_bias_z=float(b2 / max(b2_se, 1e-300)),
        chi_solver=sh["chi"], chi_ng_solver=sh["chi_ng"], z_shift_solver=sh["z"],
        chi_emulator=float(np.mean(chis)), chi_ng_emulator=float(np.mean(chis_ng)),
        shift_corr=corr, shift_corr_null_sd=corr_null,
        snr_within=col["snr_within"], wall=time.time() - t0)
    v = gain_verdict(floor["excess"], floor["excess_se"])
    cell["verdict"], cell["verdict_code"] = v["verdict"], v["code"]
    if verbose:
        print(f"      gain {cell['gain']:+.5f} +- {cell['gain_se']:.5f}; linear floor "
              f"{cell['floor_gain']:+.5f}; network - floor {cell['excess']:+.5f} "
              f"({cell['excess_z']:+.1f} SE); xmse F0/lin/net "
              + "/".join(f"{x:.4f}" for x in floor["xmse"].values())
              + f"; chi_emu {cell['chi_emulator']:.4f} vs chi_sol {cell['chi_solver']:.4f}; "
                f"corr {corr:+.3f} (null sd {corr_null:.3f}); mean bias emu "
                f"{cell['mean_bias_emulator']:.4f} (z = {cell['mean_bias_z']:+.1f}) vs F=0 "
                f"{cell['chi_ng_solver']:.4f} (z = {cell['z_shift_solver']:+.1f})")
        print(f"      verdict: {cell['verdict']}")
    return cell



# 4.  The grid

def run_rung_lag_grid(o, proj, orbit, cfg, *, main=None, fig_dir=None,
                      verbose=True) -> dict:
    """The headline 2-D result.  Columns outer: data and Sigma_lin are shared.

    main = dict(ds, sigma_bar_z, sigma_bar_r) of the stage-4 dataset; it
    is used as the column at cfg['stride'] when main_column_reusable.
    """
    spec = make_forcing_spec(o.J, o.jmin, q0=1.0, alpha_F=cfg["alpha_F"])
    spec = rescale_to_eps_inj(estimate_eps_inj(o, proj, spec, verbose=False),
                              cfg["eps_inj"])
    E0 = forcing_regime_report(spec, cfg["nu"], cfg["mu_d"],
                               verbose=False)["E_stationary_estimate"]
    lags = list(cfg["grid_lags"])
    rungs = list(cfg["grid_rungs"])
    strides = grid_strides(cfg)
    reuse = main is not None and main_column_reusable(cfg)
    if verbose:
        print(f"\n[grid] {len(rungs)} rungs x {len(lags)} lags = "
              f"{len(rungs) * len(lags)} trainings; strides {strides}")

    cells, cols = {}, {}
    for s in strides:
        col = build_column(o, proj, orbit, spec, cfg, s, E0=E0,
                           main=main if (reuse and s == cfg["stride"]) else None,
                           verbose=verbose)
        col["fwd"] = shared_forward_process(o, proj, orbit, cfg, col, rungs,
                                            verbose=verbose)
        cols[s] = col
        for rung in rungs:
            cells[(rung, s)] = run_cell(o, proj, orbit, cfg, col, rung, verbose=verbose)
        # free the column's large arrays before the next lag
        col["ds"] = {k: v for k, v in col["ds"].items() if k not in ("x", "m", "rho_full")}
        col["rho_shrunk"] = None
        col["shift_solver"] = {k: v for k, v in col["shift_solver"].items()
                               if k not in ("samples", "per_state")}

    grid = dict(rungs=rungs, lags=lags, strides=strides, cells=cells, spec=spec,
                columns={s: {k: cols[s][k] for k in
                             ("Delta", "snr_within", "shift_solver", "reused_main")}
                         for s in strides})

    def _pull(key):
        return np.array([[np.nan if cells[(r, s)].get(key) is None
                          else cells[(r, s)][key] for s in strides]
                         for r in rungs], float)

    for key in ("gain", "gain_se", "floor_gain", "excess", "excess_z",
                "chi_solver", "chi_ng_solver", "chi_emulator", "shift_corr",
                "shift_corr_null_sd", "mean_bias_emulator", "mean_bias_z"):
        grid[key] = _pull(key)
    if verbose:
        print_grid(grid)
    if fig_dir is not None:
        from PLOT import plot_rung_lag_grid
        plot_rung_lag_grid(grid, save_path=os.path.join(fig_dir, "grid.png"))
    return grid


def print_grid(grid: dict) -> None:
    rungs, lags = grid["rungs"], grid["lags"]
    W = 13

    def row(name, arr, fmt="{:+.5f}"):
        print(f"    {name:<8s}" + "".join(fmt.format(v).rjust(W)
                                          if np.isfinite(v) else "n/a".rjust(W)
                                          for v in arr))

    print("\n" + "=" * 72)
    print("  RUNG x LAG GRID")
    print("=" * 72)
    hdr = "    " + " " * 8 + "".join(f"D={L:g}".rjust(W) for L in lags)

    print("\n  training gain over that rung's own F = 0 baseline")
    print(hdr)
    for i, r in enumerate(rungs):
        row(r, grid["gain"][i])
    print("\n  standard error of the paired gain")
    print(hdr)
    for i, r in enumerate(rungs):
        row(r, grid["gain_se"][i], "{:.5f}")

    print("\n  gain of the EMPIRICAL linear floor over F = 0 (per cell, same units")
    print("  as the row's gain)")
    print(hdr)
    for i, r in enumerate(rungs):
        row(r, grid["floor_gain"][i])

    print("\n  network - floor, paired, in SE  (> 3 = x-dependent structure")
    print("  beyond any x-independent linear map)")
    print(hdr)
    for i, r in enumerate(rungs):
        row(r, grid["excess_z"][i], "{:+.1f}")

    print("\n  VERDICT  (+ above floor, 0 at floor, - BELOW floor = under-fitted,")
    print("  ? undetermined).  A '-' cell says NOTHING about non-Gaussianity.")
    print(hdr)
    for r in rungs:
        print(f"    {r:<8s}" + "".join(grid["cells"][(r, s)]["verdict_code"].rjust(W)
                                    for s in grid["strides"]))

    print("\n  conditional mean shift chi (solver, per column) and the")
    print("  emulator's reproduction of it (per cell)")
    print(hdr)
    row("solver", grid["chi_solver"][0], "{:.4f}")
    row(" non-G", grid["chi_ng_solver"][0], "{:.4f}")
    row(" x-dep", [grid["columns"][s]["shift_solver"]["chi_x_ng"] for s in grid["strides"]],
        "{:.4f}")
    row(" z x-dep", [grid["columns"][s]["shift_solver"]["z_x"] for s in grid["strides"]],
        "{:+.1f}")
    for i, r in enumerate(rungs):
        row(r, grid["chi_emulator"][i], "{:.4f}")

    print("\n  emulator conditional-mean error beyond MC (solver sd units);")
    print("  the F = 0 sampler's error is the 'non-G' row above")
    print(hdr)
    for i, r in enumerate(rungs):
        row(r, grid["mean_bias_emulator"][i], "{:.4f}")
    print(hdr)
    for i, r in enumerate(rungs):
        row(r, grid["mean_bias_z"][i], "{:+.1f}")

    print("\n  corr(shift_solver, shift_emulator)  [sign-flip null sd below]")
    print(hdr)
    for i, r in enumerate(rungs):
        row(r, grid["shift_corr"][i], "{:+.3f}")
    for i, r in enumerate(rungs):
        row(f" null {r}", grid["shift_corr_null_sd"][i], "{:.3f}")

    print("\n  reading: a column where chi_NG(solver) is at the MC floor has no")
    print("  non-Gaussian mean shift to learn, whatever its gain column says;")
    print("  a gain that collapses from L1.5 to L2 was the within-orbit diagonal,")
    print("  not non-Gaussianity.")
    print("=" * 72 + "\n")
