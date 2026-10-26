"""
MAIN.py Stochastic Wavelet Emulator of 2-D Navier-Stokes

Object.  The transition kernel p_Delta(y | x) of the stochastically forced,
Ekman-damped, Bubnov wavelet-Galerkin Navier-Stokes scheme, learned as

    y = Phi_Delta(x) + curl_h(r),   r | x ~ p(r | x),
    score(r | x) = diagonal Gaussian SURROGATE N(0, diag Sigma_bar(x))
                +  learned residual F_theta,
    Sigma_bar(x) = Sigma_bar_z * g(x):  Sigma_bar_z the dequantised orbit
                diagonal of the Stokes covariance Sigma_lin (, GROUP),
                g = 1 ('B0') or the orbit-pooled (TLM) ratio
                rho(x) ('TLM', 'TLM_FULL'). 

The scientific object is the kernel, not speed: with 'TLM' every emulator
lag runs n_probe tangent-linear rollouts, which costs more than
the solver lag it replaces.

Pipeline
1. Operators          connection coefficients, Leray projector
2. Deterministic      Taylor-Green (with drag, production stepper), advection
validation         consistency, skew symmetry, rough-field energy, order
3. Forcing            stream-function forcing (diagonal in the generative
                    coordinates), eps_inj, exponential EM;
                    stochastic Taylor-Green: closed-form law of the forced
                    NONLINEAR solver on the invariant TG shell;
                    Stokes surrogate and dequantiser
4. Dataset            forced, decorrelated transition pairs
5. Surrogate          forward-process scaling (sigma^2), diffusion time range,
                    surrogate vs empirical conditional variances
6. Training           residual denoising score matching; paired EMPIRICAL
                    linear floor (GROUP.empirical_linear_floor) and
                    CERT.gain_verdict.  A '+' means x-dependent structure,
                    NOT non-Gaussianity (decided at fixed x, stage 7 / GRID)
7. Certificates       Gaussian sampler (F = 0); conditional law at a
                    fixed x vs a solver ensemble: variance (pooled and
                    within-orbit), mean shift, one-lag energy split
8. Emulation          solver ENSEMBLE vs emulator ENSEMBLE from the same
                    dataset state: E, Z, spread, ACC, spectra, level
                    moments, vorticity PDFs, energy budget
9. Benchmark (opt.)   closed-form linear benchmark of the emulator, which
                    also certifies the empirical floor (BENCH)
10. Grid (opt.)       rung x lag grid (GRID.run_rung_lag_grid); its cost is
                    reported and gated BEFORE stage 1

Physics gates.  CERT.forcing_regime_report (a priori, stage 3) and
CERT.resolution_report (a posteriori, stage 4) raise when
cfg['strict_physics'] is True and only warn otherwise (SMOKE: J = 5 cannot
host a drag-dominated forced range AND a resolved enstrophy tail; the smoke
run tests the plumbing, not the physics).

Physics.  d<E_gal>/dt = eps_inj - 2 nu <Z_gal> - 2 mu_d <E_gal> is for
the SEMI-DISCRETE (continuous-time) system: Bubnov scheme, skew advection,
G-orthogonal projector.  The time integrator is energy-neutral in advection
only up to O(dt^5) per step (RK4 on the imaginary axis:
|R(iy)|^2 = 1 - y^6/72 + ...), which is what the measured budget closure
contains.  Time is measured in drag times 1/mu_d throughout.

All scalar results are written to {fig_dir}/metrics.json.  Run with
NSWSGM_X64=1: operator validation, Richardson order and the stochastic TG
benchmark need float64; data, training and sampling run in float32.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import warnings
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use("Agg")

from PLOT import (plot_dataset_overview, plot_sigma_calibration, plot_training,
                plot_stochastic_tg_solver, plot_vorticity_comparison,
                plot_ensemble_statistics)
from OPERATORS import (compute_operators, build_2d_operators,
                    build_leray_projector_full, validate_operators)
from ODE import (validate_advection_operator, validate_skew_symmetry,
                validate_energy_conservation_rough, make_wavelet_maps)
from TG import (validate_taylor_green_, validate_time_integrator_order,
                generate_transition_pair_dataset, validate_stochastic_taylor_green)
from FORCING import (build_orbit_map_2d, make_forcing_spec,
                    estimate_eps_inj, rescale_to_eps_inj, check_energy_budget,
                    stationary_identity, validate_orbit_map, make_forced_stepper,
                    orbit_pool)
from TLM import (make_sigma_ratio_fn, solver_conditional_ensemble,
                    gaussian_sampler_test)
from CERT import (forcing_regime_report, resolution_report, orbit_mc_floor,
                variance_test, delta_regime, gain_verdict, mean_shift_diagnostic,
                conditional_mean_bias, shift_correlation,
                within_orbit_variance_structure)
from STREAM import (StreamMap, validate_stream_map, make_kernel_dequantizer,
                    validate_kernel_dequantizer, validate_vorticity_consistency)
from WSBG import WaveletDiffusionModel, make_conditional_sampler
from DIFF import train_conditional_score_model, split_by_trajectory
from BENCH import (make_phi_delta, make_observables, solver_ensemble,
                emulator_ensemble, compare_ensembles, print_comparison,
                run_tg_diffusion_benchmark, one_lag_energy_split)
from GRID import run_rung_lag_grid, grid_cost_report
from GROUP import sigma_bar_linear_, empirical_linear_floor


# CONFIGURATION

config: Dict[str, Any] = {
    # Discretisation
    'J': 8, 'jmin': 4,
    'test_space': 'primal',        # Bubnov-Galerkin ( energy-neutral)
    'advection_scheme': 'if_rk4',
    'triad_rtol': 0.0,
    'filter_nyquist': True,

    # Physics
    'nu': 1e-4,
    'mu_d': 0.5,                   # Ekman drag; drag time 1/mu_d = 2
    'dt': 1e-3,
    'k_max': 5,
    'spec_alpha': 5 / 3,
    'k0': 2,                       # Taylor-Green shell (k0 = 2: 4 modes)

    # Forcing (stream-function, wavelet-diagonal)
    'alpha_F': 5,                  # alpha_F = 4 injects 1.8% of the enstrophy at
                                # the finest level (J=8): forced grid-scale tail
    'eps_inj': 0.05,
    'stoch_tg_E': 0.25,            # stationary energy of the stochastic-TG test
    'stoch_tg_members': 32,

    # Dataset
    'n_trajectories': 128,
    'n_burn': 8000,                # n_burn dt mu_d = 4 drag times
    'n_pairs_per_traj': 64,
    'stride': 100,                 # Delta = stride*dt = 0.1 ~ 1.6 eddy times
                                # 1/(k_f u_rms).  CERT.delta_regime: the
                                # short-lag expansion is at its limit here
                                # (Delta k_f u_rms > 1): an indication only.
                                # Sigma_TLM(x) (noiseless-path
                                # matrix) is a leading-order model.
    'pair_gap': 2,                 # 0.2 time units between stored pairs: ~3 eddy
                                # times 1/(k_f u_rms) (small scales decorrelated)
                                # but only 0.1 drag times 1/mu_d (energy-containing
                                # scales NOT decorrelated).  Pairs of one trajectory
                                # are dependent: every SE is clustered by trajectory.
    'sigma_mode': 'B0',           # 'B0' | 'TLM' | 'TLM_FULL' (same cost as 'TLM';
                                # also stores rho_full, so the grid can reuse
                                # this dataset as its Delta = stride column)
    'sigma_probes': 8,             # TLM cost: probes x stride tangent steps per pair
    'stoch_integrator': 'exp',
    'dataset_gen_batch_size': 8,
    'sigma_validation_reps': 64,

    # Diffusion model
    'beta': 1.5, 'lambda0': 1.0,
    'snr_hi': 1e2, 'snr_lo': 1e-3,

    # Training
    'n_epochs': 200, 'batch_size': 16, 'lr': 2e-4,
    'channels': (32, 64, 96),
    'weighting': 'unit', 'snr_cap': 5.0,      # the linear floor requires 'unit'
    'val_split': 0.2, 'ema_decay': 0.999, 'patience': 20,

    # Sampling
    'n_gen_steps': 64, 'eta': 1.0, 'guidance': 1.0,
    'gauss_test_samples': 256,
    'floor_n_tau': 16,             # tau quadrature of the empirical linear floor

    # Ensemble comparison (stage 8)
    'compare_members': 16,
    'compare_T': 2.0,              # time units (= mu_d T drag times)
    'compare_n_snap': 6,
    'compare_chunk': 16,

    # Closed-form diffusion benchmark (stage 9)
    'run_bench_tg': False,
    'bench_n_trajectories': 64,
    'bench_n_pairs_per_traj': 32,
    'bench_n_epochs': 40,
    'bench_n_members': 64,
    'bench_n_lags': 100,
    'bench_data_path': './data/bench_linear',

    # Rung x lag grid (the headline result).  Cost: GRID.grid_cost_report,
    # printed and gated (grid_max_cost_ratio x the stage-4 data cost) before
    # stage 1.  Probes dominate: n_probe x stride tangent steps per pair.
    'run_grid': False,
    'grid_lags': (0.02, 0.1, 0.2), # control column (Delta = 0.02: x-dependent
                                # shift below detection) + two informative ones
    'grid_rungs': ('L1.5', 'L2'),  
    'grid_n_trajectories': 64,
    'grid_n_pairs_per_traj': 32,   # 64 x 32 = 2048 pairs; floor needs >= 5 n_orb
    'grid_n_epochs': 120,
    'grid_sigma_probes': 8,       
    'grid_reuse_main_dataset': True,   
    'grid_max_cost_ratio': 1.5,    # refuse to start above this x stage-4 data
    'grid_shift_members': 128,     # K in the mean-shift diagnostic
    'grid_shift_states': 4,        # conditioning states x it is averaged over
    'grid_data_path': './data/grid',

    # Tolerances
    'tol_tg': 1e-3, 'tol_adv_consistency': 1e-3, 'tol_adv_energy': 1e-8,
    'tol_budget': 1.0, 'time_order_T_steps': 100,
    'tol_top_enstrophy_injection': 0.01, 'tol_tail_enstrophy': 0.05,
    'strict_physics': True,

    # I/O
    'data_path': './data/transition',
    'checkpoint_dir': './checkpoints',
    'fig_dir': './figures',
    'seed': 42,
    'verbose': True,
}

SMOKE = dict(
    J=5, jmin=3, nu=2e-3, dt=2e-3, alpha_F=7, strict_physics=False,
    n_trajectories=25, dataset_gen_batch_size=5, n_burn=3000,  # mu_d n_burn dt = 3;
                                # val_split 0.2 -> 5 val trajectories (cluster SE
                                # usable); 20 x 8 = 160 fit pairs >= 5 n_orb (floor)
    n_pairs_per_traj=8, stride=5, pair_gap=2,
    sigma_mode='TLM_FULL', sigma_probes=4, sigma_validation_reps=16,
    n_epochs=2, batch_size=4, channels=(8, 16), patience=5,
    n_gen_steps=8, gauss_test_samples=32,
    stoch_tg_members=32,          # K = 8: Bonferroni-t threshold 11 (no power); K = 32: ~5
    compare_members=4, compare_T=0.1, compare_n_snap=3, compare_chunk=4,
    bench_n_trajectories=4, bench_n_pairs_per_traj=8, bench_n_epochs=2,
    bench_n_members=8, bench_n_lags=8,
    data_path='./data/smoke', bench_data_path='./data/smoke_bench',
    checkpoint_dir='./checkpoints_smoke', fig_dir='./figures_smoke',
    time_order_T_steps=20,
    run_grid=True, grid_lags=(0.01, 0.02), grid_rungs=('L1', 'L2'),
    grid_n_trajectories=25, grid_n_pairs_per_traj=8, grid_n_epochs=2,
    grid_sigma_probes=4, grid_shift_members=16, grid_max_cost_ratio=10.0,
    grid_shift_states=2,          # >= 2 states separate the lattice shift from delta(x)
    grid_data_path='./data/smoke_grid',
)


def build_operators(cfg):
    o = compute_operators(cfg['J'], cfg['jmin'], verbose=cfg['verbose'],
                        test_space=cfg['test_space'])
    validate_operators(o, verbose=cfg['verbose'])
    validate_orbit_map(o)
    ops = build_2d_operators(o)
    proj = build_leray_projector_full(ops, filter_nyquist=cfg['filter_nyquist'])
    orbit = build_orbit_map_2d(cfg['J'], cfg['jmin'], n_comp=1)
    return o, ops, proj, orbit


class Metrics(dict):
    def put(self, key, value):
        self[key] = value

    def dump(self, path):
        def conv(v):
            if isinstance(v, dict):
                return {str(k): conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist() if v.size <= 64 else f"<array {v.shape}>"
            if isinstance(v, (bool, int, float, str)) or v is None:
                return v
            return str(v)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, "w") as f:
            json.dump(conv(dict(self)), f, indent=2)
        print(f"  → Saved: {path}")


def physics_gate(cfg, ok: bool, message: str) -> None:
    """Raise under cfg['strict_physics'], warn otherwise (smoke runs)."""
    if ok:
        return
    if cfg.get('strict_physics', True):
        raise RuntimeError(message)
    warnings.warn(message + " [strict_physics=False: continuing]")


def check_config(cfg) -> None:
    """Every configuration error that would otherwise surface after hours."""
    if cfg['sigma_mode'] not in ('B0', 'TLM', 'TLM_FULL'):
        raise ValueError(f"sigma_mode must be 'B0', 'TLM' or 'TLM_FULL', got {cfg['sigma_mode']!r}")
    if cfg['weighting'] != 'unit':
        raise ValueError("GROUP.empirical_linear_floor scores the unit-weighted F-loss: "
                        "weighting must be 'unit'")
    if cfg['mu_d'] <= 0:
        raise ValueError("mu_d must be > 0: no statistically stationary state without drag")
    if cfg.get('run_grid'):
        rep = grid_cost_report(cfg, verbose=cfg['verbose'])
        if rep['ratio_data'] > cfg['grid_max_cost_ratio']:
            raise RuntimeError(
                f"grid data cost {rep['ratio_data']:.2f} x stage 4 exceeds "
                f"grid_max_cost_ratio = {cfg['grid_max_cost_ratio']}.  Levers, in "
                "order of effect: fewer/smaller grid_lags (cost ~ stride), "
                "grid_sigma_probes, grid_n_pairs_per_traj, grid_n_trajectories; "
                "sigma_mode='TLM_FULL' lets the grid reuse the stage-4 dataset; "
                "or raise grid_max_cost_ratio deliberately.")


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    V = cfg['verbose']
    check_config(cfg)
    for k in ('checkpoint_dir', 'fig_dir'):
        os.makedirs(cfg[k], exist_ok=True)
    os.makedirs(os.path.dirname(cfg['data_path']) or '.', exist_ok=True)
    fig = lambda name: os.path.join(cfg['fig_dir'], name)
    metrics = Metrics(config={k: v for k, v in cfg.items() if not k.startswith('_')})
    rng = np.random.default_rng(cfg['seed'])
    J, jmin = cfg['J'], cfg['jmin']
    n = 2 ** J
    dt, nu, mu_d, stride = cfg['dt'], cfg['nu'], cfg['mu_d'], cfg['stride']
    Delta = stride * dt
    x64 = bool(jax.config.jax_enable_x64)
    if not x64:
        warnings.warn("NSWSGM_X64 is not set: float64 validations are skipped or degraded")

    
    # 1. OPERATORS
    
    if V: print("\n[1/10] Building wavelet operators")
    o, ops, proj, orbit = build_operators(cfg)
    if V:
        print(f"    Grid {n}x{n}; velocity state dim {2 * n * n}; generative dim {n * n}; "
            f"levels j = {jmin - 1} (V_{jmin}), {jmin}..{J - 1} (W_j); "
            f"dim ker(C G2^-1 C^T) = {proj.kernel_dim} (pressure Schur complement)")

    
    # 2. DETERMINISTIC VALIDATION
    
    if V: print("\n[2/10] Deterministic physics validation")
    tg = validate_taylor_green_(
        o, proj, k0=cfg['k0'], nu=nu, dt=dt, n_steps=100, tol=cfg['tol_tg'],
        advection_scheme=cfg['advection_scheme'], triad_rtol=cfg['triad_rtol'], mu_d=mu_d)
    tg_err = tg['rel_err']
    if tg_err > cfg['tol_tg']:
        raise RuntimeError(f"Taylor-Green validation failed: {tg_err:.3e} "
                        f"(solver {tg['solver_err']:.1e}, spatial {tg['spatial_err']:.1e})")
    adv_c, adv_e = validate_advection_operator(o, proj)
    if adv_c > cfg['tol_adv_consistency'] or adv_e > cfg['tol_adv_energy']:
        raise RuntimeError("advection validation failed")
    skew = validate_skew_symmetry(o, proj)
    rough = validate_energy_conservation_rough(o, proj)
    if cfg['test_space'] == 'primal' and rough > (1e-10 if x64 else 1e-5):
        raise RuntimeError(f"Bubnov advection not skew: {rough:.2e}")
    order = None
    if x64:
        order, _ = validate_time_integrator_order(
            o, proj, nu=nu, dt=dt, T=cfg['time_order_T_steps'] * dt,
            scheme=cfg['advection_scheme'], triad_rtol=cfg['triad_rtol'])
    metrics.put('deterministic', dict(tg=tg, adv_consistency=adv_c,
                                    adv_energy=adv_e, skew=skew['skew'],
                                    rough_energy=rough, time_order=order))
    if cfg.get('_only_validate'):
        metrics.dump(fig('metrics.json'))
        return {'metrics': metrics}

    
    # 3. FORCING + STOCHASTIC SOLVER VALIDATION + STOKES SURROGATE
    
    if V: print("\n[3/10] Stochastic forcing")
    validate_stream_map(o, proj)
    validate_vorticity_consistency(o, proj, verbose=V)
    stream = StreamMap(o, proj.filter_nyquist, dtype=jnp.float32)
    spec = make_forcing_spec(J, jmin, q0=1.0, alpha_F=cfg['alpha_F'])
    spec = rescale_to_eps_inj(estimate_eps_inj(o, proj, spec), cfg['eps_inj'])
    if V:
        print(f"    q0 = {spec.q0:.4e} -> eps_inj = {spec.eps_inj:.4e}; drag-only bound "
            f"eps/(2 mu_d) = {spec.eps_inj / (2 * mu_d):.4e}")
    reg = forcing_regime_report(spec, nu, mu_d, tol_top=cfg['tol_top_enstrophy_injection'])
    metrics.put('forcing_regime', {k: reg[k] for k in (
        'k_f', 'global_viscous_share', 'global_drag_share', 'injection_peak_level',
        'top_level_enstrophy_injection', 'E_stationary_estimate', 'ok', 'reasons')})
    physics_gate(cfg, reg['ok'], "forcing regime: " + "; ".join(reg['reasons']))
    u_rms_est = float(np.sqrt(2.0 * reg['E_stationary_estimate']))
    dreg = delta_regime(Delta, spec.eps_inj, spec.k_f, u_rms_est, cfg['sigma_validation_reps'])
    metrics.put('delta_regime', {k: (bool(v) if isinstance(v, bool) else float(v))
                                for k, v in dreg.items()})
    forced_step, det_step, noise = make_forced_stepper(
        o, proj, dt, nu, spec, mu_d=mu_d, advection_scheme=cfg['advection_scheme'],
        triad_rtol=cfg['triad_rtol'], stoch_integrator=cfg['stoch_integrator'])

    stg = validate_stochastic_taylor_green(
        o, proj, dt, nu, mu_d, k0=cfg['k0'], E_star=cfg['stoch_tg_E'],
        n_members=cfg['stoch_tg_members'], advection_scheme=cfg['advection_scheme'],
        seed=cfg['seed'])
    plot_stochastic_tg_solver(stg, save_path=fig('validation_stochastic_tg.png'))
    metrics.put('stochastic_tg_solver', {k: stg[k] for k in (
        'gamma', 'eig_residual', 'z_mean_max', 'var_dev_max', 'zE_max',
        'off_shell_max', 'ks_pvalue', 'threshold', 'ok', 'dtype')})
    if not stg['ok']:
        warnings.warn("stochastic Taylor-Green benchmark FAILED: the forced solver "
                    "does not reproduce the closed-form OU law")

    # Stokes surrogate in the dyadic-group block basis (GROUP.py):
    # Sigma_bar_r, the dequantiser fixed point Sigma_bar_z, at ANY J, no MC.
    amb = jnp.result_type(float)
    stream_hp = StreamMap(o, proj.filter_nyquist, dtype=amb)
    noise_hp = make_forced_stepper(o, proj, dt, nu, spec, mu_d=mu_d,
                                advection_scheme=cfg['advection_scheme'], dtype=amb,
                                stoch_integrator=cfg['stoch_integrator'])[2]
    sigma_bar_z, sigma_bar_r, _ = sigma_bar_linear_(
        o, proj, stream_hp, noise_hp, dt, nu, mu_d, stride, orbit, dtype=amb, verbose=V)
    validate_kernel_dequantizer(stream, sigma_bar_z, orbit, n_samples=4096)
    dequant = make_kernel_dequantizer(stream, sigma_bar_z, orbit, dtype=jnp.float32)

    
    # 4. DATASET
    
    if V: print("\n[4/10] Transition dataset")
    ds = generate_transition_pair_dataset(
        o, spec, proj, dt, nu, mu_d, stride,
        n_trajectories=cfg['n_trajectories'], n_burn=cfg['n_burn'],
        n_pairs_per_traj=cfg['n_pairs_per_traj'], pair_gap=cfg['pair_gap'],
        E0=reg['E_stationary_estimate'],          # not the drag-only upper bound
        seed=cfg['seed'], k_max=cfg['k_max'], spec_alpha=cfg['spec_alpha'],
        batch_size=cfg['dataset_gen_batch_size'], path=cfg['data_path'],
        advection_scheme=cfg['advection_scheme'], triad_rtol=cfg['triad_rtol'],
        sigma_mode=cfg['sigma_mode'], sigma_probes=cfg['sigma_probes'],
        sigma_lin=sigma_bar_r, stream=stream, stoch_integrator=cfg['stoch_integrator'])
    cfl = float(ds['cfl'].max())
    if V: print(f"    CFL max = {cfl:.3f} (limit {ds['cfl_limit']:.3f})")
    if cfl > ds['cfl_limit']:
        raise RuntimeError("CFL violation")
    eb = ds['E_burn']
    q = max(1, eb.shape[1] // 4)
    dE = ds['E_path'].mean(1) - eb[:, -q:].mean(1)
    t_drift = float(dE.mean() / (dE.std(ddof=1) / np.sqrt(len(dE)))) if len(dE) > 1 else 0.0
    if abs(t_drift) > 3:
        warnings.warn("energy still drifting after burn-in: increase n_burn")
    ratios = [check_energy_budget(ds['E_path'][t], ds['Z_path'][t], Delta, nu, mu_d,
                                spec.eps_inj, window=min(16, ds['E_path'].shape[1] - 1)
                                )['max_ratio'] for t in range(min(8, cfg['n_trajectories']))]
    id_ref = stationary_identity(ds['E_path'], ds['Z_path'], nu, mu_d, spec.eps_inj)
    T_traj = cfg['n_pairs_per_traj'] * cfg['pair_gap'] * Delta
    if V:
        print(f"    burn-in energy {eb[:, :q].mean():.4e} -> {eb[:, -q:].mean():.4e}; "
            f"stationarity drift t = {t_drift:+.1f}")
        print(f"    energy budget: max |residual|/bound = {max(ratios):.2f}")
        print(f"    retained window T = {T_traj:.3g} ({T_traj * mu_d:.2f} drag times); "
            f"pairs {cfg['pair_gap'] * Delta:.3g} apart")
        print(f"    stationary identity: {id_ref['lhs']:.4e} vs {spec.eps_inj:.4e} "
            f"({id_ref['rel_gap']:+.2%}, {id_ref['n_sigma']:.1f} sigma)")
    if max(ratios) > cfg['tol_budget']:
        raise RuntimeError("energy budget violated")
    metrics.put('dataset', dict(cfl=cfl, drift_t=t_drift, budget_ratio=max(ratios),
                                identity=id_ref, T_traj=T_traj))
    plot_dataset_overview(ds, nu, mu_d, spec.eps_inj, save_path=fig('dataset_overview.png'))
    rows = np.sort(rng.choice(len(ds['x']), size=min(64, len(ds['x'])), replace=False))
    res = resolution_report(np.asarray(ds['x'][rows]), o, nu, mu_d, spec.eps_inj,
                            E_gal=ds['E_path'], Z_gal=ds['Z_path'],
                            tail_tol=cfg['tol_tail_enstrophy'])
    metrics.put('resolution', {k: float(res[k]) for k in (
        'tail_enstrophy_above_n_over_4', 'tail_energy_above_n_over_4',
        'k_dissipation', 'k_d_over_nyquist', 'budget_viscous_share',
        'budget_drag_share', 'budget_closure')} | {'ok': bool(res['ok'])})
    physics_gate(cfg, res['ok'],
                f"under-resolved: {res['tail_enstrophy_above_n_over_4']:.3f} of the "
                f"Galerkin enstrophy lies above k = n/4 (tol {cfg['tol_tail_enstrophy']})")

    
    # 5. SURROGATE CALIBRATION & FORWARD-PROCESS SCALING
    
    if V: print("\n[5/10] Surrogate vs empirical conditional variance; forward process")
    idx_tr, idx_val = split_by_trajectory(ds['traj_id'], cfg['val_split'], cfg['seed'])
    diffusion = WaveletDiffusionModel(o, proj, beta=cfg['beta'], lambda0=cfg['lambda0'])
    diffusion.calibrate(ds, idx_tr, dequant=dequant, seed=cfg['seed'])
    rho_tr = np.asarray(ds['rho'][idx_tr])
    diffusion.set_time_range(sigma_bar_z, np.percentile(rho_tr, 0.5, axis=0),
                            np.percentile(rho_tr, 99.5, axis=0),
                            cfg['snr_hi'], cfg['snr_lo'])
    w2s, _ = make_wavelet_maps(o, dtype=np.float32)
    sigma_bar_r_orb = orbit_pool(sigma_bar_r[None, :], orbit).ravel()
    lev_orb = orbit_pool(diffusion.level_per_coeff[None, :].astype(float), orbit).ravel()
    val_traj = rng.choice(np.unique(ds['traj_id'][idx_val]),
                        size=min(3, len(np.unique(ds['traj_id'][idx_val]))), replace=False)
    tlm = cfg['sigma_mode'] != 'B0'
    records, errs, tests, floors = [], {'g=1': [], 'g=rho(x)': []}, [], []
    for tj in val_traj:
        p = int(rng.choice(idx_val[ds['traj_id'][idx_val] == tj]))
        xw = jnp.asarray(np.asarray(ds['x'][ds['pair_index'][p, 0]], np.float32))
        ys = solver_conditional_ensemble(forced_step, stream, *w2s(xw), stride,
                                        n_rep=cfg['sigma_validation_reps'], seed=p)
        emp = orbit_pool(ys.var(0, ddof=1)[None, :], orbit).ravel()
        rho_p = np.asarray(ds['rho'][p])
        errs['g=1'].append(float(np.median(np.abs(np.log(emp / sigma_bar_r_orb)))))
        errs['g=rho(x)'].append(float(np.median(np.abs(np.log(emp / (sigma_bar_r_orb * rho_p))))))
        records.append({'emp': emp, 'sb': sigma_bar_r_orb,
                        'sbx': sigma_bar_r_orb * rho_p if tlm else None})
        # Floor: bootstrap over ensemble MEMBERS (iid), never over the
        # coefficients of an orbit (CERT.orbit_mc_floor).
        fl = orbit_mc_floor(ys, orbit)
        floors.append(fl)
        # rho(x) is itself an n_probe-member estimate (fixed probes): its
        # sampling error adds in quadrature, floor_x = floor_K sqrt(1 + K/n_probe).
        infl = np.sqrt(1.0 + cfg['sigma_validation_reps'] / cfg['sigma_probes'])
        fl_x = {k: (v * infl if k in ('median', 'median_p95') else v) for k, v in fl.items()}
        tests.append({'g=1': variance_test(emp, sigma_bar_r_orb, fl, label=f"g = 1, pair {p}"),
                    'g=rho(x)': variance_test(emp, sigma_bar_r_orb * rho_p, fl_x,
                                                label=f"g = rho(x), pair {p}")})

    metrics.put('surrogate', dict(
        sigma_err={k: float(np.mean(v)) for k, v in errs.items()},
        mc_floor=float(np.mean([f['median'] for f in floors])),
        mc_floor_p95=float(np.max([f['median_p95'] for f in floors])),
        mc_floor_naive=float(np.mean([f['naive_independent_floor'] for f in floors])),
        mc_floor_inflation=float(np.mean([f['inflation'] for f in floors])),
        significant={k: bool(any(t[k]['significant'] for t in tests)) for k in errs},
        rho_range=[float(rho_tr.min()), float(rho_tr.max())]))
    plot_sigma_calibration(records, lev_orb, save_path=fig('sigma_calibration.png'))

    rho_fn = None
    if tlm:     # same probes and floor as the dataset's rho
        ratio = make_sigma_ratio_fn(det_step, noise, stream, sigma_bar_r, orbit, dt, stride,
                                    n_probe=cfg['sigma_probes'])
        rho_fn = jax.jit(lambda x: jax.vmap(ratio)(*w2s(x)))

    
    # 6. TRAINING
    
    if V: print("\n[6/10] Training the residual score model")
    rung = 'L1' if cfg['sigma_mode'] == 'B0' else 'L1.5'
    ckpt = os.path.join(cfg['checkpoint_dir'],
                        f"emulator_J{J}_jmin{jmin}_{cfg['sigma_mode']}_stream.pkl")
    tr = train_conditional_score_model(
        ds, diffusion, sigma_bar_z, idx_tr, idx_val, n_epochs=cfg['n_epochs'],
        batch_size=cfg['batch_size'], lr=cfg['lr'], seed=cfg['seed'],
        channels=cfg['channels'], weighting=cfg['weighting'], snr_cap=cfg['snr_cap'],
        ema_decay=cfg['ema_decay'], patience=cfg['patience'], checkpoint_path=ckpt,
        dequant=dequant)
    params, apply_F = tr['params'], tr['apply_F']
    # Paired, empirical floor: fitted on the training pairs, scored on the
    # validation pairs with the same (tau, eps, dequantiser) draws as F = 0
    # and the network, in the network's own loss (GROUP.empirical_linear_floor).
    floor = empirical_linear_floor(ds, diffusion, sigma_bar_z, idx_tr, idx_val,
                                dequant=dequant, apply_F=apply_F, params=params,
                                n_tau=cfg['floor_n_tau'], seed=cfg['seed'], verbose=V)
    verdict = gain_verdict(floor['excess'], floor['excess_se'])
    if V:
        print(f"    gain over F=0: linear {floor['gain_lin']:+.5f} ± {floor['gain_lin_se']:.5f}, "
            f"network {floor['gain_net']:+.5f};  network - linear = {floor['excess']:+.5f} "
            f"({verdict['z']:+.1f} SE): {verdict['verdict']}")
    lv = {j: (d['F0'], d['net']) for j, d in floor['per_level'].items()}
    metrics.put('training', dict(rung=rung, best_val=tr['best_val'], val_base=tr['val_base'],
                                gain=tr['best_gain'], gain_se=tr['best_gain_se'],
                                floor=dict(loss=floor['loss'], xmse=floor['xmse'],
                                            gain_lin=floor['gain_lin'],
                                            gain_lin_se=floor['gain_lin_se'],
                                            excess=floor['excess'],
                                            excess_se=floor['excess_se'],
                                            per_level=floor['per_level']),
                                verdict=verdict))
    plot_training(tr['history'], lv,
                {'gain': floor['gain_lin'],
                'levels': {j: (d['F0'], d['lin']) for j, d in floor['per_level'].items()}},
                tr['best_gain_se'], save_path=fig('training.png'),
                val_base=floor['loss']['F0'],
                level_ylabel=r"F-loss per level ($\tau$-averaged)",
                floor_label="empirical linear floor (paired)")
    generate = make_conditional_sampler(diffusion, apply_F, sigma_bar_z)
    phi = make_phi_delta(o, det_step, stride)

    
    # 7. CERTIFICATES
    
    if V: print("\n[7/10] Certificates")
    m0 = jnp.zeros((diffusion.n_state,), jnp.float32)
    ng = cfg['n_gen_steps']
    gt = gaussian_sampler_test(
        lambda k, B: generate(None, k, jnp.broadcast_to(m0, (B, m0.size)), m0, n_steps=ng)["z"],
        cfg['gauss_test_samples'], sigma_bar_z, orbit, seed=cfg['seed'])
    gt_phys = gaussian_sampler_test(
        lambda k, B: diffusion.to_psi(generate(None, k, jnp.broadcast_to(m0, (B, m0.size)), m0,
                                            n_steps=ng)["y"]),
        cfg['gauss_test_samples'], sigma_bar_r, orbit, seed=cfg['seed'])

            
    if V:
        print(f"    Gaussian sampler (F=0), z: max |Var/Sigma_bar_z - 1| = "
            f"{gt['max_abs_dev']:.3f} (tol {gt['tolerance']:.3f}) -> "
            f"{'OK' if gt['ok'] else 'FAIL'}")
        print(f"    F=0 sampler, physical S y vs diag Sigma_lin: max |ratio - 1| = "
            f"{gt_phys['max_abs_dev']:.3f}  (diagnostic: diag(Pi D Pi^T) != diag "
            f"Sigma_lin in general; noness claim)")
    if not gt['ok']:
        warnings.warn("F = 0 sampler does not reproduce N(0, Sigma_bar_z): the reverse "
                    "OU bridge, the prior or the dequantiser is wrong")

    # conditional law at a FIXED x: solver ensemble vs emulator ensemble
    p0 = int(rng.choice(idx_val))
    x0c = np.asarray(ds['x'][ds['pair_index'][p0, 0]], np.float32)
    m_x0 = np.asarray(phi(x0c))
    rho_x0 = None if rho_fn is None else rho_fn(jnp.asarray(x0c)[None])[0]
    K = cfg['sigma_validation_reps']
    ys = solver_conditional_ensemble(forced_step, stream, *w2s(jnp.asarray(x0c)), stride,
                                    n_rep=K, seed=cfg['seed'] + 3)
    gen_psi = []
    def _psi_draws(prm, seed0):
        out = []
        for i in range(0, K, cfg['compare_chunk']):
            B = min(cfg['compare_chunk'], K - i)
            o_ = generate(prm, random.PRNGKey(seed0 + i),
                            np.broadcast_to(m_x0, (B, m_x0.size)), x0c, rho_x0,
                            n_steps=ng, guidance=cfg['guidance'])
            out.append(np.asarray(stream.psi_w_from_state(o_["y"]), np.float64))
        return np.concatenate(out)

    gen_psi = _psi_draws(params, 10_000)     # network
    f0_psi = _psi_draws(None, 20_000)        # F = 0 sampler, same x, same g
    # Same-x variance error of the F = 0 sampler: the network improves the
    # conditional variance iff it beats THIS number at THIS x.
    psi_m = np.asarray(stream.psi_w_from_state(jnp.asarray(m_x0)), np.float64)
    var_s = orbit_pool(ys.var(0, ddof=1)[None, :], orbit).ravel()
    var_g = orbit_pool(gen_psi.var(0, ddof=1)[None, :], orbit).ravel()
    lr_gen = np.log(var_g / var_s)
    var_F0 = orbit_pool(f0_psi.var(0, ddof=1)[None, :], orbit).ravel()
    lr_F0 = np.log(var_F0 / var_s)
    ms_sol = mean_shift_diagnostic(ys, psi_m, orbit, seed=cfg['seed'])
    ms_emu = mean_shift_diagnostic(gen_psi, psi_m, orbit, seed=cfg['seed'] + 1)
    bias = conditional_mean_bias(ys, gen_psi, psi_m, orbit, seed=cfg['seed'])
    scorr = shift_correlation(ys, gen_psi, psi_m, orbit, seed=cfg['seed'])
    wov = within_orbit_variance_structure(ys, gen_psi, orbit)
    e_sol = one_lag_energy_split(o, stream, m_x0, ys, psi_m)
    e_emu = one_lag_energy_split(o, stream, m_x0, gen_psi, psi_m)
    fl_s, fl_g = orbit_mc_floor(ys, orbit), orbit_mc_floor(gen_psi, orbit)
    cond = dict(median_abs_log_var_ratio=float(np.median(np.abs(lr_gen))),
                max_abs_log_var_ratio=float(np.max(np.abs(lr_gen))),
                median_abs_log_var_ratio_F0=float(np.median(np.abs(lr_F0))),
                mc_median_floor=float(np.hypot(fl_s['median'], fl_g['median'])),
                within_orbit_var_corr=wov['score'], within_orbit_var_ceiling=wov['ceiling'],
                chi_solver=ms_sol['chi'], chi_ng_solver=ms_sol['chi_ng'],
                shift_z_solver=ms_sol['z'], chi_emulator=ms_emu['chi'],
                chi_mc=ms_sol['chi_mc'],
                mean_bias_emulator=bias['bias'], mean_bias_z=bias['z'],
                mean_bias_F0=ms_sol['chi_ng'],
                shift_corr=scorr['corr'], shift_corr_null_sd=scorr['null_sd'],
                energy_split={'solver': e_sol, 'emulator': e_emu})
    if V:
        print(f"    conditional law at fixed x ({K} + {K} members): median |log Var ratio| "
            f"network {cond['median_abs_log_var_ratio']:.3f} vs F = 0 "
            f"{cond['median_abs_log_var_ratio_F0']:.3f} (MC floor {cond['mc_median_floor']:.3f})")
        print(f"    within-orbit log-variance structure: network {wov['score']:.3f} of a "
            f"reproducible ceiling {wov['ceiling']:.3f} (F = 0: 0 by construction)")
        print(f"    mean shift chi: solver {ms_sol['chi']:.3f} (beyond MC {ms_sol['chi_ng']:.3f}, "
            f"z = {ms_sol['z']:+.1f}), emulator {ms_emu['chi']:.3f}, pure MC {ms_sol['chi_mc']:.3f}; "
            f"corr(solver, emulator) = {scorr['corr']:.3f} (sign-flip null sd "
            f"{scorr['null_sd']:.3f})")
        print(f"    conditional-mean error beyond MC: network {bias['bias']:.3f} "
            f"(z = {bias['z']:+.1f}) vs F = 0 {ms_sol['chi_ng']:.3f} "
            f"(z = {ms_sol['z']:+.1f})  (lower is better; z < 3 = not detected)")
        for k in ("cross", "self", "total"):
            z = (e_emu[k] - e_sol[k]) / max(np.hypot(e_emu[k + "_se"], e_sol[k + "_se"]), 1e-300)
            print(f"    one-lag energy E(y) - E(m), {k:>5s}: solver {e_sol[k]:+.3e}, "
                f"emulator {e_emu[k]:+.3e}  ({z:+.1f} SE)")
        if ms_sol['z'] < 3.0:
            print(f"    NOTE: solver mean shift not detected (z = {ms_sol['z']:+.1f} < 3): "
                "nothing beyond the Gaussian mean to learn at this Delta and K")
    metrics.put('certificates', dict(gauss_z=dict(max_dev=gt['max_abs_dev'], tol=gt['tolerance'], ok=gt['ok']),
                                    gauss_phys=dict(max_dev=gt_phys['max_abs_dev'], tol=gt_phys['tolerance'],
                                                    ok=gt_phys['ok']),
                                    conditional=cond))

    
    # 8. ENSEMBLE EMULATION vs SOLVER
    
    if V: print("\n[8/10] Ensemble emulation vs forced solver")
    EZ, vort = make_observables(o, stream)
    p1 = int(rng.choice(idx_val))
    x0 = np.asarray(ds['x'][ds['pair_index'][p1, 0]], np.float32)
    n_lags = max(1, int(round(cfg['compare_T'] / Delta)))
    snap_lags = sorted(set(np.linspace(0, n_lags, cfg['compare_n_snap']).round().astype(int)))
    Kc = cfg['compare_members']
    if V:
        print(f"    x0 = validation pair {p1}; {Kc} members each; {n_lags} lags "
            f"(T = {n_lags * Delta:.3g} = {mu_d * n_lags * Delta:.2f} drag times)")
    sol = solver_ensemble(o, forced_step, stride, x0, Kc, n_lags, EZ=EZ, snap_lags=snap_lags,
                        seed=cfg['seed'] + 11, chunk=cfg['compare_chunk'])
    emu = emulator_ensemble(generate, params, phi, x0, Kc, n_lags, EZ=EZ, rho_fn=rho_fn,
                            snap_lags=snap_lags, seed=cfg['seed'] + 12, chunk=cfg['compare_chunk'],
                            sampler_kw=dict(n_steps=ng, eta=cfg['eta'], guidance=cfg['guidance']))
    val_rows = np.unique(ds['pair_index'][idx_val, 1])
    ref_rows = np.sort(rng.choice(val_rows, size=min(64, len(val_rows)), replace=False))
    cmp = compare_ensembles(o, stream, sol, emu, Delta, nu, mu_d, spec.eps_inj,
                            ref_states=np.asarray(ds['x'][ref_rows]), vort_fn=vort)
    print_comparison(cmp)
    if mu_d * n_lags * Delta < 5 and V:
        print("    NOTE: horizon < 5 drag times: the late window tests transient "
            "statistics, not the invariant measure")
    max_div = max(proj.normalised_residual(x) for x in emu['snaps'][max(emu['snaps'])])
    cmp['metrics']['max_div_residual_emulator'] = float(max_div)
    cmp['metrics']['clip_max'] = emu['clip_max']
    metrics.put('ensemble_comparison', cmp['metrics'])
    nshow = min(2, Kc)
    W_s = np.stack([np.asarray(vort(jnp.asarray(sol['snaps'][k][:nshow]))) for k in snap_lags], 1)
    W_e = np.stack([np.asarray(vort(jnp.asarray(emu['snaps'][k][:nshow]))) for k in snap_lags], 1)
    plot_vorticity_comparison(Delta * np.array(snap_lags), W_s, W_e,
                            save_path=fig('compare_vorticity.png'))
    plot_ensemble_statistics(cmp, save_path=fig('compare_statistics.png'),
                            E_stat=id_ref['E_mean'])

    
    # 9. CLOSED-FORM DIFFUSION BENCHMARK
    
    bench = None
    if cfg['run_bench_tg']:
        if V: print("\n[9/10] Closed-form linear benchmark of the emulator and the floor")
        bench = run_tg_diffusion_benchmark(o, proj, cfg, fig_dir=cfg['fig_dir'])
        metrics.put('bench_tg_diffusion', dict(
            ok=bench['ok'], ok_curves=bench['ok_curves'], ok_floor=bench['ok_floor'],
            ok_verdict=bench['ok_verdict'], z_floor=bench['z_floor'],
            verdict=bench['verdict'], threshold=bench['threshold'], gamma=bench['gamma'],
            oracle_gain=bench['oracle']['gain'], floor_gain=bench['floor']['gain_lin'],
            **{f"{k}_max_abs_zE": bench[k]['max_abs_zE'] for k in ('solver', 'gaussian_F0', 'emulator')},
            emulator_amp_z=bench['emulator']['z_amp_mean'],
            emulator_amp_var_ratio=bench['emulator']['amp_var_ratio'],
            comparison=bench['cmp']['metrics']))
    elif V:
        print("\n[9/10] Closed-form diffusion benchmark skipped (--bench-tg to run)")

    
    # 10. RUNG x LAG GRID
    
    grid = None
    if cfg.get('run_grid'):
        if V: print("\n[10/10] Rung x lag grid")
        grid = run_rung_lag_grid(o, proj, orbit, cfg,
                        main=dict(ds=ds, sigma_bar_z=sigma_bar_z, sigma_bar_r=sigma_bar_r),
                        fig_dir=cfg['fig_dir'])
        metrics.put('grid', dict(
            rungs=list(grid['rungs']), lags=list(grid['lags']),
            gain=grid['gain'], gain_se=grid['gain_se'],
            floor_gain=grid['floor_gain'], excess_z=grid['excess_z'],
            chi_solver=grid['chi_solver'], chi_ng_solver=grid['chi_ng_solver'],
            chi_emulator=grid['chi_emulator'], shift_corr=grid['shift_corr'],
            shift_corr_null_sd=grid['shift_corr_null_sd'],
            mean_bias_emulator=grid['mean_bias_emulator'], mean_bias_z=grid['mean_bias_z'],
            verdict={f"{r}@{s}": grid['cells'][(r, s)]['verdict_code']
                    for r in grid['rungs'] for s in grid['strides']},
            chi_x_ng_solver={str(s): grid['columns'][s]['shift_solver']['chi_x_ng']
                            for s in grid['strides']},
            z_x_solver={str(s): grid['columns'][s]['shift_solver']['z_x']
                        for s in grid['strides']},
            snr_within={str(s): grid['columns'][s]['snr_within']
                        for s in grid['strides']}))

    metrics.dump(fig('metrics.json'))
    return {'config': cfg, 'operators': (o, ops, proj), 'spec': spec, 'dataset': ds,
            'diffusion': diffusion, 'training': tr, 'comparison': cmp,
            'stochastic_tg': stg, 'bench_tg': bench, 'grid': grid, 'metrics': metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic wavelet emulator of 2-D Navier-Stokes")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run (J=5)")
    parser.add_argument("--only-validate", action="store_true")
    parser.add_argument("--bench-tg", action="store_true",
                        help="run the closed-form linear benchmark of the emulator")
    parser.add_argument("--grid", action="store_true",
                        help="run the rung x lag grid (len(grid_rungs) x len(grid_lags) "
                            "trainings; cost printed and gated before stage 1)")
    parser.add_argument("--grid-cost", action="store_true",
                        help="print the grid cost report and exit")
    overrides = (("J", int), ("jmin", int), ("nu", float), ("mu_d", float), ("eps_inj", float),
                ("alpha_F", float), ("stride", int), ("pair_gap", int), ("n_epochs", int),
                ("n_trajectories", int), ("sigma_mode", str), ("test_space", str),
                ("stoch_integrator", str), ("compare_T", float),
                ("compare_members", int), ("n_gen_steps", int),
                ("grid_sigma_probes", int), ("grid_n_trajectories", int),
                ("grid_n_pairs_per_traj", int), ("grid_max_cost_ratio", float))
    for k, t in overrides:
        parser.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t)
    args = parser.parse_args()
    if args.smoke:
        config.update(SMOKE)
    for k, _ in overrides:
        if getattr(args, k) is not None:
            config[k] = getattr(args, k)
    config['_only_validate'] = args.only_validate
    config['run_bench_tg'] = config['run_bench_tg'] or args.bench_tg
    config['run_grid'] = config.get('run_grid', False) or args.grid
    if args.grid_cost:
        grid_cost_report(config)
        raise SystemExit(0)
    if config['test_space'] != 'primal':
        warnings.warn("Petrov-Galerkin: advection is not energy neutral and the "
                    "closed-form benchmarks are not available")
    results = run_pipeline(config)
    if not args.only_validate:
        print(f"\nPipeline complete. Data: {config['data_path']}; checkpoints: "
            f"{config['checkpoint_dir']}; figures + metrics.json: {config['fig_dir']}")