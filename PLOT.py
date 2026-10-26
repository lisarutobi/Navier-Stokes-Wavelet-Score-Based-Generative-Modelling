"""
PLOT.py - Visualization and Analysis Module for Wavelet-Galerkin Navier-Stokes

This module provides:
1. Mathematical utilities for fluid dynamics (vorticity, energy, turnover time)
2. Visualization functions for:
   - Vorticity fields and their evolution
   - Energy and enstrophy budgets
   - Energy spectra (Fourier and wavelet)
   - Stochastic trajectory analysis
3. Stationary energy identity of the FORCED system:
   2 mu_d <E> + 2 nu <Z_gal> = eps_inj
4. Research figures of the stochastic pipeline:
   dataset overview, Malliavin calibration, training, stochastic TG
   (solver), solver-vs-emulator ensemble comparison, closed-form diffusion
   benchmark.

All functions are designed to work with:
- Stochastic trajectories (from TG.generate_transition_pair_dataset)
- Emulated trajectories (from WSBG rollouts)

Mathematical Framework:
-----------------------
1. Vorticity: ω = ∂ₓv - ∂ᵧu (computed via central differences)
2. Eddy Turnover Time: τ = √2 / ω_rms (visualisation-grade)
3. Energy Spectrum: E(k) = 0.5 ∫_{|k|=k₀} |û(k)|² + |v̂(k)|² dΩ (isotropic average)
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='pywt')


# RESEARCH FIGURES (stochastic pipeline)

def _save(fig, save_path, dpi=200):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {save_path}")


def _band(ax, t, X, color, label, alpha=0.2):
    m = X.mean(1)
    s = X.std(1, ddof=1) if X.shape[1] > 1 else np.zeros_like(m)
    ax.plot(t, m, color=color, lw=1.8, label=label)
    ax.fill_between(t, m - s, m + s, color=color, alpha=alpha, lw=0)


def plot_dataset_overview(ds, nu, mu_d, eps_inj, save_path="dataset_overview.png",
                          n_traj_plot=12):
    """(a) energy: burn-in + retained, ensemble mean +- sd and a few members;
    (b) per-trajectory time mean of 2 mu <E> + 2 nu <Z> vs eps_inj;
    (c) running mean of the identity; (d) distribution of retained E."""
    Delta = float(ds["Delta"])
    E, Z = np.asarray(ds["E_path"]), np.asarray(ds["Z_path"])
    EB = np.asarray(ds["E_burn"])
    nb = EB.shape[1]
    tb = Delta * np.arange(1, nb + 1)
    tk = Delta * (nb + np.arange(E.shape[1]))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    _band(a, tb, EB.T, "0.5", "burn-in (mean ± sd)")
    _band(a, tk, E.T, "C0", "retained (mean ± sd)")
    for i in range(min(n_traj_plot, E.shape[0])):
        a.plot(tk, E[i], color="C0", lw=0.5, alpha=0.4)
    a.axhline(eps_inj / (2 * mu_d), color="r", ls=":", label=r"$\varepsilon/(2\mu_d)$ (drag-only bound)")
    a.axvline(nb * Delta, color="k", ls="--", lw=1)
    a.set_xlabel("t"); a.set_ylabel("E"); a.set_title("(a) kinetic energy"); a.legend(fontsize=8)
    lhs = 2 * mu_d * E + 2 * nu * Z
    pt = lhs.mean(1)
    se = pt.std(ddof=1) / np.sqrt(len(pt)) if len(pt) > 1 else np.nan
    a = ax[0, 1]
    a.plot(pt / eps_inj, "o", ms=3, alpha=0.7)
    a.axhline(1.0, color="r", ls="--", label=r"$\varepsilon_{inj}$")
    a.axhline(pt.mean() / eps_inj, color="C0",
              label=f"ensemble {pt.mean()/eps_inj:.4f} ± {se/eps_inj:.4f}")
    a.set_xlabel("trajectory"); a.set_ylabel(r"$(2\mu\langle E\rangle+2\nu\langle Z\rangle)/\varepsilon$")
    a.set_title("(b) stationary identity per trajectory"); a.legend(fontsize=8)
    a = ax[1, 0]
    run = np.cumsum(lhs, 1) / np.arange(1, lhs.shape[1] + 1)
    t = Delta * np.arange(1, lhs.shape[1] + 1)
    _band(a, t, run.T / eps_inj, "C0", "running mean (mean ± sd over trajectories)")
    a.axhline(1.0, color="r", ls="--")
    a.set_xlabel("time in retained window"); a.set_title("(c) convergence of the identity")
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.hist(E.ravel(), bins=60, density=True, alpha=0.7, color="C0")
    a.axvline(E.mean(), color="k", ls="--", label=f"<E> = {E.mean():.4e}")
    a.set_xlabel("E"); a.set_title("(d) stationary energy distribution"); a.legend(fontsize=8)
    fig.suptitle(f"Transition dataset: {E.shape[0]} trajectories, Δ = {Delta:g}, "
                 f"ν = {nu:g}, μ_d = {mu_d:g}")
    fig.tight_layout()
    _save(fig, save_path)


def plot_sigma_calibration(records, levels_of_orbit, save_path="sigma_calibration.png"):
    """Empirical conditional variance per orbit vs the surrogate (B0 and x-dependent).

    records: list of dicts {"emp": (n_orb,), "sb": (n_orb,), "sbx": (n_orb,) or None}
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for a, key, title in ((ax[0], "sb", r"$\bar\Sigma$ (B0, x-independent)"),
                          (ax[1], "sbx", r"$\bar\Sigma(x)$ (TLM)")):
        for rec in records:
            if rec.get(key) is None:
                continue
            sc = a.scatter(rec[key], rec["emp"] / rec[key], c=levels_of_orbit,
                           cmap="viridis", s=8, alpha=0.7)
        a.set_xscale("log")
        a.axhline(1.0, color="r", ls="--")
        a.set_xlabel(title)
        a.set_title(f"Var(r|x)_emp / {title}")
    ax[0].set_ylabel("ratio (per dyadic orbit)")
    if "sc" in locals():
        fig.colorbar(sc, ax=ax, label="level j")
    _save(fig, save_path)


def plot_training(history, levels, oracle=None, best_gain_se=None,
                  save_path="training.png", val_base=None,
                  level_ylabel=r"MSE at SNR$_j$ = 1", floor_label="linear-Gaussian oracle"):
    history = dict(history)
    if val_base is not None:
        history["val_base"] = val_base
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    a = ax[0]
    ep = np.arange(1, len(history["train"]) + 1)
    a.plot(ep, history["train"], label="train")
    a.plot(ep, history["val"], label="val (EMA)")
    base = history.get("val_base", 1.0)
    a.axhline(base, color="k", ls=":", label=f"F = 0 on val ({base:.4f})")
    if oracle is not None:
        a.axhline(base - oracle["gain"], color="r", ls="--",
                  label=f"F=0 minus {floor_label} gain ({oracle['gain']:.1e})")
    if best_gain_se is not None and np.isfinite(best_gain_se):
        v = min(history["val"])
        a.axhspan(v - 2 * best_gain_se, v + 2 * best_gain_se, color="C1", alpha=0.15,
                  label="best val ± 2 SE")
    a.set_xlabel("epoch"); a.set_ylabel("F-loss"); a.set_title("training"); a.legend(fontsize=8)
    a = ax[1]
    js = sorted(levels)
    base = [levels[j][0] for j in js]
    net = [levels[j][1] for j in js]
    w = 0.27
    x = np.arange(len(js))
    a.bar(x - w, base, w, label="F = 0")
    a.bar(x, net, w, label="network")
    if oracle is not None:
        a.bar(x + w, [oracle["levels"][j][1] for j in js], w, label=floor_label)
    a.set_xticks(x); a.set_xticklabels([f"j={j}" for j in js])
    a.set_ylabel(level_ylabel); a.set_title("per-level denoising error")
    a.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


def plot_stochastic_tg_solver(res, save_path="validation_stochastic_tg.png"):
    t = res["t"]
    A = res["A"]
    K = res["K"]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    a = ax[0]
    for i in range(min(K, 12)):
        a.plot(t, A[:, i, 0], color="C0", lw=0.5, alpha=0.5)
    a.plot(t, A[:, :, 0].mean(1), "C0", lw=2, label="mean (solver)")
    m, sd = res["mean_th"][:, 0], np.sqrt(res["var_th"][:, 0])
    a.plot(t, m, "k--", label=" mean")
    a.fill_between(t, m - sd, m + sd, color="k", alpha=0.1, label=" ± sd")
    a.set_title("TG amplitude $a_1(t)$"); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[1]
    for i in range(4):
        a.plot(t[1:], A[1:, :, i].var(1, ddof=1) / res["var_th"][1:, i], "o-", ms=3, label=f"mode {i+1}")
    band = 3 * np.sqrt(2 / (K - 1))
    a.axhspan(1 - band, 1 + band, color="k", alpha=0.1, label="±3σ (MC)")
    a.set_title("Var(a_i) / "); a.set_xlabel("t"); a.legend(fontsize=7)
    a = ax[2]
    _band(a, t, res["E"], "C0", "solver E (mean ± sd)")
    a.plot(t, res["E_th"], "k--", label=" E[E(t)]")
    a.set_title("kinetic energy"); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[3]
    a.semilogy(t, np.maximum(res["off_shell_t"], 1e-18), "C3o-", ms=3)
    a.set_title("energy fraction off the TG shell"); a.set_xlabel("t")
    fig.suptitle(f"Stochastic Taylor-Green (nonlinear forced solver): γ_h = {res['gamma']:.4f}, "
                 f"K = {K}, {res['dtype']};  max z(mean) {res['z_mean_max']:.2f}, "
                 f"max z(var) {res['var_dev_max']:.2f}, KS p {res['ks_pvalue']:.2f}")
    fig.tight_layout()
    _save(fig, save_path)


def plot_vorticity_comparison(times, W_solver, W_emulator, save_path="compare_vorticity.png",
                              W_ref=None):
    """Rows: solver members, emulator members (same x0, independent noise).
    W_* : (n_members_shown, n_times, n, n)."""
    rows = [("solver", w) for w in W_solver] + [("emulator", w) for w in W_emulator]
    nt = len(times)
    fig, ax = plt.subplots(len(rows), nt, figsize=(2.6 * nt, 2.6 * len(rows)), squeeze=False)
    vmax = np.percentile(np.abs(np.concatenate([np.asarray(W_solver).ravel(),
                                                np.asarray(W_emulator).ravel()])), 99.5)
    for r, (name, W) in enumerate(rows):
        for c in range(nt):
            im = ax[r, c].imshow(W[c].T, origin="lower", cmap="RdBu_r", vmin=-vmax,
                                 vmax=vmax, extent=(0, 1, 0, 1))
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"t = {times[c]:.3g}", fontsize=10)
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="vorticity")
    fig.suptitle("Vorticity from the same burned-in state: forced solver vs diffusion emulator")
    _save(fig, save_path)


def plot_field_evolution(times, rows, *, reference=None, show_diff=True,
                         field_label=r"$\omega$", cmap="RdBu_r", clip_pct=99.5,
                         diff_shared_scale=True, scalars=None, scalars_t=None,
                         scalar_label=None, scalar_log=False, title=None,
                         save_path="field_evolution.png"):
    r"""Rows = method, columns = time: the field comparison panel.

    ``times``   (nt,) physical times, one per column.
    ``rows``    ordered [(label, W), ...] with W (nt, n, n) REAL fields in the
                SAME variable and the SAME units (e.g. all vorticity from
                StreamMap.vorticity_s -- never mix estimators across rows, see
                the module note on vorticity conventions).
    ``reference`` label of one row.  Every other row then gets a second row
                ``label - reference``.  With ``diff_shared_scale`` (default)
                all difference rows share ONE scale, so "method A is ten times
                closer than method B" is visible at a glance; set it False to
                give each difference row its own scale when you only care about
                the spatial STRUCTURE of each error.
    ``scalars`` optional {label: (ns,)} drawn in a panel under the grid
                (amplitude, energy, relative error, ...).  Labels that match a
                row label reuse that row's colour.  ``scalars_t`` is their time
                axis: pass the FULL lag grid here when the field columns have
                been subsampled, so the curves keep every point while the grid
                stays readable.  Defaults to ``times``.

    Design: ONE symmetric colour scale for all field rows (so panels are
    comparable by eye, which a per-panel scale destroys), a separate one per
    difference row (so errors are readable), and a shared time axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t = np.asarray(times, float)
    nt = len(t)
    rows = [(str(k), np.asarray(W, float)) for k, W in rows]
    for k, W in rows:
        if W.shape[0] != nt:
            raise ValueError(f"row {k!r} has {W.shape[0]} snapshots, expected {nt}")

    ref_W = None
    if reference is not None:
        match = [W for k, W in rows if k == reference]
        if not match:
            raise ValueError(f"reference {reference!r} is not one of "
                             f"{[k for k, _ in rows]}")
        ref_W = match[0]

    panels = list(rows)
    if ref_W is not None and show_diff:
        panels += [(f"{k} - {reference}", W - ref_W)
                   for k, W in rows if k != reference]
    n_field = len(rows)

    # one scale for the fields; one shared (or per-row) scale for differences
    vmax = np.percentile(np.abs(np.concatenate([W.ravel() for _, W in rows])),
                         clip_pct)
    vmax = float(vmax) if vmax > 0 else 1.0
    vdiff = None
    if len(panels) > n_field:
        vd = np.percentile(np.abs(np.concatenate(
            [W.ravel() for _, W in panels[n_field:]])), clip_pct)
        vdiff = float(vd) if vd > 0 else 1.0

    n_sc = 1 if scalars else 0
    h = 2.5 * len(panels) + 2.6 * n_sc + 0.7
    fig = plt.figure(figsize=(2.5 * nt + 1.6, h))
    gs = GridSpec(len(panels) + n_sc, nt, figure=fig,
                  height_ratios=[1] * len(panels) + [1.05] * n_sc,
                  hspace=0.12, wspace=0.04)

    im_f = im_d = None
    for r, (label, W) in enumerate(panels):
        is_diff = ref_W is not None and show_diff and r >= n_field
        if is_diff:
            if diff_shared_scale:
                v = vdiff
            else:
                v = float(np.percentile(np.abs(W), clip_pct)) or 1.0
                label = f"{label}\n" + r"$\pm$" + f"{v:.2e}"
            label = (f"{label}\nmax |.| = {np.abs(W).max():.2e}"
                     if diff_shared_scale else label)
        else:
            v = vmax
        for c in range(nt):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(W[c].T, origin="lower", cmap=cmap, vmin=-v, vmax=v,
                           extent=(0, 1, 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if is_diff:
                im_d = im
            else:
                im_f = im
            if r == 0:
                ax.set_title(f"$t$ = {t[c]:.4g}", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            for sp in ax.spines.values():
                sp.set_color("0.5" if is_diff else "0.2")
                sp.set_linewidth(1.4 if is_diff else 0.8)

    if scalars:
        ax = fig.add_subplot(gs[len(panels), :])
        ts = t if scalars_t is None else np.asarray(scalars_t, float)
        colours = {k: f"C{i}" for i, (k, _) in enumerate(rows)}
        for i, (k, y) in enumerate(scalars.items()):
            y = np.asarray(y, float)
            if y.shape[0] != ts.shape[0]:
                raise ValueError(f"scalar {k!r} has {y.shape[0]} points, "
                                 f"expected {ts.shape[0]} (see scalars_t)")
            ax.plot(ts, y, "o-", ms=3,
                    color=colours.get(k, f"C{len(rows) + i}"), label=k)
        for tc in t:
            ax.axvline(tc, color="0.85", lw=0.8, zorder=0)
        if scalar_log:
            ax.set_yscale("log")
        ax.set_xlabel("$t$")
        if scalar_label:
            ax.set_ylabel(scalar_label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=min(4, max(1, len(scalars))))

    axes_f = [a for a in fig.axes if a.images and a.images[0] is not None]
    if im_f is not None:
        fig.colorbar(im_f, ax=axes_f[:nt * n_field], fraction=0.015, pad=0.01,
                     label=field_label)
    if im_d is not None:
        fig.colorbar(im_d, ax=axes_f[nt * n_field:], fraction=0.015, pad=0.01,
                     label=("difference (shared scale)" if diff_shared_scale
                            else "difference (per-row scale)"))
    if title:
        fig.suptitle(title, y=0.995)
    _save(fig, save_path)


def plot_ensemble_statistics(cmp, save_path="compare_statistics.png", E_stat=None):
    t = cmp["t"]
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    sol, emu = cmp["_sol"], cmp["_emu"]
    a = ax[0, 0]
    _band(a, t, sol["E"], "C0", "solver")
    _band(a, t, emu["E"], "C1", "emulator")
    if E_stat is not None:
        a.axhline(E_stat, color="k", ls=":", label="dataset <E>")
    a.set_title("E(t): mean ± sd"); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[0, 1]
    _band(a, t, sol["Z"], "C0", "solver")
    _band(a, t, emu["Z"], "C1", "emulator")
    a.set_title("Z(t) (Galerkin enstrophy)"); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[0, 2]
    a.plot(t, cmp["zE"], label="z(<E>)")
    a.plot(t, cmp["zZ"], label="z(<Z>)")
    a.axhspan(-3, 3, color="k", alpha=0.08)
    a.set_title("standardised mean differences"); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[1, 0]
    a.semilogy(t[1:], sol["spread"][1:], "C0", label="solver")
    a.semilogy(t[1:], emu["spread"][1:], "C1", label="emulator")
    a.set_title(r"ensemble spread $\frac{1}{2}\langle\|u_i-\bar u\|^2\rangle$"); a.set_xlabel("t")
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(t, cmp["acc"], "k", label="ACC of ensemble means")
    a.plot(t[1:], cmp["spread_ratio"], "C3", label="spread ratio emu/solver")
    a.axhline(1.0, color="k", ls=":")
    a.set_ylim(0, 1.6); a.set_title("predictable part and dispersion"); a.set_xlabel("t")
    a.legend(fontsize=8)
    snaps = cmp["snapshots"]
    last = max(snaps)
    r = snaps[last]
    a = ax[1, 2]
    for k_, ls in ((min(snaps), ":"), (last, "-")):
        rr = snaps[k_]
        a.loglog(rr["k"], rr["spec_solver"].mean(0), "C0", ls=ls, label=f"solver t={rr['t']:.2g}")
        a.loglog(rr["k"], rr["spec_emulator"].mean(0), "C1", ls=ls, label=f"emulator t={rr['t']:.2g}")
    if cmp.get("ref_spectrum") is not None:
        a.loglog(r["k"], cmp["ref_spectrum"], "k--", lw=1, label="dataset (stationary)")
    a.set_title("isotropic spectrum (ensemble mean)"); a.set_xlabel("k"); a.legend(fontsize=7)
    js = sorted(r["levels_solver"])
    a = ax[2, 0]
    a.semilogy(js, [r["levels_solver"][j]["rms"] for j in js], "s-", label="solver")
    a.semilogy(js, [r["levels_emulator"][j]["rms"] for j in js], "o--", label="emulator")
    if cmp.get("ref_levels"):
        a.semilogy(js, [cmp["ref_levels"][j]["rms"] for j in js], "k:", label="dataset")
    a.set_title(f"level RMS (t = {r['t']:.3g})"); a.set_xlabel("level j"); a.legend(fontsize=8)
    a = ax[2, 1]
    a.plot(js, [r["levels_solver"][j]["kurtosis"] for j in js], "s-", label="solver")
    a.plot(js, [r["levels_emulator"][j]["kurtosis"] for j in js], "o--", label="emulator")
    a.axhline(0, color="k", ls=":")
    a.set_title("excess kurtosis (orbit-standardised)"); a.set_xlabel("level j"); a.legend(fontsize=8)
    a = ax[2, 2]
    if "vort_solver" in r:
        sd = r["vort_solver"].std()
        bins = np.linspace(-6, 6, 81)
        for lab, w, c in (("solver", r["vort_solver"], "C0"), ("emulator", r["vort_emulator"], "C1")):
            h, _ = np.histogram(w / sd, bins=bins, density=True)
            a.semilogy(0.5 * (bins[1:] + bins[:-1]), np.maximum(h, 1e-6), c, label=lab)
        if cmp.get("ref_vort") is not None:
            h, _ = np.histogram(cmp["ref_vort"] / sd, bins=bins, density=True)
            a.semilogy(0.5 * (bins[1:] + bins[:-1]), np.maximum(h, 1e-6), "k:", label="dataset")
        a.set_title(f"vorticity PDF (W1/sd = {r['vort_W1']:.3f})"); a.set_xlabel(r"$\omega/\sigma_\omega$")
        a.legend(fontsize=8)
    m = cmp["metrics"]
    fig.suptitle("Solver vs emulator from the same burned-in state "
                 f"(K = {cmp['K_solver']}/{cmp['K_emulator']}): late <E> err {m['rel_err_meanE_late']:.3f}, "
                 f"spread ratio {m['spread_ratio_median']:.2f}, spectrum log-RMSE "
                 f"{m.get('spec_log_rmse_final', float('nan')):.3f}")
    fig.tight_layout()
    _save(fig, save_path)


def plot_tg_diffusion_benchmark(res, save_path="bench_tg_diffusion.png"):
    t = res["t"]
    K = res["K"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    a = ax[0, 0]
    for name, c in (("solver", "C0"), ("gaussian_F0", "C2"), ("emulator", "C1")):
        _band(a, t, res[name]["E"], c, name, alpha=0.12)
    a.plot(t, res["E_th"], "k--", lw=2, label=" E[E(t)]")
    a.axhline(res["E_inf"], color="k", ls=":", lw=1)
    a.set_title("energy:  vs ensembles"); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[0, 1]
    for name, c, mk in (("solver", "C0", "s"), ("gaussian_F0", "C2", "^"), ("emulator", "C1", "o")):
        zE = (res[name]["E"].mean(1) - res["E_th"])[1:] / (res[name]["E"].std(1, ddof=1)[1:] / np.sqrt(K))
        a.plot(t[1:], zE, c, lw=1, label=name)
    a.axhspan(-res["threshold"], res["threshold"], color="k", alpha=0.08)
    a.set_title("z-score of <E>(t) vs "); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[0, 2]
    m, sd = res["a_mean_th"], np.sqrt(res["a_var_th"])
    a.plot(t, m, "k--", label=" mean")
    a.fill_between(t, m - sd, m + sd, color="k", alpha=0.1, label=" ± sd")
    for name, c, mk in (("solver", "C0", "s"), ("emulator", "C1", "o")):
        for k, v in res[name]["amp"].items():
            a.plot(np.full(len(v), t[k]) + (0.01 if name == "emulator" else -0.01) * t[-1],
                   v, mk, color=c, ms=3, alpha=0.4, label=name if k == max(res[name]["amp"]) else None)
    a.set_title("single-mode TG amplitude a(t) = <u, e_TG>"); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[1, 0]
    for name, c, mk in (("solver", "C0", "s"), ("gaussian_F0", "C2", "^"), ("emulator", "C1", "o")):
        rv = res[name]["amp_var_ratio"]
        a.plot([t[k] for k in rv], list(rv.values()), mk + "-", color=c, label=name)
    a.axhspan(1 - res["var_ratio_band"], 1 + res["var_ratio_band"], color="k", alpha=0.08)
    a.set_title("Var a(t) / "); a.set_xlabel("t"); a.legend(fontsize=8)
    a = ax[1, 1]
    h = res["history"]
    a.plot(h["train"], label="train"); a.plot(h["val"], label="val")
    a.axhline(res["val_base"], color="k", ls=":", label="F = 0 (val)")
    if res.get("oracle") is not None:
        a.axhline(res["val_base"] - res["oracle"]["gain"], color="r", ls="--",
                  label=" linear-Gaussian optimum")
    a.set_title("training (linear system)"); a.set_xlabel("epoch"); a.legend(fontsize=8)
    a = ax[1, 2]
    snaps = res["cmp"]["snapshots"]
    r = snaps[max(snaps)]
    a.loglog(r["k"], r["spec_solver"].mean(0), "C0", label="solver")
    a.loglog(r["k"], r["spec_emulator"].mean(0), "C1--", label="emulator")
    a.set_title(f"spectrum at t = {r['t']:.3g}"); a.set_xlabel("k"); a.legend(fontsize=8)
    fig.suptitle(f"Closed-form stochastic TG benchmark of the diffusion emulator "
                 f"(Stokes + drag + forcing, γ = {res['gamma']:.3f}, K = {K}) -> "
                 f"{'PASS' if res['ok'] else 'FAIL'}")
    fig.tight_layout()
    _save(fig, save_path)


def plot_rung_lag_grid(grid, save_path="grid.png"):
    """Two heatmaps: excess over the linear-Gaussian floor (in SE), and the
    emulator's reproduction of the measured mean shift."""
    import matplotlib.pyplot as plt
    rungs, lags = grid["rungs"], grid["lags"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a, key, title, cmap in (
            (ax[0], "excess_z", "network - empirical linear floor  [paired SE]", "RdBu_r"),
            (ax[1], "shift_corr", r"corr(shift$_{\rm solver}$, shift$_{\rm emu}$)", "viridis")):
        Z = np.asarray(grid[key], float)
        v = (float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0) \
            if key == "excess_z" else None
        im = a.imshow(Z, cmap=cmap, aspect="auto",
                      vmin=-v if v else 0.0, vmax=v if v else 1.0)
        a.set_xticks(range(len(lags)), [f"{L:g}" for L in lags])
        a.set_yticks(range(len(rungs)), rungs)
        a.set_xlabel(r"$\Delta$"); a.set_title(title, fontsize=10)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    a.text(j, i, f"{Z[i, j]:+.1f}" if key == "excess_z"
                           else f"{Z[i, j]:.2f}", ha="center", va="center",
                           fontsize=9, color="k")
        fig.colorbar(im, ax=a)
    chi = np.asarray(grid["chi_ng_solver"], float)[0]
    ax[0].set_ylabel("baseline rung")
    fig.suptitle(r"solver $\chi_{\rm NG}$ per column: "
                 + ", ".join(f"{c:.3f}" for c in chi), fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  → Saved: {save_path}")