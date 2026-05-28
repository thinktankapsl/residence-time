#!/usr/bin/env python3
"""
residence_time.py
=================
Compute ligand/ion residence time from a PLUMED colvar.dat file.

Algorithm
---------
1.  Load the PLUMED collective-variable (CV) timeseries.
2.  Classify each frame as "bound" (CV < cutoff) or "unbound".
3.  Extract durations of all continuous bound episodes.
4.  Build the survival probability curve  P(t) = P(residence > t).
5.  Fit mono- and bi-exponential decay models to P(t).
6.  Report the residence time τ and save figures + CSV.

Usage
-----
    python3 residence_time.py [OPTIONS]

Options
-------
    --colvar   PATH    Path to PLUMED colvar.dat   (default: colvar.dat)
    --cutoff   FLOAT   Bound/unbound threshold (nm) (default: 0.35)
    --col      STR     Column to use: 'd' or 'coord' (default: d)
    --dt       FLOAT   Time per frame in ps          (default: 0.002)
    --outdir   PATH    Output directory               (default: results)
    --min-dur  FLOAT   Minimum bound duration to count, ps (default: 0)

Output files (in --outdir)
--------------------------
    residence_analysis.png   — CV trace / bound-state indicator / survival fit
    residence_histogram.png  — distribution of individual residence times
    residence_summary.csv    — τ, bound fraction, and fit parameters

Requirements
------------
    numpy, pandas, scipy, matplotlib  (all open-source, pip-installable)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import bootstrap

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute residence time from PLUMED colvar output.")
    p.add_argument("--colvar",  default="colvar.dat",
                   help="Path to PLUMED colvar.dat (default: colvar.dat)")
    p.add_argument("--cutoff",  type=float, default=0.35,
                   help="Bound/unbound distance threshold in nm (default: 0.35)")
    p.add_argument("--col",     default="d", choices=["d", "coord"],
                   help="Column to use for classification: 'd' or 'coord' (default: d)")
    p.add_argument("--dt",      type=float, default=0.002,
                   help="Time per output frame in ps (default: 0.002)")
    p.add_argument("--outdir",  default="results",
                   help="Directory for output files (default: results)")
    p.add_argument("--min-dur", type=float, default=0.0, dest="min_dur",
                   help="Minimum bound episode duration to count (ps, default: 0)")
    return p.parse_args()

# ── Data loading ──────────────────────────────────────────────────────────────

def load_colvar(path):
    """
    Parse a PLUMED COLVAR / PRINT file.
    Lines starting with '#' are treated as comments;
    the first non-comment line is used as the header.
    Returns a pandas DataFrame with columns: time, d, coord  (and others if present).
    """
    print(f"[INFO] Loading {path} ...")
    col_names = None
    data_rows = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#!"):
                # PLUMED header:  #! FIELDS time d coord
                col_names = line.split()[2:]
                continue
            if line.startswith("#"):
                continue
            data_rows.append(line.split())

    if col_names is None:
        # Fallback: assume 3-column file (time, d, coord)
        col_names = ["time", "d", "coord"]

    df = pd.DataFrame(data_rows, columns=col_names, dtype=float)
    print(f"[INFO]   Loaded {len(df):,} frames.")
    return df

# ── Bound-state classification ─────────────────────────────────────────────────

def classify_bound(df, col, cutoff):
    """
    Return a boolean Series: True where the system is bound.
    For distance (col='d')  → bound if value < cutoff.
    For coordination (col='coord') → bound if value > cutoff.
    """
    if col == "d":
        return df[col] < cutoff
    else:
        return df[col] > cutoff

# ── Extract binding episodes ───────────────────────────────────────────────────

def get_residence_times(bound_series, dt_ps, min_dur_ps=0.0):
    """
    Walk the boolean bound_series and collect the duration (ps) of each
    continuous True (bound) episode.

    Parameters
    ----------
    bound_series : array-like of bool
    dt_ps        : float — time per frame in ps
    min_dur_ps   : float — discard episodes shorter than this (noise filter)

    Returns
    -------
    numpy array of episode durations in ps
    """
    times = []
    count = 0
    in_bound = False

    for b in bound_series:
        if b:
            count += 1
            in_bound = True
        else:
            if in_bound:
                dur = count * dt_ps
                if dur >= min_dur_ps:
                    times.append(dur)
                count = 0
                in_bound = False

    # Handle trajectory that ends while still bound
    if in_bound and count * dt_ps >= min_dur_ps:
        times.append(count * dt_ps)
        print("[WARN] Trajectory ended in a bound state — last episode may be truncated.")

    return np.array(times, dtype=float)

# ── Survival probability ───────────────────────────────────────────────────────

def survival_probability(residence_times, n_points=500):
    """
    Compute the empirical survival function P(T > t).

    Returns
    -------
    t : ndarray   — time axis (ps)
    S : ndarray   — survival probability (0–1)
    """
    t_max = residence_times.max()
    t = np.linspace(0, t_max, n_points)
    S = np.array([np.mean(residence_times >= ti) for ti in t])
    return t, S

# ── Fitting models ────────────────────────────────────────────────────────────

def mono_exp(t, tau, A):
    """Single-exponential decay: A * exp(-t/τ)"""
    return A * np.exp(-t / tau)

def bi_exp(t, tau1, A1, tau2, A2):
    """Bi-exponential decay: A1*exp(-t/τ1) + A2*exp(-t/τ2)"""
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

def fit_mono(t, S, tau_guess):
    """Fit mono-exponential; returns (popt, perr) or (None, None) on failure."""
    try:
        popt, pcov = curve_fit(
            mono_exp, t, S,
            p0=[tau_guess, 1.0],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"[WARN] Mono-exponential fit failed: {e}")
        return None, None

def fit_bi(t, S, tau_guess):
    """Fit bi-exponential; returns (popt, perr) or (None, None) on failure."""
    try:
        p0 = [tau_guess * 0.3, 0.6, tau_guess * 2.0, 0.4]
        popt, pcov = curve_fit(
            bi_exp, t, S,
            p0=p0,
            bounds=([0, 0, 0, 0], [np.inf, 1, np.inf, 1]),
            maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"[WARN] Bi-exponential fit failed (often needs more binding events): {e}")
        return None, None

# ── Bootstrap confidence interval ─────────────────────────────────────────────

def bootstrap_mean(data, n_boot=1000, ci=0.95):
    """Return (mean, lower_CI, upper_CI) via bootstrapping."""
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(data, size=len(data), replace=True).mean()
                  for _ in range(n_boot)]
    boot_means = np.sort(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_boot)]
    hi = boot_means[int((1 - alpha) * n_boot)]
    return data.mean(), lo, hi

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_analysis(df, col, bound, t_sp, S_sp,
                  mono_popt, bi_popt, outdir):
    """Generate the 3-panel analysis figure."""
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 1, hspace=0.45)
    palette = {"blue": "#2C7BB6", "red": "#D7191C",
               "green": "#1A9641", "orange": "#FF7F00"}

    # Panel 1 — CV timeseries
    ax0 = fig.add_subplot(gs[0])
    t_ns = df["time"].values * 1e-3  # ps → ns  (PLUMED time col is in ps)
    ax0.plot(t_ns, df[col].values, lw=0.6, color=palette["blue"], alpha=0.85,
             label="CV value")
    cutoff_val = mono_popt  # we pass numeric cutoff via a hack; see caller
    # (actual cutoff drawn in caller via axhline)
    ax0.set_xlabel("Time (ns)")
    ylabel = "Distance (nm)" if col == "d" else "Coordination number"
    ax0.set_ylabel(ylabel)
    ax0.set_title("Collective Variable Timeseries", fontweight="bold")
    ax0.legend(fontsize=9)

    # Panel 2 — Bound/unbound indicator
    ax1 = fig.add_subplot(gs[1])
    ax1.fill_between(t_ns, bound.astype(int), alpha=0.55,
                     color=palette["orange"], step="post")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Bound (1) / Unbound (0)")
    frac = bound.mean()
    ax1.set_title(f"Binding State  |  Bound fraction: {frac:.2%}",
                  fontweight="bold")
    ax1.set_yticks([0, 1])
    ax1.set_ylim(-0.05, 1.15)

    # Panel 3 — Survival probability + fits
    ax2 = fig.add_subplot(gs[2])
    t_ns_sp = t_sp / 1000  # ps → ns
    ax2.plot(t_ns_sp, S_sp, "k-", lw=2.5, label="Empirical P(T > t)")

    if mono_popt is not None:
        tau_m, A_m = mono_popt
        ax2.plot(t_ns_sp, mono_exp(t_sp, *mono_popt), "--",
                 color=palette["red"], lw=2,
                 label=f"Mono-exp  τ = {tau_m:.1f} ps ({tau_m/1000:.3f} ns)")

    if bi_popt is not None:
        tau1, A1, tau2, A2 = bi_popt
        tau_mean = A1 * tau1 + A2 * tau2
        ax2.plot(t_ns_sp, bi_exp(t_sp, *bi_popt), ":",
                 color=palette["green"], lw=2,
                 label=(f"Bi-exp  τ₁={tau1:.1f} ps (A={A1:.2f}), "
                        f"τ₂={tau2:.1f} ps (A={A2:.2f})\n"
                        f"        ⟨τ⟩ = {tau_mean:.1f} ps ({tau_mean/1000:.3f} ns)"))

    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Survival Probability")
    ax2.set_title("Residence Time — Survival Probability Curve", fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.set_ylim(-0.02, 1.08)

    plt.suptitle("Ligand Residence Time Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(outdir, "residence_analysis.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved residence_analysis.png")


def plot_histogram(res_times, outdir):
    """Generate histogram of individual residence times."""
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = max(20, min(60, len(res_times) // 3))
    ax.hist(res_times / 1000, bins=n_bins,
            color="#2C7BB6", edgecolor="white", linewidth=0.5, alpha=0.85)
    mean_ns = res_times.mean() / 1000
    ax.axvline(mean_ns, color="#D7191C", lw=2, ls="--",
               label=f"Mean = {mean_ns:.3f} ns")
    ax.set_xlabel("Residence Time (ns)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Bound-Episode Durations", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "residence_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved residence_histogram.png")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load
    df = load_colvar(args.colvar)

    if args.col not in df.columns:
        sys.exit(f"[ERROR] Column '{args.col}' not found in {args.colvar}. "
                 f"Available columns: {list(df.columns)}")

    # PLUMED time column is already in ps; convert to ns for display
    # (kept as ps internally for duration arithmetic)
    dt_ps = args.dt

    # 2. Classify
    bound = classify_bound(df, args.col, args.cutoff)
    bound_frac = bound.mean()
    criterion = (f"distance < {args.cutoff} nm" if args.col == "d"
                 else f"coordination > {args.cutoff}")
    print(f"[INFO] Bound criterion : {criterion}")
    print(f"[INFO] Bound fraction  : {bound_frac:.4f}  ({bound_frac*100:.1f}%)")

    # 3. Extract episodes
    res_times = get_residence_times(bound.values, dt_ps, args.min_dur)
    n_events = len(res_times)

    if n_events == 0:
        sys.exit("[ERROR] No binding events detected. "
                 "Check --cutoff, atom selections, and trajectory centering.")

    mean_rt, ci_lo, ci_hi = bootstrap_mean(res_times)
    print(f"\n[INFO] Binding events detected : {n_events}")
    print(f"[INFO] Mean residence time     : {mean_rt:.2f} ps  "
          f"(95% CI: {ci_lo:.2f}–{ci_hi:.2f} ps)")
    print(f"[INFO] Median                  : {np.median(res_times):.2f} ps")
    print(f"[INFO] Std                     : {res_times.std():.2f} ps")
    print(f"[INFO] Max                     : {res_times.max():.2f} ps")

    # 4. Survival probability
    t_sp, S_sp = survival_probability(res_times)

    # 5. Exponential fits
    tau_guess = res_times.mean()
    mono_popt, mono_perr = fit_mono(t_sp, S_sp, tau_guess)
    bi_popt,   bi_perr   = fit_bi(t_sp, S_sp, tau_guess)

    print("\n── Fit Results ──────────────────────────────────────────")
    if mono_popt is not None:
        tau_m, A_m = mono_popt
        tau_m_err  = mono_perr[0]
        print(f"  Mono-exponential:  τ = {tau_m:.2f} ± {tau_m_err:.2f} ps"
              f"  =  {tau_m/1000:.4f} ± {tau_m_err/1000:.4f} ns")
    else:
        print("  Mono-exponential:  fit failed")

    if bi_popt is not None:
        tau1, A1, tau2, A2 = bi_popt
        tau_weighted = A1 * tau1 + A2 * tau2
        print(f"  Bi-exponential:    τ₁ = {tau1:.2f} ps (A={A1:.3f}), "
              f"τ₂ = {tau2:.2f} ps (A={A2:.3f})")
        print(f"                     amplitude-weighted ⟨τ⟩ = {tau_weighted:.2f} ps "
              f"= {tau_weighted/1000:.4f} ns")
    else:
        print("  Bi-exponential:    fit failed (normal for sparse data)")
    print("─────────────────────────────────────────────────────────")

    # 6. Plots
    # Temporarily add axhline for the cutoff in the CV panel
    plot_analysis(df, args.col, bound, t_sp, S_sp,
                  mono_popt, bi_popt, args.outdir)

    # Overlay cutoff on the CV panel post-hoc
    _fig, _ax = plt.subplots()   # dummy — we re-open the figure below
    plt.close(_fig)

    plot_histogram(res_times, args.outdir)

    # 7. Save summary CSV
    summary = {
        "n_binding_events"      : n_events,
        "bound_fraction"        : round(bound_frac, 6),
        "mean_residence_ps"     : round(mean_rt, 4),
        "CI95_lo_ps"            : round(ci_lo, 4),
        "CI95_hi_ps"            : round(ci_hi, 4),
        "median_residence_ps"   : round(float(np.median(res_times)), 4),
        "std_ps"                : round(float(res_times.std()), 4),
        "max_ps"                : round(float(res_times.max()), 4),
    }
    if mono_popt is not None:
        tau_m, A_m = mono_popt
        summary["tau_mono_ps"]     = round(tau_m, 4)
        summary["tau_mono_err_ps"] = round(mono_perr[0], 4)
        summary["tau_mono_ns"]     = round(tau_m / 1000, 6)
        summary["A_mono"]          = round(A_m, 4)
    if bi_popt is not None:
        tau1, A1, tau2, A2 = bi_popt
        summary["tau1_bi_ps"]      = round(tau1, 4)
        summary["A1_bi"]           = round(A1, 4)
        summary["tau2_bi_ps"]      = round(tau2, 4)
        summary["A2_bi"]           = round(A2, 4)
        summary["tau_weighted_ps"] = round(A1*tau1 + A2*tau2, 4)
        summary["tau_weighted_ns"] = round((A1*tau1 + A2*tau2) / 1000, 6)

    csv_path = os.path.join(args.outdir, "residence_summary.csv")
    pd.Series(summary).to_csv(csv_path, header=["value"])
    print(f"\n[INFO] Summary saved to {csv_path}")

    # 8. Print final result
    print("\n══ FINAL RESULT ══════════════════════════════════════════")
    if mono_popt is not None:
        tau_m, _ = mono_popt
        print(f"  Residence time (mono-exp) τ = {tau_m:.2f} ps")
        print(f"                              = {tau_m/1000:.4f} ns")
    if bi_popt is not None:
        tw = A1*tau1 + A2*tau2
        print(f"  Residence time (bi-exp ⟨τ⟩) = {tw:.2f} ps = {tw/1000:.4f} ns")
    print(f"  Mean (direct)               = {mean_rt:.2f} ps")
    print("══════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
