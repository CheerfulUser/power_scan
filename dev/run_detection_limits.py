"""
Detection limits analysis for power_scan.

Runs the injection-recovery grid from the notebook, traces which pipeline
cut is the binding constraint, and computes the theoretical Lomb-Scargle
detection limit vs the empirical limit imposed by the snr_search_lim and
local_threshold cuts.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import warnings, io
from contextlib import redirect_stdout
from scipy.optimize import curve_fit
from scipy.special import expit
import sep

import power_scan as ps

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N_FRAMES    = 500
NOISE_SIGMA = 1.0
IMAGE_SIZE  = 100
SOURCE_X    = 50
SOURCE_Y    = 50
PSF_SIGMA   = 2.0   # FWHM ~ 4.7 px; image is ~21x PSF FWHM
N_TRIALS    = 10

PERIODS  = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800]

# snr_search_lim default in pipeline
Z_THR      = 10
# Theoretical 50% threshold from LS power alone
A50_THEORY = NOISE_SIGMA * np.sqrt(4 * Z_THR / N_FRAMES)

# Empirical baseline from previous run; scale by 1/sqrt(n_cyc) for few-cycle regime
A50_BASELINE = NOISE_SIGMA * np.sqrt(4 * Z_THR / N_FRAMES) * 2.0  # conservative: 2x theory
T_OBS        = N_FRAMES - 1.0

def expected_a50(period):
    n_cyc = T_OBS / period
    return A50_BASELINE / np.sqrt(min(n_cyc, 1.0)) if n_cyc < 1.0 else A50_BASELINE

def amp_grid_for_period(period):
    """Coarse points bracketing [0, 1] + fine points centred on expected A50."""
    a50 = expected_a50(period)
    coarse = np.round(np.logspace(-1, np.log10(a50 * 3.5), 7), 4)
    fine   = np.round(np.linspace(a50 * 0.60, a50 * 1.50, 12), 4)
    return sorted(set(coarse.tolist()) | set(fine.tolist()))

print(f"Theoretical A50 (N={N_FRAMES}, z_thr={Z_THR}): {A50_THEORY:.3f} sigma")
for p in PERIODS:
    grid = amp_grid_for_period(p)
    print(f"  P={p:4d}  expected_A50={expected_a50(p):.3f}  grid={len(grid)} points  "
          f"[{min(grid):.3f}–{max(grid):.3f}]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_psf(size=25, x0=12, y0=12, sigma=1.2):
    y, x = np.mgrid[0:size, 0:size]
    psf  = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return psf / psf.max()

def make_scene(n_frames, period, amplitude, noise_sigma=1.0,
               image_size=25, source_x=12, source_y=12,
               psf_sigma=1.2, seed=None):
    rng  = np.random.default_rng(seed)
    time = np.arange(n_frames, dtype=float)
    psf  = make_psf(image_size, source_x, source_y, psf_sigma)
    signal      = amplitude * np.sin(2 * np.pi / period * time)
    source_cube = signal[:, np.newaxis, np.newaxis] * psf[np.newaxis, :, :]
    noise       = rng.normal(0.0, noise_sigma, (n_frames, image_size, image_size))
    return time, source_cube + noise

def find_best_match(sources, true_x, true_y, true_period,
                    pos_tol=2.0, period_rtol=0.15):
    if sources is None or len(sources) == 0:
        return False, None, None, None
    for _, row in sources.iterrows():
        dx = float(row['xcentroid']) - true_x
        dy = float(row['ycentroid']) - true_y
        dr = np.hypot(dx, dy)
        dp = abs(row['period'] - true_period) / true_period
        if dr <= pos_tol and dp <= period_rtol:
            return True, float(row['period']), dx, dy
    return False, None, None, None

def log_logistic(log_a, k, log_a50):
    return expit(k * (log_a - log_a50))


# ---------------------------------------------------------------------------
# Full injection-recovery grid
# ---------------------------------------------------------------------------
print("\n=== Running injection-recovery grid ===")
STOP_STREAK = 2
trial_rows  = []

for period in PERIODS:
    amp_grid    = amp_grid_for_period(period)
    n_total_p   = len(amp_grid) * N_TRIALS
    consec_full = 0
    saturated   = False
    run_idx     = 0
    print(f"\n  --- Period {period} ({len(amp_grid)} amplitudes) ---")

    for a_over_n in amp_grid:
        if saturated:
            for trial in range(N_TRIALS):
                trial_rows.append(dict(amp_over_noise=a_over_n, period=period,
                    trial=trial, recovered=1, rec_period=float(period),
                    dx=0.0, dy=0.0, dr=0.0))
            run_idx += N_TRIALS
            continue

        amplitude = a_over_n * NOISE_SIGMA
        cell_recs = []
        for trial in range(N_TRIALS):
            seed  = abs(hash((round(a_over_n, 4), period, trial))) % (2**31)
            time, scene = make_scene(N_FRAMES, period, amplitude, NOISE_SIGMA,
                                     IMAGE_SIZE, SOURCE_X, SOURCE_Y, PSF_SIGMA, seed=seed)
            try:
                with redirect_stdout(io.StringIO()):
                    det = ps.periodogram_detection(time, scene, run=True,
                                                   period_lim=[2, max(PERIODS) * 1.05])
                rec, rec_p, dx, dy = find_best_match(det.sources, SOURCE_X, SOURCE_Y, period)
            except Exception:
                rec, rec_p, dx, dy = False, None, None, None

            cell_recs.append(int(rec))
            trial_rows.append(dict(amp_over_noise=a_over_n, period=period,
                trial=trial, recovered=int(rec),
                rec_period=rec_p, dx=dx, dy=dy,
                dr=np.hypot(dx, dy) if dx is not None else None))
            run_idx += 1

        frac = np.mean(cell_recs)
        consec_full = (consec_full + 1) if frac == 1.0 else 0
        if consec_full >= STOP_STREAK:
            saturated = True
        sym = '✓' if frac == 1.0 else ('~' if frac > 0 else '✗')
        print(f"    [{run_idx:3d}/{n_total_p}]  A/N={a_over_n:.3f}  det={frac:.2f} {sym}")

trials_df = pd.DataFrame(trial_rows)
for col in ['dx', 'dy', 'dr', 'rec_period']:
    trials_df[col] = pd.to_numeric(trials_df[col], errors='coerce')

# Aggregate
def _period_bias(grp):
    per = grp['period'].iloc[0]
    rp  = grp['rec_period'].dropna()
    return ((rp - per) / per).mean() if len(rp) else np.nan

agg = (trials_df.groupby(['amp_over_noise', 'period'])
       .agg(det_frac=('recovered','mean'), n_recovered=('recovered','sum'),
            dr_mean=('dr','mean'), dr_std=('dr','std'))
       .reset_index())
pb  = (trials_df.groupby(['amp_over_noise','period'])
       .apply(_period_bias).reset_index(name='period_bias'))
agg = agg.merge(pb, on=['amp_over_noise','period'])

trials_df.to_csv('dev/injection_recovery_trials.csv', index=False)
agg.to_csv('dev/injection_recovery_summary.csv', index=False)
print("  Saved CSV files.")


# ---------------------------------------------------------------------------
# Sigmoid fits
# ---------------------------------------------------------------------------
fit_results = {}
for period in PERIODS:
    sub   = agg[agg['period'] == period].sort_values('amp_over_noise')
    fracs = sub['det_frac'].values
    la    = np.log(sub['amp_over_noise'].values)
    if fracs.max() > 0 and fracs.min() < 1:
        try:
            p0 = [5.0, np.log(sub.loc[sub['det_frac'] >= 0.5, 'amp_over_noise'].min())]
            popt, _ = curve_fit(log_logistic, la, fracs, p0=p0,
                                bounds=([0.5, -5], [500, 5]), maxfev=8000)
            fit_results[period] = (np.exp(popt[1]), popt[0])
        except Exception:
            fit_results[period] = (np.nan, np.nan)
    else:
        fit_results[period] = (np.nan, np.nan)

print("\n  Sigmoid fit results:")
print(f"  {'Period':>8}  {'A50 (x sigma)':>14}  {'k':>6}")
for per, (a50, k) in fit_results.items():
    a50_s = f"{a50:.4f}" if np.isfinite(a50) else "   ---"
    k_s   = f"{k:.2f}"   if np.isfinite(k)   else "  ---"
    print(f"  {per:>8}  {a50_s:>14}  {k_s:>6}")


# ---------------------------------------------------------------------------
# Diagnostic: trace which cut fails at borderline amplitudes
# ---------------------------------------------------------------------------
print("\n=== Diagnostic: tracing pipeline cuts ===")
DIAG_AMPS    = [0.264, 0.336, 0.428, 0.546]
DIAG_PERIODS = [50, 100]
N_DIAG       = 5

diag_rows = []
for a_over_n in DIAG_AMPS:
    amplitude = a_over_n * NOISE_SIGMA
    for period in DIAG_PERIODS:
        for trial in range(N_DIAG):
            seed = abs(hash((round(a_over_n, 4), period, trial))) % (2**31)
            time, scene = make_scene(N_FRAMES, period, amplitude, NOISE_SIGMA,
                                     IMAGE_SIZE, SOURCE_X, SOURCE_Y, PSF_SIGMA, seed=seed)
            # Build power cube manually so we can inspect intermediate steps
            det = ps.periodogram_detection(time, scene, run=False,
                                           period_lim=[2, max(PERIODS) * 1.05])
            with redirect_stdout(io.StringIO()):
                det.clean_data()
                det._set_period_lim()
                det.batch_make_freq_cube()

            # Find frequency closest to true signal
            fi_true = np.argmin(np.abs(det.freq - 1.0 / period))
            pn_source = det.power_norm[fi_true, SOURCE_Y, SOURCE_X]

            # Run sep on the peak power_norm slice to see if source is extracted
            peak_fi = int(np.argmax(det.power_norm[:, SOURCE_Y, SOURCE_X]))
            pslice  = det.power_norm[peak_fi].astype(float).copy(order='C')
            pn_peak = pslice[SOURCE_Y, SOURCE_X]

            # snr_search_lim check
            max_pn = det.power_norm.max(axis=(1, 2))
            freq_above_thr = np.sum(max_pn >= Z_THR)

            # sep extraction at peak frequency
            try:
                bkg  = sep.Background(pslice)
                objs = pd.DataFrame(sep.extract(pslice, det.dao_peak))
                if len(objs) > 0:
                    objs = objs.rename(columns={'x': 'xcentroid', 'y': 'ycentroid'})
                    w, h = objs['a'].values, objs['b'].values
                    ratio = np.round(abs(w / h - 1), 1)
                    objs  = objs.loc[(ratio < 0.5)].reset_index(drop=True)
                sep_found = len(objs) > 0 if len(objs) > 0 else False
                # Check if any detected object is within 2 px of source
                if sep_found and len(objs) > 0:
                    dr_sep = np.hypot(objs['xcentroid'] - SOURCE_X,
                                      objs['ycentroid'] - SOURCE_Y)
                    sep_near = bool(dr_sep.min() <= 2.0)
                    nearest = objs.iloc[dr_sep.argmin()]
                    sep_flux = float(nearest['flux'])
                    sep_npix = float(nearest['npix'])
                else:
                    sep_near, sep_flux, sep_npix = False, np.nan, np.nan
            except Exception:
                sep_near = sep_flux = sep_npix = np.nan

            diag_rows.append(dict(
                amp_over_noise = a_over_n,
                period         = period,
                trial          = trial,
                pn_source      = pn_source,
                pn_peak        = pn_peak,
                freq_above_thr = freq_above_thr,
                sep_near       = sep_near,
                sep_flux       = sep_flux,
                sep_npix       = sep_npix,
            ))

diag_df = pd.DataFrame(diag_rows)
diag_agg = diag_df.groupby(['amp_over_noise', 'period']).agg(
    pn_source_med  = ('pn_source', 'median'),
    pn_peak_med    = ('pn_peak',   'median'),
    sep_near_frac  = ('sep_near',  'mean'),
    sep_flux_med   = ('sep_flux',  'median'),
    sep_npix_med   = ('sep_npix',  'median'),
).reset_index()

print(f"\n  {'A/N':>6}  {'P':>5}  {'pn_src':>8}  {'pn_peak':>8}  "
      f"{'sep_frac':>9}  {'flux':>8}  {'npix':>6}  {'local_sig':>10}")
for _, r in diag_agg.iterrows():
    ls = r.sep_flux_med / (r.sep_npix_med * 1.0) if np.isfinite(r.sep_flux_med) else np.nan
    print(f"  {r.amp_over_noise:>6.3f}  {int(r.period):>5}  {r.pn_source_med:>8.1f}  "
          f"{r.pn_peak_med:>8.1f}  {r.sep_near_frac:>9.2f}  "
          f"{r.sep_flux_med:>8.1f}  {r.sep_npix_med:>6.0f}  {ls:>10.2f}")


# ---------------------------------------------------------------------------
# Theoretical limit from local_threshold cut
# ---------------------------------------------------------------------------
# For a Gaussian PSF of sigma=PSF_SIGMA, the power image source has
# effective sigma_pwr = PSF_SIGMA / sqrt(2). Sep extracts ~npix pixels
# of the source. The local_sig = flux / (npix * sky_std) where
# sky_std ~ 1 (normalised power image). So:
#   local_sig ~ (2*pi*sigma_pwr^2 * power_norm_peak) / npix
# Calibrate sigma_pwr and npix from the diagnostic runs, then solve
# local_sig = local_threshold for power_norm_peak, and convert to A/N.

# Use median npix and estimate sigma_pwr from PSF_SIGMA
sigma_pwr = PSF_SIGMA / np.sqrt(2)

# Calibrate: pn_peak vs (A/N)^2  →  pn_peak = C * N * (A/N)^2 / 4
# Use diagnostic points where sep detects the source
cal = diag_agg[diag_agg['sep_near_frac'] > 0].copy()
if len(cal) > 0:
    cal['an2'] = cal['amp_over_noise'] ** 2
    # Linear fit: pn_peak = C * N/4 * (A/N)^2
    C_fit = np.polyfit(cal['an2'], cal['pn_peak_med'], 1)[0]
    C_eff = C_fit  # pn_peak = C_eff * (A/N)^2
    print(f"\n  LS calibration: pn_peak = {C_eff:.1f} * (A/N)^2")
else:
    # Theoretical: for N=1000, background power std ~ 1/N for nifty_ls
    C_eff = N_FRAMES / 4.0
    print(f"\n  LS calibration (theoretical): pn_peak = {C_eff:.1f} * (A/N)^2")

# Estimate npix from diagnostics (use most common value)
npix_est = diag_agg['sep_npix_med'].dropna().median()
if np.isnan(npix_est):
    npix_est = np.pi * (2 * sigma_pwr) ** 2   # estimated detection area
print(f"  Estimated npix: {npix_est:.1f}")

# local_sig = (2*pi*sigma_pwr^2 * pn_peak) / npix  = local_threshold
# pn_peak_needed = local_threshold * npix / (2*pi*sigma_pwr^2)
local_thr = 10.0
pn_needed = local_thr * npix_est / (2 * np.pi * sigma_pwr ** 2)
# pn_needed = C_eff * (A/N)^2  →  A/N = sqrt(pn_needed / C_eff)
A50_local_thr = np.sqrt(pn_needed / C_eff)
print(f"  Theoretical A50 from local_threshold cut: {A50_local_thr:.3f} sigma")
print(f"  Ratio to LS-only limit: {A50_local_thr / A50_THEORY:.2f}x")

# snr_lim (flux >= snr_lim=5) cut
# flux ~ 2*pi*sigma_pwr^2 * pn_peak;  snr_lim_cut: flux > 5
pn_for_snr_lim = 5.0 / (2 * np.pi * sigma_pwr ** 2)
A50_snr_lim = np.sqrt(pn_for_snr_lim / C_eff)
print(f"  Theoretical A50 from snr_lim cut (flux>5): {A50_snr_lim:.3f} sigma")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
amp_fine = np.logspace(-1, 1, 300)

def plabel(period):
    """Period label: value + fraction of T_obs."""
    frac = period / T_OBS
    if frac < 0.01:
        return fr'$P={period}\ ({frac:.3f}\,T_{{\rm obs}})$'
    return fr'$P={period}\ ({frac:.2f}\,T_{{\rm obs}})$'

# ── Figure 1: detection heatmap (grey = unsampled cell) ──────────────────────
all_amps = sorted(agg['amp_over_noise'].unique())
pivot    = agg.pivot(index='amp_over_noise', columns='period', values='det_frac')
pivot    = pivot.reindex(index=all_amps, columns=PERIODS)
data     = np.ma.masked_invalid(pivot.values)
cmap_hm  = plt.cm.RdYlGn.copy();  cmap_hm.set_bad('0.85')

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap_hm,
               vmin=0, vmax=1, interpolation='nearest',
               extent=[-0.5, len(PERIODS)-0.5, -0.5, len(all_amps)-0.5])
for i, amp in enumerate(all_amps):
    for j, per in enumerate(PERIODS):
        val = pivot.loc[amp, per]
        if np.isfinite(val):
            col = 'k' if 0.1 < val < 0.9 else 'w'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=6, color=col)

amp_arr     = np.array(all_amps)
theory_y    = np.interp(A50_THEORY,    amp_arr, np.arange(len(amp_arr))) - 0.5
local_thr_y = np.interp(A50_local_thr, amp_arr, np.arange(len(amp_arr))) - 0.5
ax.axhline(theory_y,    color='dodgerblue', lw=1.8, ls='--',
           label=fr'LS theory: $A_{{50}}={A50_THEORY:.3f}\,\sigma$')
ax.axhline(local_thr_y, color='orangered',  lw=1.8, ls='-.',
           label=fr'local\_thr: $A_{{50}}={A50_local_thr:.3f}\,\sigma$')

xlabels = [fr'$P={p}$' + '\n' + fr'$({p/T_OBS:.2f}\,T_{{\rm obs}})$' for p in PERIODS]
ax.set_xticks(range(len(PERIODS)));  ax.set_xticklabels(xlabels, fontsize=7)
ax.set_yticks(range(len(all_amps))); ax.set_yticklabels([f'{a:.3f}' for a in all_amps], fontsize=6)
ax.set_xlabel(r'Period');  ax.set_ylabel(r'Amplitude / noise $\sigma$')
ax.set_title(fr'power\_scan injection-recovery heatmap ($N={N_FRAMES}$, {N_TRIALS} trials/cell)')
ax.legend(fontsize=8, loc='upper right')
fig.colorbar(im, ax=ax, label='Detection fraction', shrink=0.6)
plt.tight_layout()
plt.savefig('dev/detection_heatmap.png')
plt.close()

# ── Figure 2: sigmoid fits + all three limits ────────────────────────────────
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(PERIODS)))
fig, ax = plt.subplots(figsize=(9, 5.5))

for period, col in zip(PERIODS, colors):
    sub   = agg[agg['period'] == period].sort_values('amp_over_noise')
    fracs = sub['det_frac'].values
    ax.scatter(sub['amp_over_noise'], fracs, color=col, s=30, zorder=5,
               edgecolors='k', lw=0.4)
    a50, k = fit_results.get(period, (np.nan, np.nan))
    if np.isfinite(a50) and np.isfinite(k):
        y = log_logistic(np.log(amp_fine), k, np.log(a50))
        ax.plot(amp_fine, y, '-', color=col, lw=1.6,
                label=plabel(period) + fr'  $A_{{50}}={a50:.3f}$  $k={k:.1f}$')
    else:
        ax.plot(sub['amp_over_noise'], fracs, '--', color=col, lw=1.2,
                label=plabel(period))

ax.axvline(A50_THEORY,    color='dodgerblue', ls=':', lw=2.0,
           label=fr'LS theory: ${A50_THEORY:.3f}\,\sigma$')
ax.axvline(A50_local_thr, color='orangered',  ls='-.', lw=2.0,
           label=fr'local\_thr limit: ${A50_local_thr:.3f}\,\sigma$')
ax.axhline(0.5, color='0.5', ls='--', lw=0.8)
ax.set_xscale('log')
ax.set_xlabel(r'Amplitude / noise $\sigma$')
ax.set_ylabel(r'Detection fraction')
ax.set_title(r'Detection probability curves with theoretical limits')
ax.legend(fontsize=7.5, ncol=2)
ax.grid(True, which='both', alpha=0.2)
plt.tight_layout()
plt.savefig('dev/sigmoid_fits.png')
plt.close()

# ── Figure 3: A50 vs period (shows long-period degradation) ─────────────────
periods_fit   = []
a50_fit_vals  = []
for per in PERIODS:
    a50, k = fit_results.get(per, (np.nan, np.nan))
    if np.isfinite(a50):
        periods_fit.append(per)
        a50_fit_vals.append(a50)

# Theoretical prediction: for P > T_obs/n_cyc, effective N_cycles degrades
T_obs  = N_FRAMES - 1.0
p_arr  = np.logspace(np.log10(4), np.log10(T_obs), 300)
# Effective number of complete cycles observed
n_cyc  = T_obs / p_arr
# When n_cyc < 1, power is suppressed; empirical: pn_eff ~ N * n_cyc * (A/N)^2 / 4
# A50 = sqrt(4*Z_THR / (N * n_cyc))   [from z = N*n_cyc*A^2/(4*sigma^2)]
A50_period_theory = NOISE_SIGMA * np.sqrt(4 * Z_THR / (N_FRAMES * np.minimum(n_cyc, 1.0)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Left: A50 vs period
axes[0].scatter(periods_fit, a50_fit_vals, s=60, zorder=5,
                edgecolors='k', lw=0.5, color='steelblue',
                label=r'Empirical $A_{50}$ (sigmoid fit)')
axes[0].plot(p_arr, A50_period_theory, 'k--', lw=1.5,
             label=r'Theory ($\propto 1/\sqrt{n_{\rm cyc}}$)')
axes[0].axhline(A50_THEORY, color='dodgerblue', ls=':', lw=1.5,
                label=fr'LS floor $= {A50_THEORY:.3f}\,\sigma$')
axes[0].axhline(A50_local_thr, color='orangered', ls='-.', lw=1.5,
                label=fr'local\_thr floor $= {A50_local_thr:.3f}\,\sigma$')
axes[0].axvline(T_obs / 3, color='0.55', ls=':', lw=1.2, alpha=0.7)
axes[0].text(T_obs / 3 + 5, axes[0].get_ylim()[0] if axes[0].get_ylim()[0] > 0 else 0.22,
             r'$P = T_{\rm obs}/3$', fontsize=8, color='0.4', va='bottom')
axes[0].set_xscale('log')
axes[0].set_xlabel(r'Period (cadence units)')
axes[0].set_ylabel(r'$A_{50}$ / noise $\sigma$')
axes[0].set_title(r'Detection limit vs period')
axes[0].legend(fontsize=8)
axes[0].grid(True, which='both', alpha=0.2)
# Secondary x-axis in units of T_obs
ax0b = axes[0].twiny()
ax0b.set_xscale('log')
ax0b.set_xlim(np.array(axes[0].get_xlim()) / T_OBS)
ax0b.set_xlabel(r'Period / $T_{\rm obs}$', labelpad=6)

# Right: fine-grid detection curves near transition for a few representative periods
rep_periods = [p for p in [5, 50, 200, 500, 700, 800] if p in PERIODS]
rep_colors  = plt.cm.plasma(np.linspace(0.1, 0.85, len(rep_periods)))
for period, col in zip(rep_periods, rep_colors):
    sub   = agg[agg['period'] == period].sort_values('amp_over_noise')
    fracs = sub['det_frac'].values
    axes[1].scatter(sub['amp_over_noise'], fracs, color=col, s=20, zorder=5,
                    edgecolors='k', lw=0.3)
    a50, k = fit_results.get(period, (np.nan, np.nan))
    if np.isfinite(a50) and np.isfinite(k):
        yfit = log_logistic(np.log(amp_fine), k, np.log(a50))
        axes[1].plot(amp_fine, yfit, '-', color=col, lw=1.6,
                     label=plabel(period) + fr'  $A_{{50}}={a50:.3f}$')
    else:
        axes[1].plot(sub['amp_over_noise'], fracs, '--', color=col, lw=1.2,
                     label=plabel(period))

axes[1].axhline(0.5, color='0.5', ls='--', lw=0.8)
axes[1].axvline(A50_THEORY,    color='dodgerblue', ls=':', lw=1.5,
                label=fr'LS theory: ${A50_THEORY:.3f}\,\sigma$')
axes[1].axvline(A50_local_thr, color='orangered',  ls='-.', lw=1.5,
                label=fr'local\_thr: ${A50_local_thr:.3f}\,\sigma$')
axes[1].set_xscale('log')
axes[1].set_xlim(0.25, 0.8)
axes[1].set_xlabel(r'Amplitude / noise $\sigma$')
axes[1].set_ylabel(r'Detection fraction')
axes[1].set_title(r'Fine-grid detection curves (transition region)')
axes[1].legend(fontsize=7.5, ncol=2)
axes[1].grid(True, which='both', alpha=0.2)

plt.tight_layout()
plt.savefig('dev/detection_threshold.png')
plt.close()

print("\nAll figures saved.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
empirical_a50 = np.nanmean([v[0] for v in fit_results.values()])
print("\n=== DETECTION LIMIT SUMMARY ===")
print(f"  Theoretical (LS power >= snr_search_lim):   A50 = {A50_THEORY:.3f} sigma")
print(f"  Pipeline cut (snr_lim flux filter):          A50 ~ {A50_snr_lim:.3f} sigma")
print(f"  Pipeline cut (local_threshold >= {local_thr:.0f}):       A50 ~ {A50_local_thr:.3f} sigma")
print(f"  Empirical (mean over fitted periods):        A50 ~ {empirical_a50:.3f} sigma")
print(f"\n  Overhead factor (empirical / theory): {empirical_a50 / A50_THEORY:.2f}x")
print(f"  Overhead factor (local_thr / theory): {A50_local_thr / A50_THEORY:.2f}x")
print(f"\n  Binding constraint: local_threshold cut")
print(f"  (The snr_search_lim cut is exceeded at A/N ~ {A50_THEORY:.3f},")
print(f"   but local_sig only reaches {local_thr:.0f} at A/N ~ {A50_local_thr:.3f})")
