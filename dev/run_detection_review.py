"""
Run power_scan on the test sector with weakened detection parameters (near
theoretical limit) and save one plot_object figure per detected source.

Usage
-----
    cd /Users/rri38/Documents/work/code/power_scan
    python dev/run_detection_review.py

After the run, inspect dev/detection_review/ and delete any figures that
correspond to false positives.  Then run the follow-up tuning script.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import power_scan as ps

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = 'data_test'
FLUX_FILE  = os.path.join(DATA_DIR, 'sector27_cam3_ccd2_cut1_of64_ReducedFlux.npy')
TIME_FILE  = os.path.join(DATA_DIR, 'sector27_cam3_ccd2_cut1_of64_Times.npy')
OUT_DIR    = 'dev/detection_review'

os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ───────────────────────────────────────────────────────────────
print('loading data …')
flux = np.load(FLUX_FILE)
time = np.load(TIME_FILE)
print(f'  flux shape : {flux.shape}  dtype: {flux.dtype}')
print(f'  time range : {time.min():.2f} – {time.max():.2f}  '
      f'(baseline {time.max()-time.min():.1f} d)')

# ── weakened detection parameters ───────────────────────────────────────────
# snr_search_lim : threshold for per-frequency "is it worth running sep?"
#   theoretical ~4 for N≈3000, set to 4 to stay near floor
# local_threshold : local significance inside sep annulus — drop to 3
# snr_lim : global power SNR cut — drop to 3
# sep peak threshold also needs to be low; this is passed as dao_peak
#   (used as sep threshold) — set to 8 (roughly 2× theory for 266×266 images)

PARAMS = dict(
    snr_lim         = 3,
    snr_search_lim  = 4,
    local_threshold = 3,
    dao_peak        = 8,          # sep extraction threshold (peak counts in bkg-sub map)
    aperture_radius = 1.5,
    fwhm            = 3,
    period_lim      = 'auto',     # auto → [2*cadence, T_obs/2] ≈ [0.014, 12.2] d
    psf_kernel      = None,
    block_size      = 133,        # process in 133×133 spatial blocks to limit RAM
    run             = False,      # we call run() manually for visibility
)

print('\nrunning pipeline with weakened parameters …')
print('  ' + '  '.join(f'{k}={v}' for k, v in PARAMS.items() if k != 'run'))

pipe = ps.periodogram_detection(time, flux, **PARAMS)
pipe.run()

if pipe.sources is None or len(pipe.sources) == 0:
    print('No sources detected — try loosening parameters further.')
    sys.exit(0)

n = len(pipe.sources)
print(f'\n{n} source(s) detected.')
print(pipe.sources[['xcentroid', 'ycentroid', 'freq', 'local_sig',
                     'phase_coherence', 'harmonic_ratio',
                     'odd_even_asymmetry']].to_string())

# ── save one figure per source ───────────────────────────────────────────────
print(f'\nsaving figures to {OUT_DIR}/ …')
pipe.make_lcs()

for i in range(n):
    row   = pipe.sources.iloc[i]
    freq  = row['freq']
    period = 1.0 / freq
    x     = row['xcentroid']
    y     = row['ycentroid']
    fname = f'source_{i:03d}_P{period:.3f}d_x{x:.0f}y{y:.0f}.png'
    fpath = os.path.join(OUT_DIR, fname)

    pipe.plot_object(index=i, savepath=OUT_DIR)

    # plot_object saves as var_{i}.png — rename to descriptive name
    default_path = os.path.join(OUT_DIR, f'var_{i}.png')
    if os.path.exists(default_path):
        os.rename(default_path, fpath)

    print(f'  [{i:03d}]  P={period:.3f} d  ({x:.1f}, {y:.1f})  → {fname}')
    plt.close('all')

# ── save source table ────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, 'sources.csv')
pipe.sources.to_csv(csv_path, index=True, index_label='source_index')
print(f'\nsource table saved to {csv_path}')
print('\nDone.  Delete false-positive figures from detection_review/ then run the tuning step.')
