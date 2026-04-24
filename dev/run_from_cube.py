"""
Run detection + figure saving from a pre-built power cube.
Fast iteration — skips the expensive periodogram step.

Usage:
    cd /Users/rri38/Documents/work/code/power_scan
    python dev/run_from_cube.py

Edit PARAMS below to tune detection thresholds, then re-run instantly.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import power_scan as ps

CUBE_DIR  = 'dev/cube_cache'
OUT_DIR   = 'dev/detection_review'

# ── detection parameters to tune ────────────────────────────────────────────
PARAMS = dict(
    dao_peak            = 8,
    local_threshold     = 8,     # local annulus SNR cut (was 3; analysis → 8)
    snr_lim             = 3,
    snr_search_lim      = 4,
    sep_wh_ratio        = 0.5,
    phase_coherence_lim    = 0.70,  # Rayleigh R cut (0–1)
    period_max_frac        = 1.06,  # max period = period_max_frac * T_obs/2  (≈12.9 d here)
    odd_even_asymmetry_lim = 0.30,  # eclipsing-binary fold asymmetry upper limit
)

# ── load cube ────────────────────────────────────────────────────────────────
FLUX_FILE = 'data_test/sector27_cam3_ccd2_cut1_of64_ReducedFlux.npy'
TIME_FILE = 'data_test/sector27_cam3_ccd2_cut1_of64_Times.npy'

print('loading cube...')
power_norm = np.load(f'{CUBE_DIR}/power_norm.npy')
freq       = np.load(f'{CUBE_DIR}/freq.npy')
time       = np.load(f'{CUBE_DIR}/time.npy')
data       = np.load(FLUX_FILE)
print(f'  power_norm: {power_norm.shape}   freq: {freq.shape}')

# ── build a pipeline object and inject the pre-built cube ───────────────────
pipe = ps.periodogram_detection(time, data,
    snr_lim             = PARAMS['snr_lim'],
    snr_search_lim      = PARAMS['snr_search_lim'],
    local_threshold     = PARAMS['local_threshold'],
    dao_peak            = PARAMS['dao_peak'],
    phase_coherence_lim    = PARAMS['phase_coherence_lim'],
    period_max_frac        = PARAMS['period_max_frac'],
    odd_even_asymmetry_lim = PARAMS['odd_even_asymmetry_lim'],
    aperture_radius=1.5, fwhm=3, period_lim='auto',
    psf_kernel=None, run=False)

pipe.power_norm = power_norm
pipe.power      = power_norm  # power not cached; power_norm used as stand-in for plotting
pipe.freq       = freq
# period limits (needed by find_fundamental)
dt = np.median(np.diff(time))
pipe._period_low  = 2 * dt
pipe._period_high = (time.max() - time.min()) / 2.0

print('finding sources...')
pipe.find_freq_sources()
pipe.detection_cleaning()

if pipe.sources is None or len(pipe.sources) == 0:
    print('No sources detected.')
    sys.exit(0)

pipe.find_peak_power()
pipe.find_fundamental()
pipe.measure_phase_coherence()
pipe.measure_harmonic_power()
pipe.measure_odd_even_asymmetry()

n = len(pipe.sources)
print(f'{n} source(s) detected.')

# ── save figures ─────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
# clear old pngs
for f in os.listdir(OUT_DIR):
    if f.endswith('.png'):
        os.remove(os.path.join(OUT_DIR, f))

pipe.make_lcs()
for i in range(n):
    row    = pipe.sources.iloc[i]
    period = 1.0 / row['freq']
    x, y   = row['xcentroid'], row['ycentroid']
    fname  = f'source_{i:03d}_P{period:.3f}d_x{x:.0f}y{y:.0f}.png'
    pipe.plot_object(index=i, savepath=OUT_DIR)
    default = os.path.join(OUT_DIR, f'var_{i}.png')
    if os.path.exists(default):
        os.rename(default, os.path.join(OUT_DIR, fname))
    plt.close('all')

pipe.sources['period'] = 1 / pipe.sources['freq']
pipe.sources.to_csv(f'{OUT_DIR}/sources.csv', index=True, index_label='source_index')
print(f'figures and sources.csv saved to {OUT_DIR}/')
print('Done.')
