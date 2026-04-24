"""
Build the power/power_norm cube once and save to disk.
Only needs to be run once per dataset.

Usage:
    cd /Users/rri38/Documents/work/code/power_scan
    python dev/build_power_cube.py
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import power_scan as ps

FLUX_FILE  = 'data_test/sector27_cam3_ccd2_cut1_of64_ReducedFlux.npy'
TIME_FILE  = 'data_test/sector27_cam3_ccd2_cut1_of64_Times.npy'
CUBE_DIR   = 'dev/cube_cache'

os.makedirs(CUBE_DIR, exist_ok=True)

print('loading data...')
flux = np.load(FLUX_FILE)
time = np.load(TIME_FILE)

pipe = ps.periodogram_detection(time, flux,
    snr_lim=3, snr_search_lim=4, local_threshold=3, dao_peak=8,
    aperture_radius=1.5, fwhm=3, period_lim='auto',
    psf_kernel=None, block_size=133, run=False)

pipe.clean_data()
print('building power cube...')
pipe.block_make_freq_cube()

print('saving...')
np.save(f'{CUBE_DIR}/power_norm.npy', pipe.power_norm.astype(np.float32))
np.save(f'{CUBE_DIR}/freq.npy',       pipe.freq)
np.save(f'{CUBE_DIR}/time.npy',       pipe.time)

print(f'saved to {CUBE_DIR}/')
print(f'  power_norm : {pipe.power_norm.shape}  (float32)')
print(f'  freq       : {pipe.freq.shape}  ({pipe.freq.min():.4f} – {pipe.freq.max():.4f} d⁻¹)')
