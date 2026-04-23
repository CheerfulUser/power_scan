"""Generate README figures for power_scan."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
import power_scan as ps

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
    'savefig.bbox': 'tight',
})

OUTDIR = 'docs/figures'

# ---------------------------------------------------------------------------
# Synthetic dataset used by all figures
# ---------------------------------------------------------------------------
rng = np.random.default_rng(123)
time = np.linspace(0, 200, 400)
ny, nx = 40, 40
data = rng.normal(100, 1, (len(time), ny, nx)).astype(float)

injected_period = 15.0
injected_amp    = 500.0   # large enough that per-pixel SNR stays high after PSF broadening
psf_sigma       = 2.0     # Gaussian PSF width in pixels (FWHM ~ 4.7 px)
src_y, src_x    = 20, 20

# Measure the peak fraction of the PSF kernel so the LC amplitude is known
_kernel_test = np.zeros((ny, nx))
_kernel_test[src_y, src_x] = 1.0
psf_peak_fraction = gaussian_filter(_kernel_test, sigma=psf_sigma)[src_y, src_x]

# Inject variable source as a delta-function time series then convolve with PSF
source_cube = np.zeros((len(time), ny, nx))
source_cube[:, src_y, src_x] = injected_amp * np.sin(2 * np.pi * time / injected_period)
for i in range(len(time)):
    source_cube[i] = gaussian_filter(source_cube[i], sigma=psf_sigma)
data += source_cube

# Add static field stars (constant brightness, PSF-convolved)
static_stars = [
    (8,  6,  300.0),
    (31, 12, 500.0),
    (10, 30, 250.0),
    (33, 28, 420.0),
    (18, 35, 180.0),
    (25,  7, 350.0),
]
for sy, sx, sflux in static_stars:
    stamp = np.zeros((ny, nx))
    stamp[sy, sx] = sflux
    psf_stamp = gaussian_filter(stamp, sigma=psf_sigma)
    data += psf_stamp[np.newaxis, :, :]   # broadcast across all time frames

print('Running pipeline ...')
det = ps.periodogram_detection(
    time, data,
    snr_lim=3,
    local_threshold=2,
    snr_search_lim=8,
    period_lim=[5.0, 100.0],
    cpu=1,
    run=True,
)
print(f'  {len(det.sources)} source(s) detected.')

# Dense time array for smooth phase fold in the pipeline figure
# Use the central-pixel amplitude (injected_amp * psf_peak_fraction)
lc_amp = injected_amp * psf_peak_fraction
rng2 = np.random.default_rng(123)
time_dense = np.linspace(0, 200, 3000)
flux_dense = rng2.normal(0, 1, len(time_dense))
flux_dense += lc_amp * np.sin(2 * np.pi * time_dense / injected_period)

freq_det   = det.sources['freq'].iloc[0]
period_det = 1.0 / freq_det
phase_dense = ((time_dense - time_dense[0]) * freq_det) % 1

phase_bin    = 0.01
bins         = np.arange(0, 1 + phase_bin, phase_bin)
bin_centers  = (bins[:-1] + bins[1:]) / 2
bin_flux     = np.array([
    np.median(flux_dense[(phase_dense >= bins[i]) & (phase_dense < bins[i + 1])])
    for i in range(len(bins) - 1)
])


# ===========================================================================
# Figure 1 — Pipeline overview  (clean redesign)
# ===========================================================================
fig = plt.figure(figsize=(16, 4.2))
fig.patch.set_facecolor('white')

# --- Stacked input frames (manually positioned) ----------------------------
frame_w  = 0.140
frame_h  = 0.68
frame_dx = 0.020
frame_dy = 0.020
base_x   = 0.040
base_y   = 0.16

# Pick frames at peak, trough and mid-phase to show variability
t0 = time[0]
frame_times = [
    np.argmin(np.abs(time - (t0 + injected_period * 0.25))),   # source at max
    np.argmin(np.abs(time - (t0 + injected_period * 0.75))),   # source at min
    np.argmin(np.abs(time - (t0 + injected_period * 0.00))),   # source at mean
]
# Stretch centred on background, wide enough to show ±1 PSF-peak amplitude
vmin_disp = 100 - 1.5 * lc_amp
vmax_disp = 100 + 1.5 * lc_amp

frame_axes  = []
for k in range(3):
    ax_f = fig.add_axes([
        base_x + k * frame_dx,
        base_y + k * frame_dy,
        frame_w, frame_h,
    ])
    ax_f.imshow(data[frame_times[k]], origin='lower', cmap='gray',
                vmin=vmin_disp, vmax=vmax_disp, aspect='equal')
    for sp in ax_f.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color('0.5')
    frame_axes.append(ax_f)

# Back two frames: no tick labels
for ax_f in frame_axes[:2]:
    ax_f.set_xticks([])
    ax_f.set_yticks([])

# Front frame: minimal ticks + labels + source marker + title
ax_front = frame_axes[2]
ax_front.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax_front.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax_front.tick_params(length=3, pad=2)
ax_front.set_xlabel(r'$x$ (px)', labelpad=3)
ax_front.set_ylabel(r'$y$ (px)', labelpad=3)
ax_front.set_title(r'\textbf{Input images}' + f'\n({len(time)} frames)', pad=5)
ax_front.scatter([src_x], [src_y], s=60, marker='o',
                 edgecolors='cyan', facecolors='none', linewidths=1.4, zorder=5)


# --- Power image ----------------------------------------------------------
ax_pow = fig.add_axes([0.340, base_y, 0.155, frame_h])
peak_ind = int(det.sources['power_ind'].iloc[0])
im_pow   = ax_pow.imshow(det.power_norm[peak_ind], origin='lower',
                          cmap='inferno', aspect='equal')
ax_pow.scatter([src_x], [src_y], s=60, marker='o',
               edgecolors='cyan', facecolors='none', linewidths=1.4, zorder=5)
ax_pow.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax_pow.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax_pow.tick_params(length=3, pad=2)
ax_pow.set_xlabel(r'$x$ (px)', labelpad=3)
ax_pow.set_ylabel(r'$y$ (px)', labelpad=3)
f_peak = det.freq[peak_ind]
ax_pow.set_title(r'\textbf{Power image}' + '\n' +
                 r'($f = {:.4f}$'.format(f_peak) + r'$\,\mathrm{d}^{-1}$)',
                 pad=5)

divider = make_axes_locatable(ax_pow)
cax = divider.append_axes('right', size='5%', pad=0.06)
cb  = fig.colorbar(im_pow, cax=cax)
cb.set_label('SNR', labelpad=4)
cb.ax.tick_params(length=3, pad=2)

# --- Phase-folded light curve ---------------------------------------------
ax_ph = fig.add_axes([0.580, base_y, 0.390, frame_h])
ax_ph.plot(phase_dense, flux_dense, '.', color='steelblue',
           ms=1.2, alpha=0.12, rasterized=True, zorder=1)
ax_ph.plot(bin_centers, bin_flux, 'o-', color='C1',
           ms=3.5, lw=1.6, zorder=5, label='Binned')
ax_ph.axhline(0, color='0.5', lw=0.6, ls='--', zorder=0)
ax_ph.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax_ph.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_ph.tick_params(length=3, pad=2)
ax_ph.set_xlabel(r'Phase', labelpad=3)
ax_ph.set_ylabel(r'Counts $-$ mean', labelpad=3)
ax_ph.set_title(r'\textbf{Phase-folded light curve}' +
                r' ($P = {:.1f}$\,d)'.format(period_det), pad=5)
ax_ph.legend(fontsize=8, loc='upper right', handlelength=1.5)

fig.savefig(f'{OUTDIR}/pipeline.png', bbox_inches='tight')
plt.close(fig)
print('  Saved pipeline.png')


# ===========================================================================
# Figure 2 — Full detection summary (4-panel)
# ===========================================================================
i = 0
fig2, axes = plt.subplot_mosaic(
    '''AAII
       BBII
       CCCC''',
    figsize=(11, 8),
    gridspec_kw={'hspace': 0.45, 'wspace': 0.35},
)

freq_s     = det.sources['freq'].iloc[i]
period_s   = 1.0 / freq_s

# A: raw light curve
t = det.lcs[i][0]
f = det.lcs[i][1]
axes['A'].plot(t, f, '.', color='steelblue', ms=2, alpha=0.7)
axes['A'].set_xlabel(r'Time (d)')
axes['A'].set_ylabel(r'Counts')
axes['A'].set_title(r'Light curve')
axes['A'].xaxis.set_major_locator(MaxNLocator(nbins=5))

# B: phase fold
axes['B'].plot(det.phase[i][0], det.phase[i][1], '.', color='steelblue',
               alpha=0.15, ms=2.5, rasterized=True)
axes['B'].plot(det.binned[i][0], det.binned[i][1], 'o-', color='C1',
               ms=4, lw=1.8, label='Binned')
axes['B'].set_xlabel(r'Phase')
axes['B'].set_ylabel(r'Counts')
axes['B'].set_title(r'Phase fold ($P = {:.2f}$\,d)'.format(period_s))
axes['B'].legend(fontsize=9)

# C: power spectrum
pfreq  = det.source_power_norm[i][0]
pnorm  = det.source_power_norm[i][1]
axes['C'].semilogx(1.0 / pfreq, pnorm, '-', color='steelblue', lw=1, label='SNR')
axes['C'].axvline(period_s, color='C1', ls='--', lw=1.5,
                  label=r'$P = {:.2f}$\,d'.format(period_s))
axes['C'].set_xlabel(r'Period (d)')
axes['C'].set_ylabel(r'SNR power')
axes['C'].set_title(r'Power spectrum')
axes['C'].legend(fontsize=9)

# I: spatial cutout
cut_rad  = 5
gind     = int(det.sources['power_ind'].iloc[i])
im_slice = det.power_norm[gind]
xc, yc   = det.sources['xcentroid'].iloc[i], det.sources['ycentroid'].iloc[i]
xi, yi   = int(round(xc)), int(round(yc))
xlo = max(xi - cut_rad, 0);  xhi = min(xi + cut_rad + 1, im_slice.shape[1])
ylo = max(yi - cut_rad, 0);  yhi = min(yi + cut_rad + 1, im_slice.shape[0])
cutout   = im_slice[ylo:yhi, xlo:xhi]
vmin, vmax = np.percentile(cutout, 5), np.percentile(cutout, 99)
img = axes['I'].imshow(cutout, origin='lower', cmap='inferno',
                        vmin=vmin, vmax=vmax)
axes['I'].scatter(xc - xlo, yc - ylo, s=80, marker='o',
                  edgecolors='cyan', facecolors='none', linewidths=1.5)
axes['I'].set_title(r'Spatial cutout (peak freq.)')
axes['I'].set_xlabel(r'$x$ (px)')
axes['I'].set_ylabel(r'$y$ (px)')
fig2.colorbar(img, ax=axes['I'], label='SNR', shrink=0.8)

fig2.suptitle(r'\textbf{power\_scan} --- detection summary',
              fontsize=13, fontweight='bold')
fig2.savefig(f'{OUTDIR}/detection_summary.png', bbox_inches='tight')
plt.close(fig2)
print('  Saved detection_summary.png')


# ===========================================================================
# Figure 3 — Power cube slices at three frequencies
# ===========================================================================
peak_ind  = int(det.sources['power_ind'].iloc[0])
low_ind   = max(peak_ind - len(det.freq) // 6, 0)
high_ind  = min(peak_ind + len(det.freq) // 6, len(det.freq) - 1)
indices   = [low_ind, peak_ind, high_ind]
labels    = [
    r'$f = {:.4f}$\,d$^{{-1}}$'.format(det.freq[low_ind]) + '\n(off-peak)',
    r'$f = {:.4f}$\,d$^{{-1}}$'.format(det.freq[peak_ind]) + '\n(peak --- detected)',
    r'$f = {:.4f}$\,d$^{{-1}}$'.format(det.freq[high_ind]) + '\n(off-peak)',
]

fig3, axs = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
vmax_g = np.percentile(det.power_norm[peak_ind], 99.5)
vmin_g = np.percentile(det.power_norm[peak_ind], 1)

for ax, idx, lbl in zip(axs, indices, labels):
    im = ax.imshow(det.power_norm[idx], origin='lower', cmap='inferno',
                   vmin=vmin_g, vmax=vmax_g, aspect='equal')
    if idx == peak_ind:
        ax.scatter([src_x], [src_y], s=80, marker='o',
                   edgecolors='cyan', facecolors='none', linewidths=1.8)
    ax.set_title(lbl, fontsize=10)
    ax.set_xlabel(r'$x$ (px)', labelpad=2)
    ax.set_ylabel(r'$y$ (px)', labelpad=2)
    fig3.colorbar(im, ax=ax, label='SNR', shrink=0.85)

fig3.suptitle(r'Normalised power spectrum cube --- three frequency slices',
              fontsize=12, fontweight='bold')
fig3.savefig(f'{OUTDIR}/power_cube_slices.png', bbox_inches='tight')
plt.close(fig3)
print('  Saved power_cube_slices.png')

print('Done.')
