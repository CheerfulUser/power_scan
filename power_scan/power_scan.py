import numpy as np
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.signal import find_peaks
from photutils.detection import DAOStarFinder
import pandas as pd
import nifty_ls
from joblib import Parallel, delayed 
import matplotlib.pyplot as plt
from copy import deepcopy
import sep
# import warnings
# from photutils.utils import NoDetectionsWarning

# warnings.filterwarnings("ignore", category=NoDetectionsWarning)


def _local_sig(source,image,threshold=10,sky_in=5,sky_out=10):
    from photutils.aperture import RectangularAnnulus, ApertureStats
    source = source.reset_index(drop=True)
    good = []
    source['local_sig'] = 0.0
    for i in range(len(source)):
        x = source.xcentroid.iloc[i]
        y = source.ycentroid.iloc[i]
        xint = int(np.round(x,0))
        yint = int(np.round(y,0))
        
        annulus_aperture = RectangularAnnulus([xint,yint], w_in=5, w_out=10,h_out=10)

        aperstats_sky = ApertureStats(image, annulus_aperture)
        
        sig = (source.flux.iloc[i]-aperstats_sky.mean) / (source.npix.iloc[i]*aperstats_sky.std)
        source.loc[i,'local_sig'] = sig
        if sig >= threshold:
            good += [i]
    source = source.iloc[good]
    return source



def _detect_sources(frequency,power,index=None,peak=50,fwhm=3,method='sep',
                    local_threshold=10,sep_wh_ratio=0.5,psf_kernel=None):
    import warnings
    from photutils.utils import NoDetectionsWarning
    warnings.filterwarnings("ignore", category=NoDetectionsWarning)
    p = power.astype(float)
    if method.lower() == 'dao':
        finder = DAOStarFinder(peak,fwhm,exclude_border=True,min_separation=3)
        s = finder.find_stars(p)
        if s is not None:
            s = s.to_pandas()
    elif method.lower() == 'sep':
        bkg = sep.Background(p)
        sep_kwargs = {}
        if psf_kernel is not None:
            sep_kwargs['filter_kernel'] = np.asarray(psf_kernel, dtype=np.float32)
        objects = sep.extract(p, peak, **sep_kwargs)
        objects = pd.DataFrame(objects)
        w = objects['a'].values
        h = objects['b'].values
        f = objects['flux'].values
        ratio = np.round(abs(w/h - 1),1)
        ind = (ratio < sep_wh_ratio)
        s = objects.loc[ind].reset_index(drop=True)
        s = s.rename(columns={'x':'xcentroid','y':'ycentroid'})
        s['id'] = np.arange(1,len(s)+1)

    else:
        raise ValueError('method must either be sep or dao.')
    if (s is not None):
        s = _local_sig(s,power,local_threshold)
        if (len(s) > 0):
            s['freq'] = frequency
            if index is not None:
                s['power_ind'] = int(index)
            return s 
        else:
            return None
    else:
        return None


def _refine_peak_freq(freqs, power_norm_1d, peak_idx, half_width=2):
    """
    Sub-bin peak refinement by fitting a Gaussian to the 5 points centred on
    peak_idx in a 1-D power spectrum slice.

    Returns (refined_freq, snap_idx) where refined_freq is the Gaussian centre
    and snap_idx is the nearest grid index (used to keep power_ind consistent).
    Falls back to (freqs[peak_idx], peak_idx) if the fit fails or there are
    too few points.
    """
    from scipy.optimize import curve_fit

    lo = max(0, peak_idx - half_width)
    hi = min(len(freqs) - 1, peak_idx + half_width)
    if hi - lo < 4:
        return freqs[peak_idx], peak_idx

    f = freqs[lo:hi + 1]
    p = power_norm_1d[lo:hi + 1]

    try:
        popt, _ = curve_fit(
            lambda x, A, f0, sig: A * np.exp(-0.5 * ((x - f0) / sig) ** 2),
            f, p,
            p0=[p.max(), freqs[peak_idx], (f[-1] - f[0]) / 4.0],
            bounds=([0, f[0], 0], [np.inf, f[-1], f[-1] - f[0]]),
        )
        refined_freq = float(popt[1])
        snap_idx     = int(np.argmin(np.abs(freqs - refined_freq)))
        return refined_freq, snap_idx
    except Exception:
        return freqs[peak_idx], peak_idx


def _odd_even_asymmetry(time, flux, freq, n_bins=20):
    """
    Odd-even half-cycle test for eclipsing binaries.

    Folds the light curve at 2/freq (doubled period) and compares the shape
    of the first half-cycle (phase 0-0.5) against the second (phase 0.5-1.0).
    For an eclipsing binary the two halves contain different eclipses and are
    asymmetric; for a sinusoidal variable or RR Lyrae the two halves are
    mirror images with the same amplitude.

    Returns the asymmetry score A = RMS(first_half - second_half) / amplitude,
    where amplitude is the peak-to-peak range of the 2P-folded, binned curve.
    A close to 0 means the two halves are identical (symmetric variable or too
    few cycles to tell).  A > ~0.3 is a strong indicator that the true period
    is 2× the supplied freq.

    Returns NaN when there are too few observations per bin to be reliable.
    """
    phase = ((time - time[0]) * (freq / 2.0)) % 1.0

    bins = np.linspace(0, 1, n_bins + 1)
    half = n_bins // 2
    first, second = [], []
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if mask.sum() < 2:
            (first if i < half else second).append(np.nan)
        else:
            (first if i < half else second).append(np.median(flux[mask]))

    first  = np.array(first)
    second = np.array(second)

    # Mean-centre each half independently so depth offsets don't dominate
    first  -= np.nanmedian(first)
    second -= np.nanmedian(second)

    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < max(3, half // 2):
        return np.nan

    full = np.concatenate([first, second])
    amplitude = np.nanmax(full) - np.nanmin(full)
    if amplitude == 0:
        return 0.0

    return float(np.sqrt(np.nanmean((first[valid] - second[valid]) ** 2)) / amplitude)


def _phase_coherence(time, data, x, y, freq, radius=2.0):
    """
    Rayleigh R statistic for spatial phase coherence at position (x, y) and
    frequency freq.  Fits a sinusoid phase independently at every pixel within
    the aperture and returns the mean resultant vector length R ∈ [0, 1].
    R ≈ 1 means all pixels oscillate in phase (genuine point source);
    R ≈ 0 means phases are random (noise spike).
    """
    ny, nx = data.shape[1], data.shape[2]
    xi, yi = int(round(x)), int(round(y))
    r = int(np.ceil(radius))
    t = time - time[0]
    omega = 2 * np.pi * freq

    phases = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if np.hypot(dx, dy) > radius:
                continue
            px, py = xi + dx, yi + dy
            if 0 <= px < nx and 0 <= py < ny:
                lc = data[:, py, px].astype(float)
                lc -= lc.mean()
                c = np.dot(lc, np.cos(omega * t))
                s = np.dot(lc, np.sin(omega * t))
                phases.append(np.arctan2(s, c))

    if len(phases) < 2:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * np.array(phases)))))


def _harmonic_power(freq, power_norm, freq_array, x, y, radius=1.5):
    """
    Return the mean normalised power at the first harmonic (2 * freq) within
    a circular aperture of the given radius centred on (x, y).
    Returns NaN when the harmonic falls outside the frequency grid.
    """
    h_freq = 2.0 * freq
    if h_freq > freq_array.max() or h_freq < freq_array.min():
        return np.nan

    h_idx = int(np.argmin(np.abs(freq_array - h_freq)))
    ny, nx = power_norm.shape[1], power_norm.shape[2]
    xi, yi = int(round(x)), int(round(y))
    r = int(np.ceil(radius))

    vals = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if np.hypot(dx, dy) > radius:
                continue
            px, py = xi + dx, yi + dy
            if 0 <= px < nx and 0 <= py < ny:
                vals.append(power_norm[h_idx, py, px])

    return float(np.mean(vals)) if vals else np.nan


def _Spatial_group(result,min_samples=1,distance=1,njobs=-1,write_col='objid'):
    """
    Groups events based on proximity.
    """

    from sklearn.cluster import DBSCAN

    pos = np.array([result.xcentroid,result.ycentroid]).T
    cluster = DBSCAN(eps=distance,min_samples=min_samples,n_jobs=njobs).fit(pos)
    labels = cluster.labels_
    unique_labels = set(labels)
    for label in unique_labels:
        result.loc[label == labels,write_col] = label + 1
    result[write_col] = result[write_col].astype(int)
    return result
 
def compress_freq_groups(result):
    inds = []
    for i in result['objid'].unique():
        obj_ind = result['objid'].values==i
        obj = result.iloc[obj_ind]
        ind = np.argmax(obj['local_sig'].values)
        inds += [np.where(obj_ind)[0][ind]]
    inds = np.array(inds)
    final = result.iloc[inds]
    return final


def Generate_LC(time,flux,x,y,frame_start=None,frame_end=None,method='sum',
                radius=1.5):
    
    from photutils.aperture import CircularAperture, RectangularAnnulus, ApertureStats, aperture_photometry
    from scipy.signal import fftconvolve
    
    t = time
    f = flux

    if frame_start is not None:
        if frame_end is not None:
            t = t[frame_start:frame_end+1]
            f = f[frame_start:frame_end+1]
        else:
            t = t[frame_start:]
            f = f[frame_start:]   
    elif frame_end is not None:
        t = t[:frame_end+1]
        f = f[:frame_end+1]     

    if method.lower() == 'aperture':
        aperture = CircularAperture([x, y], radius)
        annulus_aperture = RectangularAnnulus([x,y], w_in=5, w_out=20,h_out=20)
        flux = []
        flux_err = []
        for i in range(len(f)):
            m = sigma_clip(f[i],masked=True,sigma=5).mask
            mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
            aperstats_sky = ApertureStats(f[i], annulus_aperture,mask = mask)
            phot_table = aperture_photometry(f[i], aperture)
            bkg_std = aperstats_sky.std
            flux_err += [aperture.area * bkg_std]
            flux += [phot_table['aperture_sum'].value[0]]
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        return t, flux, flux_err
    elif method.lower() == 'sum':
        xint = int(np.round(x,0))
        yint = int(np.round(y,0))
        buffer = np.floor(radius).astype(int)
        f = np.nansum(f[:,yint-buffer:yint+buffer+1,xint-buffer:xint+buffer+1],axis=(1,2))

        return t,f

def run_reg_ls(lc, max_freq=None):
    t,f = lc
    import nifty_ls
    frequency, power = LombScargle(t, f, np.ones_like(f)).autopower(method="fastnifty",maximum_frequency=max_freq,
                                                                    nyquist_factor=1.5,
                                                                    samples_per_peak=5)
    return [frequency, power]



class periodogram_detection():
    def __init__(self,time,data,error=None,aperture_radius=1.5,
                 snr_lim=5,fwhm=3,dao_peak=20,cpu=-1,snr_search_lim=10,
                 period_lim='auto',block_size=None,edge_buffer=0,detection_method='sep',
                 local_threshold=10,savepath=None,psf_kernel=None,
                 phase_coherence_lim=None,period_max_frac=None,
                 odd_even_asymmetry_lim=None,
                 run=True):
        """
        Detect faint variable objects in time-series image data.

        Parameters
        ----------
        time : array
            times of observations
        data : array
            3 dimensional data array where the first dimension is time, and the remaining are the spatial dimensions

        Options
        -------
        error : array, optional
            Currently not implemented

        aperture_radius : float
            Size of the aperture to perform photometry

        snr_lim : float
            minimum accepted SNR for a detected source in the power domain

        fwhm : float
            size of the fwhm used in DAOstarfinder

        dao_peak : float
            Minimum peak detection threshold for DAOstarfinder

        cpu : int
            Number of cores for parallel procession

        snr_search_lim : float 
            At least 1 pixel in the power image must pass this limit for source detection to be run on a frequency

        period_lim : str, or array
            Limits set for the detection of sources. If 'auto' then the default values of low = 2*sampling rate, high = 1/2*observing window.
            Set custom limits by inputting either a list or an array with 2 elements.

        run : bool
            Run the reduction 
    
        """
        self.time = time
        self.data = data
        self.error = error

        # operating params
        self.snr_lim = snr_lim
        self.fwhm = fwhm
        self.dao_peak = dao_peak
        self.cpu = cpu
        self.aperture_radius = aperture_radius
        self.snr_search_lim = snr_search_lim
        self.period_lim = period_lim
        self.savepath = savepath
        self.savename = None
        self.block_size = block_size
        self.edge_buffer = edge_buffer
        self.detection_method = detection_method
        self.local_threshold = local_threshold
        self.psf_kernel = psf_kernel
        self.phase_coherence_lim = phase_coherence_lim
        self.period_max_frac = period_max_frac
        self.odd_even_asymmetry_lim = odd_even_asymmetry_lim

        # calculated
        self.freq = None
        self.power = None
        self.detections = None
        self.sources = None
        self.lcs = None
        self.phase = None
        self._period_low = None
        self._period_high = None
        self.source_power = None
        self.source_power_norm = None

        if run:
            self.run()
    


    
    def clean_data(self):
        good = np.where(np.isfinite(np.sum(self.data,axis=(1,2))))
        self.data = self.data[good]
        #self.error = self.error[good]
        self.time = self.time[good]
        
        ind = np.argsort(self.time)
        self.time = self.time[ind]
        self.data = self.data[ind]
    
    def _set_period_lim(self):
        if isinstance(self.period_lim, str) and self.period_lim == 'auto':
            self._period_low = np.median(np.diff(self.time)) * 2 
            print(self.time.shape)
            self._period_high = (self.time[-1] - self.time[0]) / 1.5 
        else:
            try:
                self._period_low = np.min(self.period_lim)
                self._period_high = np.max(self.period_lim)
            except:
                m = "period_lim must either be 'auto', or arraylike with 2 elements."
                raise ValueError(m)
        

    def block_make_freq_cube(self):
        if self._period_low is None:
            self._set_period_lim()

        dx = np.arange(0,self.data.shape[2]+self.block_size,self.block_size)
        dy = np.arange(0,self.data.shape[1]+self.block_size,self.block_size)
        power_blocks = []

        temp = nifty_ls.lombscargle(self.time-self.time[0],self.data[:,0,0],
                                               fmin=(1/self._period_high),fmax=(1/self._period_low),nterms=1)
        freq = temp.freq()
        power = np.zeros((len(freq),self.data.shape[1],self.data.shape[2]))
        for i in range(len(dy)-1):
            for j in range(len(dx)-1):
                cut = self.data[:,dy[i]:dy[i+1],dx[j]:dx[j+1]]
                shaped = cut.reshape(len(cut),cut.shape[1]*cut.shape[2]).T
                batched = nifty_ls.lombscargle(self.time-self.time[0],shaped,
                                               fmin=1/self._period_high,fmax=1/self._period_low,nterms=1)
                power_block = batched.power.T.reshape(batched.power.shape[1],cut.shape[1],cut.shape[2])
                power[:,dy[i]:dy[i+1],dx[j]:dx[j+1]] = power_block
        
        self.power = power
        self.freq = freq
        m,med,std = sigma_clipped_stats(self.power,axis=(1,2))
        self.power_norm = (self.power-med[:,np.newaxis,np.newaxis]) / std[:,np.newaxis,np.newaxis]
        
        # if self._period_low is None:
        #     self._set_period_lim()
        ind = (self.freq < 1/self._period_low) & (self.freq > 1/self._period_high)
        self.power = self.power[ind]
        self.freq = self.freq[ind]
        self.period = 1/self.freq
        self.power_norm = self.power_norm[ind]


    def batch_make_freq_cube(self):
        if self._period_low is None:
            self._set_period_lim()
        
        shaped = self.data.reshape(len(self.data),self.data.shape[1]*self.data.shape[2]).T
        batched = nifty_ls.lombscargle(self.time-self.time[0],shaped,
                                       fmin=1/self._period_high,fmax=1/self._period_low,nterms=1)
        self.power = batched.power.T.reshape(batched.power.shape[1],self.data.shape[1],self.data.shape[2])
        self.freq = batched.freq()
        m,med,std = sigma_clipped_stats(self.power,axis=(1,2))
        self.power_norm = (self.power-med[:,np.newaxis,np.newaxis]) / std[:,np.newaxis,np.newaxis]
        
        # if self._period_low is None:
        #     self._set_period_lim()
        ind = (self.freq < 1/self._period_low) & (self.freq > 1/self._period_high)
        self.power = self.power[ind]
        self.freq = self.freq[ind]
        self.period = 1/self.freq
        self.power_norm = self.power_norm[ind]

    def loky_make_freq_cube(self):
        import multiprocessing
        if self._period_low is None:
            self._set_period_lim()

        shaped = self.data.reshape(len(self.data),self.data.shape[1]*self.data.shape[2]).T

        results = Parallel(n_jobs=int(multiprocessing.cpu_count()),backend='loky')(delayed(run_reg_ls)(lc) for lc in shaped)




    def find_freq_sources(self,peak=None,fwhm=None):

        if fwhm is None:
            fwhm = self.fwhm
        if peak is None:
            peak = self.dao_peak
        ind = np.nanmax(self.power_norm,axis=(1,2)) >= self.snr_search_lim
        index = np.arange(0,len(self.freq))[ind]
        source = Parallel(n_jobs=self.cpu)(delayed(_detect_sources)(self.freq[i],deepcopy(self.power_norm[i]),i,peak,fwhm,self.detection_method,self.local_threshold,psf_kernel=self.psf_kernel) for i in index)
        sources = None
        if len(source) > 0:
            for s in source:
                if s is not None:
                    if sources is None:
                        sources = s
                    else:
                        sources = pd.concat([sources,s])
            self.detections = sources
        else:
            self.detections = None

    def _spatial_alias_clean(self,close_freq=1e-3,plot=False):
        sources = deepcopy(self.sources)
        groups = _Spatial_group(sources,distance=3,write_col='id2')
        ids = groups['id2'].unique()
        keep = pd.DataFrame([])
        for i in ids:
            group = groups.loc[groups['id2']==i]
            group.reset_index(drop=True, inplace=True)
            if len(group) > 1:
                mod1 = (group.freq.values[:,np.newaxis] / group.freq.values[np.newaxis,:]) % 1
                mod2 = 1/(group.freq.values[:,np.newaxis] / group.freq.values[np.newaxis,:]) % 1 # invert to get higher order alias
                # make a symmetric array
                sym = (mod1 < close_freq) | (mod2 < close_freq)
                unique = np.unique(sym,axis=0)
                for u in unique:
                    if np.sum(u) > 1:
                        snr = group.loc[u,'flux']
                        ind = np.argmax(snr)
                        adding = pd.DataFrame([group.loc[ind]]) # pandas is trash
                        keep = pd.concat([keep,adding], ignore_index=True)
                        if plot:
                            self.plot_object(index=group.objid.values-1)
                    else:
                        keep = pd.concat([keep,group.loc[u]], ignore_index=True)
            
            else:
                keep = pd.concat([keep,group], ignore_index=True)

        keep = keep.drop(['id2'],axis=1)
        keep = keep.rename(columns={'objid':'old_objdid','id':'objid'})
        keep['objid'] = np.arange(1,len(keep)+1,dtype=int)
        self.sources = keep

    def detection_cleaning(self,snr_lim=None):
        if snr_lim is None:
            snr_lim = self.snr_lim
        detect = self.detections
        if detect is not None:
            detect = detect.loc[detect['flux'] >= snr_lim]
            # Pre-filter by period before grouping so junk long-period detections
            # cannot out-compete real detections at the same spatial position.
            if self.period_max_frac is not None and self._period_high is not None:
                period_max = self._period_high * self.period_max_frac
                period_ok = (1.0 / detect['freq'].values) <= period_max
                if period_ok.any():
                    detect = detect.loc[period_ok]
            sources = _Spatial_group(detect)
            self.sources = compress_freq_groups(sources)
            self._spatial_alias_clean()

            if self.edge_buffer > 0:
                xind = (self.sources.xcentroid.values > self.edge_buffer) & (self.sources.xcentroid.values < self.data.shape[2] - self.edge_buffer)
                yind = (self.sources.ycentroid.values > self.edge_buffer) & (self.sources.ycentroid.values < self.data.shape[1] - self.edge_buffer)
                ind = xind & yind
                self.sources = self.sources.iloc[ind]
        else:
            self.sources = None


    def _make_lc(self,source,radius):
        x = source.xcentroid
        y = source.ycentroid
        t,lc = Generate_LC(time=self.time,flux=self.data,x=x,y=y,radius=radius)
        lc = np.array([t,lc])
        freq,l = Generate_LC(time=self.freq,flux=self.power_norm,x=x,y=y,radius=radius)
        source_power_norm = np.array([freq,l])
        freq,l = Generate_LC(time=self.freq,flux=self.power,x=x,y=y,radius=radius)
        source_power = np.array([freq,l])
        if np.nansum(abs(l)) > 0:
            good = True
        else:
            good = False
        
        return lc, source_power_norm,source_power, good

    def get_lightcurves(self,radius=None):
        if radius is None:
            radius = self.aperture_radius

        # lcs = []
        # source_power = []
        # good = []
        # for i in range(len(self.sources)):
        #     source = self.sources.iloc[i]
        #     x = source.xcentroid
        #     y = source.ycentroid
        #     t,l = Generate_LC(time=self.time,flux=self.data,x=x,y=y,radius=radius)
        #     lcs += [np.array([t,l])]
        #     t,l = Generate_LC(time=self.freq,flux=self.power_norm,x=x,y=y,radius=radius)
        #     if np.nansum(abs(l)) > 0:
        #         good += [i]
        #     source_power += [np.array([t,l])]
        index = np.arange(0,len(self.sources))
        lcs,source_power_norm,source_power,good = zip(*Parallel(n_jobs=self.cpu)(delayed(self._make_lc)(self.sources.iloc[i],radius) for i in index))
        self.lcs = np.array(lcs)
        self.source_power = np.array(source_power)
        self.source_power_norm = np.array(source_power_norm)
        good = np.array(good)
        self.sources = self.sources.iloc[good].reset_index(drop=True)
        self.lcs = np.array(lcs)[good]
        self.source_power = np.array(source_power)[good]
        self.source_power_norm = np.array(source_power_norm)[good]
        self.med_power = np.nanmedian(self.power,axis=(1,2))
        


    def phase_fold(self):
        phases = []
        for i in range(len(self.lcs)):
            freq = self.sources['freq'].iloc[i]
            lc = self.lcs[i]
            phase = lc[0] - lc[0,0]
            #phase = ((lc[0] - lc[0,0]) / (1/freq)) % 1
            phase = ((lc[0] - lc[0,0]) / (1/freq)) % 1
            phases += [np.array([phase,lc[1]])]
        phases = np.array(phases)
        self.phase = phases

    def bin_phase(self,phase_bin=0.01):
        binned = []
        bins = np.arange(0,1 + phase_bin,phase_bin)
        for phase in self.phase:
            av = []
            ps = []
            for i in range(len(bins) - 1):
                ind = (bins[i] <= phase[0]) & (bins[i+1] > phase[0])
                med = np.median(phase[1,ind])
                p = (bins[i] + bins[i+1]) / 2
                av += [med]
                ps += [p]
            av = np.array(av)
            ps = np.array(ps)
            lc = np.array([ps,av])
            binned += [lc]
        binned = np.array(binned)
        self.binned = binned

    def find_peak_power(self):
        from scipy.signal import find_peaks
        self.sources['power'] = np.nan
        if self.source_power is None:
            self.get_lightcurves()
        for i in range(len(self.lcs)):
            power = self.source_power[i,1] 
            power_norm = self.source_power_norm[i,1] 
            peaks, _ = find_peaks(power_norm, height=np.max(power_norm)*0.5)
            if len(peaks) == 0:
                peaks = np.array([int(np.argmax(power_norm))])

            p_ind = peaks[np.argmax(power[peaks])]

            self.sources.loc[i,'power_ind'] = p_ind
            self.sources.loc[i,'freq'] = self.freq[p_ind]
            self.sources.loc[i,'power'] = power[p_ind]

    def refine_centroids(self, half_width=4):
        """
        Re-centroid each source by fitting a 2D Gaussian to the power_norm slice
        at the peak frequency.  Replaces the sep flux-weighted centroid, which can
        be biased when a PSF filter kernel is used.
        """
        from scipy.optimize import curve_fit

        def gaussian2d(xy, A, x0, y0, sx, sy, C):
            x, y = xy
            return (A * np.exp(-0.5 * ((x - x0) / sx) ** 2
                               -0.5 * ((y - y0) / sy) ** 2) + C).ravel()

        ny, nx = self.power_norm.shape[1], self.power_norm.shape[2]
        for i in range(len(self.sources)):
            x0 = self.sources['xcentroid'].iloc[i]
            y0 = self.sources['ycentroid'].iloc[i]
            p_ind = int(self.sources['power_ind'].iloc[i])
            im = self.power_norm[p_ind]

            xi, yi = int(round(x0)), int(round(y0))
            xlo = max(0, xi - half_width)
            xhi = min(nx, xi + half_width + 1)
            ylo = max(0, yi - half_width)
            yhi = min(ny, yi + half_width + 1)
            cutout = im[ylo:yhi, xlo:xhi].astype(float)
            if cutout.size < 9:
                continue

            yg, xg = np.mgrid[ylo:yhi, xlo:xhi].astype(float)
            p0 = [cutout.max() - cutout.min(), x0, y0, 1.5, 1.5, cutout.min()]
            bounds_lo = [0, xlo, ylo, 0.3, 0.3, -np.inf]
            bounds_hi = [np.inf, xhi, yhi, half_width, half_width, np.inf]
            try:
                popt, _ = curve_fit(gaussian2d, (xg.ravel(), yg.ravel()),
                                    cutout.ravel(), p0=p0,
                                    bounds=(bounds_lo, bounds_hi), maxfev=2000)
                self.sources.loc[i, 'xcentroid'] = float(popt[1])
                self.sources.loc[i, 'ycentroid'] = float(popt[2])
            except Exception:
                pass

    def find_fundamental(self, odd_even_threshold=0.3):
        if self.lcs is None:
            self.make_lcs()
        for j in range(len(self.lcs)):
            freq = self.sources['freq'].iloc[j]
            lc = self.lcs[j]
            alias = np.array([1/2,1,2])
            grad_sum = []
            for i in range(len(alias)):
                phase = lc[0] - lc[0,0]
                phase = ((lc[0] - lc[0,0]) / (1/(freq*alias[i]))) % 1
                new_period = 1/(freq*alias[i])
                p = lc[1,np.argsort(phase)]
                metric = np.sum(abs(np.diff(p)))
                if new_period > (lc[0,-1]-lc[0,0])/1.5:
                    metric = 1e6
                grad_sum += [metric]
            grad_sum = np.array(grad_sum)
            ind = np.argmin(grad_sum)
            if ind != 1:
                m = abs(self.freq - freq * alias[ind])
                new_ind = np.argmin(m)
                if new_ind < len(self.source_power[j,1]):
                    new_power = self.source_power[j,1,new_ind]
                    ratio = new_power / self.sources['power'].iloc[j]
                    new_power = self.source_power_norm[j,1,new_ind]
                    old_power = self.source_power_norm[j,1,self.sources['power_ind'].iloc[j]]
                    ratio_n = new_power / old_power
                    if (ratio < 0.2) & (ratio_n < 0.2):
                        ind = 1
                else:
                    ind = 1

            # Odd-even check: the total-variation metric always prefers a
            # single-dip fold, so it will never spontaneously choose 2P for
            # an eclipsing binary. If the period was kept (ind==1), test
            # whether the 2P fold has asymmetric half-cycles — the hallmark
            # of unequal primary and secondary eclipses.
            if ind == 1:
                doubled_period = 2.0 / freq
                if self._period_high is not None and doubled_period <= self._period_high:
                    asym = _odd_even_asymmetry(lc[0], lc[1], freq)
                    if np.isfinite(asym) and asym > odd_even_threshold:
                        ind = 2

            # Snap to nearest grid point then refine with a 5-point Gaussian
            # fit so the stored frequency lands on the true sub-bin peak.
            snap_idx = int(np.argmin(np.abs(self.freq - freq * alias[ind])))
            refined_freq, peak_idx = _refine_peak_freq(
                self.freq, self.source_power_norm[j, 1], snap_idx)
            self.sources.loc[j, 'freq']      = refined_freq
            self.sources.loc[j, 'power_ind'] = peak_idx

        self.sources['period'] = 1/self.sources['freq'].values
        self.phase_fold()
        self.bin_phase()

    

        

        
    def measure_phase_coherence(self, radius=None):
        """
        Compute the Rayleigh R phase-coherence statistic for every source.

        Fits a sinusoid independently at each pixel within the aperture and
        measures how tightly the inferred phases cluster.  A genuine point
        source yields R close to 1; a noise spike in the power image yields
        R close to 0 because adjacent pixels are not driven by the same signal.

        Adds a 'phase_coherence' column (float, 0–1) to self.sources.
        """
        if radius is None:
            radius = self.aperture_radius
        if self.sources is None or len(self.sources) == 0:
            return
        r_vals = []
        for i in range(len(self.sources)):
            src = self.sources.iloc[i]
            r = _phase_coherence(self.time, self.data,
                                 float(src['xcentroid']), float(src['ycentroid']),
                                 float(src['freq']), radius)
            r_vals.append(r)
        self.sources = self.sources.copy()
        self.sources['phase_coherence'] = r_vals

    def measure_harmonic_power(self, radius=None):
        """
        Measure the normalised power at the first harmonic (2f) for every source.

        A real periodic source tends to have detectable power at harmonics of
        the fundamental; an isolated noise spike in the power cube typically
        does not.  The ratio harmonic_power / peak_power provides a
        discriminator between true detections and chance fluctuations.

        Adds 'harmonic_power' (mean SNR at 2f within the aperture) and
        'harmonic_ratio' (harmonic_power / peak power_norm) to self.sources.
        NaN is recorded when the harmonic lies outside the frequency grid.
        """
        if radius is None:
            radius = self.aperture_radius
        if self.sources is None or len(self.sources) == 0:
            return
        h_powers, h_ratios = [], []
        for i in range(len(self.sources)):
            src = self.sources.iloc[i]
            peak_pn = float(self.power_norm[
                int(src['power_ind']),
                int(round(float(src['ycentroid']))),
                int(round(float(src['xcentroid'])))
            ])
            hp = _harmonic_power(float(src['freq']), self.power_norm, self.freq,
                                 float(src['xcentroid']), float(src['ycentroid']),
                                 radius)
            h_powers.append(hp)
            ratio = (hp / peak_pn) if (np.isfinite(hp) and peak_pn > 0) else np.nan
            h_ratios.append(ratio)
        self.sources = self.sources.copy()
        self.sources['harmonic_power'] = h_powers
        self.sources['harmonic_ratio'] = h_ratios

    def measure_odd_even_asymmetry(self, n_bins=20):
        """
        Apply the odd-even half-cycle test to all sources.

        For each source, folds the light curve at twice the detected period and
        measures whether the two half-cycles differ in shape.  A high asymmetry
        score (> ~0.3) indicates the source is likely an eclipsing binary whose
        true orbital period is 2× the currently stored period.

        Adds 'odd_even_asymmetry' (float, 0–1) to self.sources.  NaN is
        recorded when there are too few observations per phase bin.
        """
        if self.sources is None or len(self.sources) == 0:
            return
        scores = []
        for i in range(len(self.sources)):
            freq = float(self.sources['freq'].iloc[i])
            t, f = self.lcs[i][0], self.lcs[i][1]
            scores.append(_odd_even_asymmetry(t, f, freq, n_bins))
        self.sources = self.sources.copy()
        self.sources['odd_even_asymmetry'] = scores

    def plot_object(self,index=None,savepath=None,cut_rad=3,power_scale='linear',power_plot='snr'):
        if self.phase is None:
            self.make_lcs()
        #for i in range(len(self.phase)):
        cut_rad = 3
        if index is None:
            index = np.arange(0,len(self.sources))
        else:
            if type(index) == int:
                index = [index]
        for i in index:
            up = np.nanmax(self.binned[i][1]) + 0.5 * np.nanmax(abs(self.binned[i][1]))
            down = np.nanmin(self.binned[i][1]) - 0.5 * np.nanmax(abs(self.binned[i][1]))
            fig,ax = plt.subplot_mosaic('''AAII
                                           BBII
                                           CCCC
                                           ''')
            ax['A'].set_title('Lightcurve')
            time = self.lcs[i][0]
            start = int(time[0])
            time = time - start
            ax['A'].plot(time,self.lcs[i][1],'.')
            ax['A'].set_ylim(down,up)
            ax['A'].set_xlabel(f'MJD + {int(start)}')
            ax['A'].set_ylabel('Counts')
            ax['B'].set_title('Phase fold (p = ' + str(np.round(1/self.sources['freq'].iloc[i],2)) + ' days)')
            ax['B'].plot(self.phase[i][0],self.phase[i][1],'.',alpha=0.1)
            ax['B'].plot(self.binned[i][0],self.binned[i][1],'.')
            ax['B'].set_xlabel('Phase')
            ax['B'].set_ylabel('Counts')
            ax['B'].set_ylim(down,up)

            ax['C'].set_title('Power spectrum')
            if power_plot.lower() == 'snr':
                ax['C'].semilogx(1/self.source_power_norm[i][0],self.source_power_norm[i][1],'-',label='SNR')
            elif power_plot.lower() == 'power':
                ax['C'].semilogx(1/self.source_power[i][0],self.source_power[i][1],'-',label='Power')
            elif power_plot.lower() == 'both':
                ax['C'].semilogx(1/self.source_power[i][0],self.source_power[i][1],'-',label='Power')
                ax['C'].semilogx(1/self.source_power_norm[i][0],self.source_power_norm[i][1],'-',label='SNR')
            else:
                raise ValueError('Only valid options for power_plots: snr, power, both')
            if power_scale.lower() == 'log':
                ax['C'].set_yscale('log')
            ax['C'].axvline(1/self.sources['freq'].iloc[i],color='k',ls=':')
            ax['C'].set_xlabel('Period (days)')
            #ax['C'].set_ylabel('"SNR" power')
            ax['C'].set_ylabel('Power')
            ax['C'].legend()




            ind = self.freq == self.sources['freq'].iloc[i]
            gind = int(self.sources.power_ind.iloc[i])
            im = self.power_norm[gind]
            x = self.sources.xcentroid.iloc[i]
            y = self.sources.ycentroid.iloc[i]
            xind = int(np.round(x,0))
            yind = int(np.round(y,0))
            xlow = np.max([xind-cut_rad,0])
            xhigh = np.min([xind+cut_rad+1,im.shape[1]])
            ylow = np.max([yind-cut_rad,0])
            yhigh = np.min([yind+cut_rad+1,im.shape[0]])

            im = im[ylow:yhigh,xlow:xhigh]
            vmin = np.percentile(im,16)
            vmax = np.percentile(im,99)
            im = ax['I'].imshow(im,origin='lower',vmin=vmin,vmax=vmax)
            fig.colorbar(im,ax=ax['I'])

            ax['I'].scatter(x-xlow,y-ylow,c='C1')
            ax['I'].set_title('Peak power')

            h, w = im.get_array().shape
            locs = ax['I'].get_xticks()
            locs = locs[(locs >= 0) & (locs < w)]
            ax['I'].set_xticks(locs, labels=[f'{loc + xlow:.0f}' for loc in locs])

            locs = ax['I'].get_yticks()
            locs = locs[(locs >= 0) & (locs < h)]
            ax['I'].set_yticks(locs, labels=[f'{loc + ylow:.0f}' for loc in locs])


            plt.tight_layout()
            if savepath is not None:
                plt.savefig(f'{savepath}/var_{i}.png')

    def save_detections(self,savepath=None,savename=None):
        if savepath is None:
            if self.savepath is None:
                print('No save path selected, saving to current directory')
                savepath = '.'
            else:
                savepath = self.savepath
        if savename is None:
            if self.savename is None:
                print('No save name selected, saving to: power_scan_var.csv')
                savename = 'power_scan_var.csv'
            else:
                savename = self.savename

        self.sources.to_csv(savepath+savename,index=False)

    #def save_lightcurves(self)

    def make_lcs(self):
        print('making light curve')
        self.get_lightcurves()
        self.phase_fold()
        self.bin_phase()


    def filter_sources(self):
        """Apply quality cuts based on phase_coherence_lim, period_max_frac, and local_threshold."""
        if self.sources is None:
            return
        mask = np.ones(len(self.sources), dtype=bool)

        if self.phase_coherence_lim is not None and 'phase_coherence' in self.sources:
            mask &= self.sources['phase_coherence'].values >= self.phase_coherence_lim

        if self.period_max_frac is not None and self._period_high is not None:
            period_max = self._period_high * self.period_max_frac
            mask &= (1.0 / self.sources['freq'].values) <= period_max

        if self.odd_even_asymmetry_lim is not None and 'odd_even_asymmetry' in self.sources:
            asym = self.sources['odd_even_asymmetry'].values
            mask &= np.where(np.isfinite(asym), asym <= self.odd_even_asymmetry_lim, True)

        n_before = len(self.sources)
        self.sources = self.sources[mask].reset_index(drop=True)
        n_after = len(self.sources)
        if n_before != n_after:
            print(f'filter_sources: {n_before} → {n_after} sources')

    def run(self):
        self.clean_data()
        print('making cube')
        if self.block_size is None:
            self.batch_make_freq_cube()
        else:
            self.block_make_freq_cube()
        print('finding sources')
        self.find_freq_sources()
        print('cleaning detections')
        self.detection_cleaning()
        if self.sources is not None:
            print('finding peak frequency')
            self.find_peak_power()
            print('refining centroids')
            self.refine_centroids()
            print('finding fundamental period')
            self.find_fundamental()
            print('measuring phase coherence')
            self.measure_phase_coherence()
            print('measuring harmonic power')
            self.measure_harmonic_power()
            print('odd-even asymmetry test')
            self.measure_odd_even_asymmetry()
            self.filter_sources()
        else:
            print('No sources detected.')
            return
            



