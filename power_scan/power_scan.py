import numpy as np
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clipped_stats
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
                    local_threshold=10,sep_wh_ratio=0.5):
    import warnings
    from photutils.utils import NoDetectionsWarning
    warnings.filterwarnings("ignore", category=NoDetectionsWarning)

    power = power.astype(float)
    if method.lower() == 'dao':
        finder = DAOStarFinder(peak,fwhm,exclude_border=True,min_separation=3)
        s = finder.find_stars(power)
        if s is not None:
            s = s.to_pandas()
    elif method.lower() == 'sep':
        bkg = sep.Background(power)
        objects = sep.extract(power, peak)#, err=bkg.globalrms)
        objects = pd.DataFrame(objects)
        w = objects['a'].values
        h = objects['b'].values
        f = objects['flux'].values
        ratio = np.round(abs(w/h - 1),1)
        ind = (ratio < sep_wh_ratio)
        s = objects.iloc[ind]
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
            m = sigma_clip(data,masked=True,sigma=5).mask
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



class periodogram_detection():
    def __init__(self,time,data,error=None,aperture_radius=1.5,
                 snr_lim=5,fwhm=3,dao_peak=20,cpu=-1,snr_search_lim=10,
                 period_lim='auto',block_size=None,edge_buffer=0,detection_method='sep',
                 local_threshold=10,savepath=None,run=True):
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
        self.block_size = block_size
        self.edge_buffer = edge_buffer
        self.detection_method = detection_method
        self.local_threshold = local_threshold

        # calculated
        self.freq = None
        self.power = None
        self.detections = None
        self.sources = None
        self.lcs = None
        self.phase = None
        self._period_low = None
        self._period_high = None

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
        if self.period_lim == 'auto':
            self._period_low = np.median(np.diff(self.time)) * 2 
            self._period_high = (self.time[-1] - self.time[0]) / 2 
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
                                               fmin=(1/self._period_high),fmax=(1/self._period_low))
        freq = temp.freq()
        power = np.zeros((len(freq),self.data.shape[1],self.data.shape[2]))
        for i in range(len(dy)-1):
            for j in range(len(dx)-1):
                cut = self.data[:,dy[i]:dy[i+1],dx[j]:dx[j+1]]
                shaped = cut.reshape(len(cut),cut.shape[1]*cut.shape[2]).T
                batched = nifty_ls.lombscargle(self.time-self.time[0],shaped,
                                               fmin=1/self._period_high,fmax=1/self._period_low)
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
                                       fmin=1/self._period_high,fmax=1/self._period_low)
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


    def find_freq_sources(self,peak=None,fwhm=None):

        if fwhm is None:
            fwhm = self.fwhm
        if peak is None:
            peak = self.dao_peak
        ind = np.nanmax(self.power_norm,axis=(1,2)) >= self.snr_search_lim
        index = np.arange(0,len(self.freq))[ind]
        source = Parallel(n_jobs=self.cpu)(delayed(_detect_sources)(self.freq[i],self.power_norm[i],i,peak,fwhm,self.detection_method,self.local_threshold) for i in index)
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
        detect = detect.loc[detect['flux'] >= snr_lim]
        sources = _Spatial_group(detect)
        self.sources = compress_freq_groups(sources)
        self._spatial_alias_clean()

        if self.edge_buffer > 0:
            xind = (self.sources.xcentroid.values > self.edge_buffer) & (self.sources.xcentroid.values < self.data.shape[2] - self.edge_buffer)
            yind = (self.sources.ycentroid.values > self.edge_buffer) & (self.sources.ycentroid.values < self.data.shape[1] - self.edge_buffer)
            ind = xind & yind
            self.sources = self.sources.iloc[ind]


    def _make_lc(self,source,radius):
        x = source.xcentroid
        y = source.ycentroid
        t,lc = Generate_LC(time=self.time,flux=self.data,x=x,y=y,radius=radius)
        lc = np.array([t,lc])
        freq,l = Generate_LC(time=self.freq,flux=self.power_norm,x=x,y=y,radius=radius)
        if np.nansum(abs(l)) > 0:
            good = True
        else:
            good = False
        source_power = np.array([freq,l])
        return lc, source_power, good

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
        lcs,source_power,good = zip(*Parallel(n_jobs=self.cpu)(delayed(self._make_lc)(self.sources.iloc[i],radius) for i in index))
        self.lcs = np.array(lcs)
        self.source_power = np.array(source_power)
        good = np.array(good)
        self.sources = self.sources.iloc[good]
        self.lcs = np.array(lcs)[good]
        self.source_power = np.array(source_power)[good]
        


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


        
    def plot_object(self,index=None,savepath=None,cut_rad=3,power_scale='linear'):
        if self.lcs is None:
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

            ax['C'].set_title('"SNR" Power spectrum')
            ax['C'].semilogx(1/self.source_power[i][0],self.source_power[i][1],'-')
            if power_scale.lower() == 'log':
                ax['C'].set_yscale('log')
            ax['C'].axvline(1/self.sources['freq'].iloc[i],color='k',ls=':')
            ax['C'].set_xlabel('Period (days)')
            ax['C'].set_ylabel('"SNR" power')



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
            im = ax['I'].imshow(im,origin='lower')
            fig.colorbar(im,ax=ax['I'])

            ax['I'].scatter(x-xlow,y-ylow,c='C1')
            ax['I'].set_title('Peak power')


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
        



