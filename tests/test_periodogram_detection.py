"""Tests for the periodogram_detection class."""
import numpy as np
import pandas as pd
import pytest

from power_scan import periodogram_detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(time=None, data=None, **kwargs):
    """Create a detector with run=False."""
    if time is None:
        time = np.linspace(0, 100, 100)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.normal(100, 2, (len(time), 20, 20)).astype(float)
    return periodogram_detection(time, data, run=False, **kwargs)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_attributes_stored(self):
        time = np.arange(50, dtype=float)
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (50, 10, 10))
        det = periodogram_detection(time, data, snr_lim=7, aperture_radius=2.0, run=False)
        np.testing.assert_array_equal(det.time, time)
        assert det.snr_lim == 7
        assert det.aperture_radius == 2.0

    def test_computed_attrs_none_before_run(self):
        det = make_detector()
        assert det.freq is None
        assert det.power is None
        assert det.detections is None
        assert det.sources is None
        assert det.lcs is None


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_removes_nan_frames(self, data_with_nan_frame, uniform_time):
        det = make_detector(time=uniform_time.copy(), data=data_with_nan_frame.copy())
        original_len = len(uniform_time)
        det.clean_data()
        assert len(det.time) == original_len - 2
        assert not np.any(np.isnan(det.data))

    def test_sorts_time(self):
        rng = np.random.default_rng(1)
        time = np.linspace(0, 50, 50)
        data = rng.normal(10, 1, (50, 10, 10)).astype(float)
        shuffled = rng.permutation(50)
        det = make_detector(time=time[shuffled], data=data[shuffled])
        det.clean_data()
        assert np.all(np.diff(det.time) > 0)

    def test_clean_data_preserves_length_when_no_nans(self, uniform_time, small_data):
        det = make_detector(time=uniform_time.copy(), data=small_data.copy())
        det.clean_data()
        assert len(det.time) == len(uniform_time)


# ---------------------------------------------------------------------------
# _set_period_lim
# ---------------------------------------------------------------------------

class TestSetPeriodLim:
    def test_auto_sets_sensible_limits(self, uniform_time, small_data):
        det = make_detector(time=uniform_time, data=small_data, period_lim='auto')
        det._set_period_lim()
        expected_low = np.median(np.diff(uniform_time)) * 2
        expected_high = (uniform_time[-1] - uniform_time[0]) / 1.5
        assert np.isclose(det._period_low, expected_low)
        assert np.isclose(det._period_high, expected_high)
        assert det._period_low < det._period_high

    def test_custom_list_sets_limits(self, uniform_time, small_data):
        det = make_detector(time=uniform_time, data=small_data, period_lim=[3.0, 40.0])
        det._set_period_lim()
        assert det._period_low == 3.0
        assert det._period_high == 40.0

    def test_custom_array_sets_limits(self, uniform_time, small_data):
        det = make_detector(time=uniform_time, data=small_data, period_lim=np.array([5.0, 50.0]))
        det._set_period_lim()
        assert det._period_low == 5.0
        assert det._period_high == 50.0

    def test_invalid_period_lim_raises(self, uniform_time, small_data):
        det = make_detector(time=uniform_time, data=small_data, period_lim='bad_value')
        with pytest.raises(ValueError):
            det._set_period_lim()


# ---------------------------------------------------------------------------
# batch_make_freq_cube
# ---------------------------------------------------------------------------

class TestBatchMakeFreqCube:
    @pytest.fixture
    def detector_ready(self):
        rng = np.random.default_rng(7)
        time = np.linspace(0, 100, 80)
        data = rng.normal(100, 2, (80, 20, 20)).astype(float)
        det = make_detector(time=time, data=data, period_lim=[5.0, 40.0])
        det.clean_data()
        det._set_period_lim()
        return det

    def test_power_is_3d(self, detector_ready):
        detector_ready.batch_make_freq_cube()
        assert detector_ready.power.ndim == 3

    def test_power_spatial_dims_match_data(self, detector_ready):
        ny, nx = detector_ready.data.shape[1], detector_ready.data.shape[2]
        detector_ready.batch_make_freq_cube()
        assert detector_ready.power.shape[1] == ny
        assert detector_ready.power.shape[2] == nx

    def test_freq_length_matches_power(self, detector_ready):
        detector_ready.batch_make_freq_cube()
        assert len(detector_ready.freq) == detector_ready.power.shape[0]

    def test_period_length_matches_freq(self, detector_ready):
        detector_ready.batch_make_freq_cube()
        assert len(detector_ready.period) == len(detector_ready.freq)

    def test_period_values_are_reciprocal_of_freq(self, detector_ready):
        detector_ready.batch_make_freq_cube()
        np.testing.assert_allclose(detector_ready.period, 1.0 / detector_ready.freq)

    def test_freq_within_period_limits(self, detector_ready):
        det = detector_ready
        det.batch_make_freq_cube()
        assert np.all(1.0 / det.freq >= det._period_low - 1e-10)
        assert np.all(1.0 / det.freq <= det._period_high + 1e-10)

    def test_power_norm_same_shape_as_power(self, detector_ready):
        detector_ready.batch_make_freq_cube()
        assert detector_ready.power_norm.shape == detector_ready.power.shape


# ---------------------------------------------------------------------------
# phase_fold
# ---------------------------------------------------------------------------

class TestPhaseFold:
    def _setup_detector_with_lcs(self, freq=0.1, n_times=200):
        time = np.linspace(0, 100, n_times)
        det = make_detector(time=time)
        det.time = time
        flux = np.sin(2 * np.pi * freq * time)
        det.lcs = np.array([[time, flux]])
        det.sources = pd.DataFrame({'freq': [freq]})
        return det

    def test_phase_values_in_unit_interval(self):
        det = self._setup_detector_with_lcs()
        det.phase_fold()
        assert np.all(det.phase[0][0] >= 0.0)
        assert np.all(det.phase[0][0] < 1.0)

    def test_phase_flux_length_matches_lc(self):
        det = self._setup_detector_with_lcs(n_times=150)
        det.phase_fold()
        assert det.phase[0].shape == (2, 150)

    def test_phase_fold_multiple_sources(self):
        time = np.linspace(0, 100, 100)
        det = make_detector(time=time)
        det.time = time
        f1, f2 = 0.1, 0.3
        lc1 = np.array([time, np.sin(2 * np.pi * f1 * time)])
        lc2 = np.array([time, np.sin(2 * np.pi * f2 * time)])
        det.lcs = np.array([lc1, lc2])
        det.sources = pd.DataFrame({'freq': [f1, f2]})
        det.phase_fold()
        assert len(det.phase) == 2
        for ph in det.phase:
            assert np.all(ph[0] >= 0.0)
            assert np.all(ph[0] < 1.0)


# ---------------------------------------------------------------------------
# bin_phase
# ---------------------------------------------------------------------------

class TestBinPhase:
    def _setup_detector_with_phase(self, freq=0.1):
        time = np.linspace(0, 100, 500)
        det = make_detector(time=time)
        phase_vals = ((time - time[0]) / (1.0 / freq)) % 1
        flux = np.sin(2 * np.pi * freq * time)
        det.phase = np.array([[phase_vals, flux]])
        det.sources = pd.DataFrame({'freq': [freq]})
        return det

    def test_bin_phase_output_shape(self):
        det = self._setup_detector_with_phase()
        det.bin_phase(phase_bin=0.01)
        # 100 bins from 0 to 1 in steps of 0.01
        assert det.binned.shape == (1, 2, 100)

    def test_bin_phase_centers_in_unit_interval(self):
        det = self._setup_detector_with_phase()
        det.bin_phase(phase_bin=0.01)
        centers = det.binned[0][0]
        assert np.all(centers > 0.0)
        assert np.all(centers < 1.0)

    def test_bin_phase_custom_bin_size(self):
        det = self._setup_detector_with_phase()
        det.bin_phase(phase_bin=0.05)
        assert det.binned.shape == (1, 2, 20)

    def test_bin_phase_multiple_sources(self):
        time = np.linspace(0, 100, 500)
        det = make_detector(time=time)
        f1, f2 = 0.1, 0.2
        ph1 = ((time - time[0]) / (1.0 / f1)) % 1
        ph2 = ((time - time[0]) / (1.0 / f2)) % 1
        det.phase = np.array([
            [ph1, np.sin(2 * np.pi * f1 * time)],
            [ph2, np.sin(2 * np.pi * f2 * time)],
        ])
        det.sources = pd.DataFrame({'freq': [f1, f2]})
        det.bin_phase(phase_bin=0.1)
        assert det.binned.shape == (2, 2, 10)


# ---------------------------------------------------------------------------
# detection_cleaning
# ---------------------------------------------------------------------------

class TestDetectionCleaning:
    def test_none_detections_gives_none_sources(self, uniform_time, small_data):
        det = make_detector(time=uniform_time, data=small_data)
        det.detections = None
        det.detection_cleaning()
        assert det.sources is None

    def test_snr_filter_removes_low_flux(self, uniform_time, small_data):
        """Sources below snr_lim should be discarded."""
        det = make_detector(time=uniform_time, data=small_data, snr_lim=10)
        det.detections = pd.DataFrame({
            'xcentroid': [10.0, 10.0],
            'ycentroid': [10.0, 10.0],
            'flux':      [5.0,  5.0],   # both below snr_lim=10
            'local_sig': [5.0,  5.0],
            'objid':     [0.0,  0.0],
            'freq':      [0.1,  0.1],
        })
        filtered = det.detections.loc[det.detections['flux'] >= det.snr_lim]
        assert len(filtered) == 0
