"""End-to-end integration tests for the full detection pipeline."""
import numpy as np
import pytest

from power_scan import periodogram_detection


@pytest.fixture(scope='module')
def detector_with_signal():
    """
    Run the full pipeline on synthetic data with a strong sinusoidal source
    injected at pixel (15, 15).  The signal is deliberately loud so the
    detection is deterministic regardless of nifty_ls internals.
    """
    rng = np.random.default_rng(123)
    time = np.linspace(0, 200, 400)
    data = rng.normal(100, 1, (400, 30, 30)).astype(float)

    period = 15.0
    amp = 50.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            data[:, 15 + dy, 15 + dx] += amp * np.sin(2 * np.pi * time / period)

    det = periodogram_detection(
        time, data,
        snr_lim=5,
        local_threshold=5,
        snr_search_lim=8,
        period_lim=[5.0, 100.0],
        cpu=1,
        run=True,
    )
    return det


class TestFullPipeline:
    def test_sources_not_none(self, detector_with_signal):
        assert detector_with_signal.sources is not None

    def test_at_least_one_source_detected(self, detector_with_signal):
        assert len(detector_with_signal.sources) >= 1

    def test_detected_period_close_to_injected(self, detector_with_signal):
        injected = 15.0
        detected = detector_with_signal.sources['period'].values
        assert np.any(np.abs(detected - injected) / injected < 0.15)

    def test_source_position_near_injection(self, detector_with_signal):
        x = detector_with_signal.sources['xcentroid'].values
        y = detector_with_signal.sources['ycentroid'].values
        dist = np.sqrt((x - 15.0) ** 2 + (y - 15.0) ** 2)
        assert np.any(dist < 3.0)

    def test_lcs_shape_after_pipeline(self, detector_with_signal):
        det = detector_with_signal
        n_sources = len(det.sources)
        assert det.lcs.shape[0] == n_sources
        assert det.lcs.shape[1] == 2

    def test_binned_shape_after_pipeline(self, detector_with_signal):
        det = detector_with_signal
        n_sources = len(det.sources)
        assert det.binned.shape[0] == n_sources
        assert det.binned.shape[1] == 2

    def test_freq_and_period_in_sources_df(self, detector_with_signal):
        cols = detector_with_signal.sources.columns
        assert 'freq' in cols
        assert 'period' in cols

    def test_period_is_reciprocal_of_freq(self, detector_with_signal):
        sources = detector_with_signal.sources
        np.testing.assert_allclose(
            sources['period'].values,
            1.0 / sources['freq'].values,
            rtol=1e-6,
        )
