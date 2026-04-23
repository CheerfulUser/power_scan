"""Tests for standalone helper functions."""
import numpy as np
import pandas as pd
from power_scan import Generate_LC, compress_freq_groups
from power_scan.power_scan import _Spatial_group


# ---------------------------------------------------------------------------
# Generate_LC
# ---------------------------------------------------------------------------

class TestGenerateLC:
    def test_sum_returns_correct_length(self):
        time = np.linspace(0, 10, 50)
        flux = np.ones((50, 20, 20))
        t, f = Generate_LC(time, flux, x=10, y=10, method='sum', radius=1.5)
        assert len(t) == 50
        assert len(f) == 50

    def test_sum_values_match_box_sum(self):
        # buffer = floor(1.5) = 1 → 3×3 box = 9 pixels × value 2.0 = 18.0
        time = np.linspace(0, 10, 30)
        flux = np.full((30, 20, 20), 2.0)
        _, f = Generate_LC(time, flux, x=10, y=10, method='sum', radius=1.5)
        np.testing.assert_allclose(f, 18.0)

    def test_sum_radius_2_gives_5x5_box(self):
        # buffer = floor(2.0) = 2 → 5×5 = 25 pixels × value 1.0 = 25.0
        time = np.arange(20, dtype=float)
        flux = np.ones((20, 20, 20))
        _, f = Generate_LC(time, flux, x=10, y=10, method='sum', radius=2.0)
        np.testing.assert_allclose(f, 25.0)

    def test_frame_start_slices_time(self):
        time = np.arange(50, dtype=float)
        flux = np.ones((50, 20, 20))
        t, f = Generate_LC(time, flux, x=10, y=10, frame_start=10)
        assert len(t) == 40
        assert t[0] == 10.0

    def test_frame_end_slices_time(self):
        time = np.arange(50, dtype=float)
        flux = np.ones((50, 20, 20))
        t, f = Generate_LC(time, flux, x=10, y=10, frame_end=19)
        assert len(t) == 20
        assert t[-1] == 19.0

    def test_frame_start_and_end(self):
        time = np.arange(50, dtype=float)
        flux = np.ones((50, 20, 20))
        t, f = Generate_LC(time, flux, x=10, y=10, frame_start=5, frame_end=14)
        assert len(t) == 10
        assert t[0] == 5.0
        assert t[-1] == 14.0

    def test_invalid_method_returns_none(self):
        # Generate_LC has no else branch; unknown methods fall through and return None.
        time = np.arange(10, dtype=float)
        flux = np.ones((10, 20, 20))
        result = Generate_LC(time, flux, x=10, y=10, method='invalid')
        assert result is None


# ---------------------------------------------------------------------------
# _Spatial_group
# ---------------------------------------------------------------------------

class TestSpatialGroup:
    def test_nearby_sources_share_objid(self):
        df = pd.DataFrame({
            'xcentroid': [10.0, 10.3, 90.0],
            'ycentroid': [10.0, 10.3, 90.0],
            'objid':     [0.0,  0.0,  0.0],
        })
        result = _Spatial_group(df, distance=2.0)
        assert result['objid'].iloc[0] == result['objid'].iloc[1]
        assert result['objid'].iloc[0] != result['objid'].iloc[2]

    def test_distant_sources_get_unique_objids(self):
        df = pd.DataFrame({
            'xcentroid': [0.0, 50.0, 100.0],
            'ycentroid': [0.0, 50.0, 100.0],
            'objid':     [0.0, 0.0,  0.0],
        })
        result = _Spatial_group(df, distance=1.0)
        assert len(result['objid'].unique()) == 3

    def test_objid_column_is_integer(self):
        df = pd.DataFrame({
            'xcentroid': [5.0, 6.0],
            'ycentroid': [5.0, 6.0],
            'objid':     [0.0, 0.0],
        })
        result = _Spatial_group(df, distance=5.0)
        assert result['objid'].dtype in (int, np.int64, np.int32)

    def test_custom_write_col(self):
        df = pd.DataFrame({
            'xcentroid': [1.0, 2.0],
            'ycentroid': [1.0, 2.0],
            'group_id':  [0.0, 0.0],
        })
        result = _Spatial_group(df, distance=5.0, write_col='group_id')
        assert 'group_id' in result.columns

    def test_single_source_gets_objid_1(self):
        df = pd.DataFrame({
            'xcentroid': [10.0],
            'ycentroid': [10.0],
            'objid':     [0.0],
        })
        result = _Spatial_group(df, distance=1.0)
        assert result['objid'].iloc[0] == 1


# ---------------------------------------------------------------------------
# compress_freq_groups
# ---------------------------------------------------------------------------

class TestCompressFreqGroups:
    def test_keeps_one_per_group(self, clustered_sources_df):
        df = clustered_sources_df.copy()
        df['objid'] = [1, 1, 2, 2, 3]
        result = compress_freq_groups(df)
        assert len(result) == 3

    def test_keeps_max_local_sig(self, clustered_sources_df):
        df = clustered_sources_df.copy()
        df['objid'] = [1, 1, 2, 2, 3]
        result = compress_freq_groups(df)
        g1 = result[result['objid'] == 1]
        g2 = result[result['objid'] == 2]
        assert g1['local_sig'].values[0] == 15.0
        assert g2['local_sig'].values[0] == 12.0

    def test_single_source_per_group_unchanged(self):
        df = pd.DataFrame({
            'xcentroid': [10.0, 50.0],
            'ycentroid': [10.0, 50.0],
            'objid':     [1,    2],
            'local_sig': [8.0,  6.0],
        })
        result = compress_freq_groups(df)
        assert len(result) == 2
