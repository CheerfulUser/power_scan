import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def uniform_time():
    return np.linspace(0, 100, 200)


@pytest.fixture
def small_data(uniform_time):
    rng = np.random.default_rng(42)
    return rng.normal(100, 2, (len(uniform_time), 20, 20)).astype(float)


@pytest.fixture
def data_with_nan_frame(uniform_time):
    rng = np.random.default_rng(42)
    data = rng.normal(100, 2, (len(uniform_time), 20, 20)).astype(float)
    data[5] = np.nan
    data[10] = np.nan
    return data


@pytest.fixture
def clustered_sources_df():
    """Two pairs of nearby sources and one isolated source."""
    return pd.DataFrame({
        'xcentroid': [10.0, 10.2, 50.0, 50.4, 90.0],
        'ycentroid': [10.0, 10.2, 50.0, 50.4, 90.0],
        'objid':     [0.0,  0.0,  0.0,  0.0,  0.0],
        'local_sig': [5.0,  15.0, 8.0,  12.0, 7.0],
        'flux':      [5.0,  15.0, 8.0,  12.0, 7.0],
        'freq':      [0.1,  0.1,  0.2,  0.2,  0.3],
    })
