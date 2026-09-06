import pandas as pd
import pytest

from app.data_sources.imd_rainfall import read_imd_rainfall_netcdf


def test_missing_netcdf_is_explicitly_failed():
    with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
        read_imd_rainfall_netcdf("does-not-exist.nc")
