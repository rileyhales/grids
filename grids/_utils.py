import datetime
import re

import h5py
import netCDF4 as nc
import numpy as np
import xarray as xr
from dateutil.relativedelta import relativedelta

try:
    import pygrib
except ImportError:
    pygrib = None

__all__ = ['_assign_engine', '_array_by_engine', '_guess_time_var', '_attr_by_engine', '_check_var_in_dataset',
           '_array_to_stat_list', '_delta_to_datetime', '_gen_stat_list', 'ALL_ENGINES', 'SPATIAL_X_VARS',
           'SPATIAL_Y_VARS']

ALL_ENGINES = ('xarray', 'opendap', 'auth-opendap', 'netcdf4', 'cfgrib', 'pygrib', 'h5py', 'rasterio',)
ALL_STATS = ('mean', 'median', 'max', 'min', 'sum', 'std',)

T_VARS = ('time', )
SPATIAL_X_VARS = ('x', 'lon', 'longitude', 'longitudes', 'degrees_east', 'eastings',)
SPATIAL_Y_VARS = ('y', 'lat', 'latitude', 'longitudes', 'degrees_north', 'northings',)

NETCDF_EXTENSIONS = ('.nc', '.nc4')
GRIB_EXTENSIONS = ('.grb', 'grb2', '.grib', '.grib2')
HDF_EXTENSIONS = ('.h5', '.hd5', '.hdf5')
GEOTIFF_EXTENSIONS = ('.gtiff', '.tiff', 'tif')


def _assign_engine(sample_file):
    if sample_file.startswith('http') and 'nasa.gov' in sample_file:  # nasa opendap server requires auth
        return 'auth-opendap'
    elif sample_file.startswith('http'):  # reading from opendap
        return 'opendap'
    elif any(sample_file.endswith(i) for i in NETCDF_EXTENSIONS):
        return 'netcdf4'
    elif any(sample_file.endswith(i) for i in GRIB_EXTENSIONS):
        return 'cfgrib'
    elif any(sample_file.endswith(i) for i in HDF_EXTENSIONS):
        return 'h5py'
    elif any(sample_file.endswith(i) for i in GEOTIFF_EXTENSIONS):
        return 'rasterio'
    else:
        raise ValueError(f'Could not guess appropriate file reading ending, please specify it')


def _guess_time_var(dims):
    # do any of the recognized time variables show up in the dim_order
    for var in T_VARS:
        if var in dims:
            return var
    # do any of the dims match the time pattern
    for dim in dims:
        if re.match('time*', dim):
            return dim
    return 'time'


def _array_by_engine(open_file, var: str or int, slices: tuple = slice(None)) -> np.array:
    if isinstance(open_file, xr.Dataset):  # xarray, cfgrib
        return open_file[var][slices].data
    elif isinstance(open_file, xr.DataArray):  # rasterio
        if isinstance(var, int):
            return open_file.data[var][slices]
        return open_file[var].data[slices]
    elif isinstance(open_file, nc.Dataset):  # netcdf4
        return open_file[var][slices]
    elif isinstance(open_file, list):  # pygrib
        return open_file[var].values[slices]
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        return open_file[var][slices]  # might need to use [...] for string data
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _attr_by_engine(open_file, var: str, attribute: str) -> str:
    if isinstance(open_file, xr.Dataset) or isinstance(open_file, xr.DataArray):  # xarray, cfgrib, rasterio
        return open_file[var].attrs[attribute]
    elif isinstance(open_file, nc.Dataset):  # netcdf4
        return open_file[var].getncattr(attribute)
    elif isinstance(open_file, list):  # pygrib
        return open_file[var][attribute]
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        return open_file[var].attrs[attribute].decode('UTF-8')
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _check_var_in_dataset(open_file, var) -> bool:
    if isinstance(open_file, xr.Dataset) or isinstance(open_file, nc.Dataset):  # xarray, netcdf4
        return bool(var in open_file.variables)
    elif isinstance(open_file, list):  # pygrib comes as lists of messages
        return bool(var <= len(open_file))
    elif isinstance(open_file, h5py.File) or isinstance(open_file, h5py.Dataset):  # h5py
        return bool(var in open_file.keys())
    elif isinstance(open_file, xr.DataArray):
        return bool(var <= open_file.band.shape[0])
    else:
        raise ValueError(f'Unrecognized opened file dataset: {type(open_file)}')


def _array_to_stat_list(array: np.array, statistic: str) -> list:
    list_of_stats = []
    # add the results to the lists of values and times
    if array.ndim == 1 or array.ndim == 2:
        if statistic == 'mean':
            list_of_stats.append(np.nanmean(array))
        elif statistic == 'median':
            list_of_stats.append(np.nanmedian(array))
        elif statistic == 'max':
            list_of_stats.append(np.nanmax(array))
        elif statistic == 'min':
            list_of_stats.append(np.nanmin(array))
        elif statistic == 'sum':
            list_of_stats.append(np.nansum(array))
        elif statistic == 'std':
            list_of_stats.append(np.nanstd(array))
        elif '%' in statistic:
            list_of_stats.append(np.nanpercentile(array, int(statistic.replace('%', ''))))
        else:
            raise ValueError(f'Unrecognized statistic, {statistic}. Use stat_type= mean, min or max')
    elif array.ndim == 3:
        for v in array:
            list_of_stats += _array_to_stat_list(v, statistic)
    else:
        raise ValueError('Too many dimensions in the array. You probably did not mean to do stats like this')
    return list_of_stats


def _delta_to_datetime(tvals: np.array, ustr: str, origin_format: str = '%Y-%m-%d %X') -> np.array:
    interval = ustr.split(' ')[0].lower()
    origin = datetime.datetime.strptime(ustr.split(' since ')[-1], origin_format)
    if interval == 'years':
        delta = relativedelta(years=1)
    elif interval == 'months':
        delta = relativedelta(months=1)
    elif interval == 'weeks':
        delta = relativedelta(weeks=1)
    elif interval == 'days':
        delta = relativedelta(days=1)
    elif interval == 'hours':
        delta = relativedelta(hours=1)
    elif interval == 'minutes':
        delta = relativedelta(minutes=1)
    elif interval == 'seconds':
        delta = relativedelta(seconds=1)
    elif interval == 'milliseconds':
        delta = datetime.timedelta(milliseconds=1)
    elif interval == 'microseconds':
        delta = datetime.timedelta(microseconds=1)
    else:
        raise ValueError(f'Unrecognized time interval: {interval}')

    # the values in the time variable, scaled to a number of time deltas, plus the origin time
    a = tvals * delta + origin
    return np.array([i.strftime("%Y-%m-%d %X") for i in a])


def _gen_stat_list(stats: str or list):
    if isinstance(stats, str):
        if stats == 'all':
            return ALL_STATS
        else:
            return stats.lower().replace(' ', '').split(',')
    elif isinstance(stats, tuple) or isinstance(stats, list):
        if any(stat not in ALL_STATS for stat in stats):
            raise ValueError(f'Unrecognized statistic requested. Choose from: {ALL_STATS}')
        return stats
