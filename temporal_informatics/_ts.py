"""
Author: Riley Hales
Copyright: Riley Hales, RCH Engineering, 2020
License: BSD Clear 3 Clause License
All rights reserved
"""
import datetime
import os
import warnings

import affine
import geopandas
import numpy as np
import pandas as pd
import rasterio.features

from ._utils import _array_by_engine
from ._utils import _array_to_stat_list
from ._utils import _attr_by_engine
from ._utils import _check_var_in_dataset
from ._utils import _delta_to_datetime
from ._utils import _gen_stat_list
from ._utils import _open_by_engine
from ._utils import _pick_engine

__all__ = ['TimeSeries']

ALL_STATS = ('mean', 'median', 'max', 'min', 'sum', 'std',)
ALL_ENGINES = ('xarray', 'netcdf4', 'cfgrib', 'pygrib', 'h5py', 'rasterio',)
RECOGNIZED_TIME_INTERVALS = ('years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds',)
SPATIAL_X_VARS = ('x', 'lon', 'longitude', 'longitudes', 'degrees_east', 'eastings',)
SPATIAL_Y_VARS = ('y', 'lat', 'latitude', 'longitudes', 'degrees_north', 'northings',)


class TimeSeries:
    """
    Creates a time series of values from arrays contained in netCDF, grib, hdf, or geotiff formats. Values in the
    series are extracted by specifying coordinates of a point, range of coordinates, a spatial data file, or computing
    statistics for the entire array.

    Args:
        files (list): A list (even if len==1) of either absolute file paths to netcdf, grib, hdf5, or geotiff files or
            urls to an OPENDAP service (but beware the data transfer speed bottleneck)
        var (str or int): The name of a variable as it is stored in the file (e.g. often 'temp' or 'T' instead of
            Temperature) or the band number if you are using grib files and you specify the engine as pygrib. If the var
            is contained in a group, include the group name as a unix style path e.g. 'group_name/var'
        dim_order (tuple): A tuple of the names of the dimensions for `var`, listed in order.

    Keyword Args:
        t_var (str): Name of the time variable if it is used in the files. Default: 'time'
        statistics (list or str): How to reduce arrays of values to a single scalar value for the timeseries.
            Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
            Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        engine (str): the python package used to power the file reading. Defaults to best for the type of input data
        xr_kwargs (dict): A dictionary of kwargs that you might need when opening complex grib files with xarray
        fill_value (int): The value used for filling no_data spaces in the source file's array. Default: -9999.0
        epsg (str): an EPSG number, e.g. 4326 (the default), for the raster data's CRS. Required for spatial series
        interp_units (bool): If your data conforms to the CF NetCDF standard for time data, choose True to
            convert the values in the time variable to datetime strings in the pandas output. The units string for the
            time variable of each file is checked separately unless you specify it in the unit_str parameter.
        unit_str (str): a CF Standard conforming string indicating how the spacing and origin of the time values.
            Only specify this if ALL files that you query will contain the same units string. This is helpful if your
            files do not contain a units string. Usually this looks like "step_size since YYYY-MM-DD HH:MM:SS" such as
            "days since 2000-01-01 00:00:00".
        origin_format (str): A datetime.strptime string for extracting the origin time from the units string.
        strp_filename (str): A datetime.strptime string for extracting datetimes from patterns in file names.
    """

    statistics = ('mean',)

    def __init__(self, files: list, var: str or int, dim_order: tuple, **kwargs):
        # parameters configuring how the data is interpreted
        self.files = files
        self.var = var
        self.dim_order = dim_order

        # optional parameters describing how to access the data
        self.engine = kwargs.get('engine', _pick_engine(self.files[0]))
        self.xr_kwargs = kwargs.get('xr_kwargs', None)
        self.fill_value = kwargs.get('fill_value', -9999.0)

        # optional parameters modifying how the time data is interpreted
        self.t_var = kwargs.get('t_var', 'time')
        self.interp_units = kwargs.get('interp_units', False)
        self.strp_filename = kwargs.get('strp_filename', False)
        self.unit_str = kwargs.get('unit_str', None)
        self.origin_format = kwargs.get('origin_format', '%Y-%m-%d %X')

        # optional parameter modifying which statistics to process
        self.statistics = _gen_stat_list(kwargs.get('statistics', ('mean',)))

        # optional parameter helping to handle spatial data series
        self.epsg = kwargs.get('epsg', False)

        # validate that some parameters are compatible
        if self.engine == 'rasterio':
            assert isinstance(self.var, int), 'GeoTIFF variables must be integer band numbers'
            if not self.dim_order == ('y', 'x'):
                warnings.warn('For GeoTIFFs, the correct dim order is ("y", "x")')
                self.dim_order = ('y', 'x')

    def point(self, *coordinates: int or float or None) -> pd.DataFrame:
        """
        Extracts a time series at a point for a given series of coordinate values

        Args:
            coordinates (int or float or None): provide a coordinate value (integer or float) for each dimension of the
                array which you are creating a time series for. You need to provide exactly the same number of
                coordinates as there are dimensions
        Returns:
            pandas.DataFrame with an index, a column named datetime, and a column named values.
        """
        assert len(self.dim_order) == len(coordinates)

        # make the return item
        results = dict(datetime=[], values=[])

        # map coordinates -> cell indices -> python slice() objects
        slices = self._map_coords_to_slice(coordinates)

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = _open_by_engine(file, self.engine, self.xr_kwargs)
            results['datetime'] += list(self._handle_time_steps(opened_file, file))

            # extract the appropriate values from the variable
            vs = _array_by_engine(opened_file, self.var)[slices]
            if vs.ndim == 0:
                if vs == self.fill_value:
                    vs = np.nan
                results['values'].append(vs)
            elif vs.ndim == 1:
                vs[vs == self.fill_value] = np.nan
                for v in vs:
                    results['values'].append(v)
            else:
                raise ValueError('There are too many dimensions after slicing')
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def range(self, min_coordinates: tuple, max_coordinates: tuple) -> pd.DataFrame:
        """
        Args:
            min_coordinates (tuple): a tuple containing minimum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
            max_coordinates (tuple): a tuple containing maximum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        assert len(self.dim_order) == len(min_coordinates) == len(max_coordinates)

        # make the return item
        results = dict(datetime=[])

        # add a list for each stat requested
        for stat in self.statistics:
            results[stat] = []

        # map coordinates -> cell indices -> python slice() objects
        slices = self._map_coords_to_slice(min_coordinates, max_coordinates)

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = _open_by_engine(file, self.engine, self.xr_kwargs)
            results['datetime'] += list(self._handle_time_steps(opened_file, file))

            # slice the variable's array, returns array with shape corresponding to dimension order and size
            vs = _array_by_engine(opened_file, self.var)[slices]
            vs[vs == self.fill_value] = np.nan
            for stat in self.statistics:
                results[stat] += _array_to_stat_list(vs, stat)
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def spatial(self, geom: str) -> pd.DataFrame:
        """
        Applicable only to source data with 2 spatial dimensions and, optionally, a time dimension.

        Args:
            geom (str): path to any spatial geometry file, such as a shapefile or geojson, which can be read by
                geopandas. You also need to specify the source raster's CRS string with the epsg parameter.
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        # verify that a crs has been specified
        if not self.epsg:
            raise ValueError('An epsg has not been specified for the source data. Please specify with TimeSeries.epsg')

        # make the return item
        results = dict(datetime=[])

        # add a list for each stat requested
        for stat in self.statistics:
            results[stat] = []

        mask = self._create_spatial_mask_array(geom)

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = _open_by_engine(file, self.engine, self.xr_kwargs)
            results['datetime'] += list(self._handle_time_steps(opened_file, file))

            # slice the variable's array, returns array with shape corresponding to dimension order and size
            vals = _array_by_engine(opened_file, self.var)
            vals[vals == self.fill_value] = np.nan

            # if the dimensions are the same, apply the mask
            if vals.ndim == 2:
                vals = np.where(mask, vals, np.nan).squeeze()
                for stat in self.statistics:
                    results[stat] += _array_to_stat_list(vals, stat)
            # otherwise, iterate over the time dimension
            elif vals.ndim == 3:
                if self.t_var in self.dim_order:
                    # roll axis brings the time dimension to the "front" so we iterate over it in a for loop
                    for time_step in np.rollaxis(vals, self.dim_order.index(self.t_var)):
                        time_step = np.flip(time_step, axis=0)
                        time_step = np.where(mask, time_step, np.nan).squeeze()
                        for stat in self.statistics:
                            results[stat] += _array_to_stat_list(time_step, stat)
                else:
                    raise RuntimeError('3D gridded data incompatible with 2D spatial mask')
            else:
                raise RuntimeError(f'Wrong dimensions. mask dims: {mask.ndim}, data\'s dims: {vals.ndim}, file: {file}')

            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def masks(self, mask: np.array) -> pd.DataFrame:
        """
        Subsets the source arrays with any mask matching the dimensions of the source data. Useful when you want to

        Args:
            mask (np.array): a numpy array of boolean values, the same shape as the source data files (not including the
                time dimension, if applicable). True values mark cells that you want to keep.
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        # make the return item
        results = dict(datetime=[])
        # add a list for each stat requested
        for statistic in self.statistics:
            results[statistic] = []

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = _open_by_engine(file, self.engine, self.xr_kwargs)
            results['datetime'] += list(self._handle_time_steps(opened_file, file))

            # slice the variable's array, returns array with shape corresponding to dimension order and size
            vals = _array_by_engine(opened_file, self.var)
            vals[vals == self.fill_value] = np.nan

            # if the dimensions are the same
            if vals.ndim == mask.ndim:
                vals = np.where(mask, vals, np.nan).squeeze()
                for statistic in self.statistics:
                    results[statistic] += _array_to_stat_list(vals, statistic)
            elif vals.ndim == mask.ndim + 1:
                if self.t_var in self.dim_order:
                    # roll axis brings the time dimension to the "front" so we iterate over it in a for loop
                    for time_step in np.rollaxis(vals, self.dim_order.index(self.t_var)):
                        time_step = np.flip(time_step, axis=0)
                        time_step = np.where(mask, time_step, np.nan).squeeze()
                        for statistic in self.statistics:
                            results[statistic] += _array_to_stat_list(time_step, statistic)
                else:
                    raise RuntimeError(
                        f'Wrong dimensions. mask dims: {mask.ndim}, data\'s dims {vals.ndim}, file: {file}')
            else:
                raise RuntimeError(f'Wrong dimensions. mask dims: {mask.ndim}, data\'s dims {vals.ndim}, file: {file}')

            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def stats(self, *statistics) -> pd.DataFrame:
        """
        Computes statistics for the entire array of data contained in each file.

        Args:
            statistics (str): Optional: the name of each of the statistics you want to be calculated for the array.
                Defaults to the value of TimeSeries.statistics and overrides that value is specified here. Options are
                mean, median, max, min, sum, std (standard deviation) and any percentile number including the % such as
                '25%' for the 25th percentile.
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        # set the specified statistics
        if statistics:
            self.statistics = _gen_stat_list(statistics)

        # make the return item
        results = dict(datetime=[])
        # add a list for each stat requested
        for statistic in statistics:
            results[statistic] = []

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = _open_by_engine(file, self.engine, self.xr_kwargs)
            results['datetime'] += list(self._handle_time_steps(opened_file, file))

            # slice the variable's array, returns array with shape corresponding to dimension order and size
            vals = _array_by_engine(opened_file, self.var)
            vals[vals == self.fill_value] = np.nan
            for statistic in self.statistics:
                if self.t_var in self.dim_order:
                    # roll axis brings the time dimension to the "front" so we iterate over it in a for loop
                    for time_step_array in np.rollaxis(vals, self.dim_order.index(self.t_var)):
                        results[statistic] += _array_to_stat_list(time_step_array, statistic)
                else:
                    results[statistic] += _array_to_stat_list(vals, statistic)
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def _map_coords_to_slice(self, coords_min: tuple, coords_max: tuple = False, ) -> tuple:
        slices = []

        tmp_file = _open_by_engine(self.files[0], self.engine, self.xr_kwargs)

        for order, coord_var in enumerate(self.dim_order):
            val1 = coords_min[order]
            if val1 is None:
                slices.append(slice(None))
                continue

            vals = _array_by_engine(tmp_file, coord_var)

            # reduce the number of dimensions on the coordinate variable if applicable
            if vals.ndim < 2:
                pass
            if vals.ndim == 2:
                if vals[0, 0] == vals[0, 1]:
                    vals = vals[:, 0]
                elif vals[0, 0] == vals[1, 0]:
                    vals = vals[0, :]
                else:
                    raise RuntimeError("A 2D coordinate variable had non-uniform values and couldn't be reduced")
            elif vals.ndim > 2:
                raise RuntimeError(f"Invalid data. Coordinate variables should be 1 dimensional")

            min_val = vals.min()
            max_val = vals.max()

            if not max_val >= val1 >= min_val:
                raise ValueError(f'Coordinate value ({val1}) is outside the min/max range ({min_val}, '
                                 f'{max_val}) for the dimension {coord_var}')
            index1 = (np.abs(vals - val1)).argmin()

            if not coords_max:
                slices.append(index1)
                continue

            val2 = coords_max[order]
            if not max_val >= val2 >= min_val:
                raise ValueError(f'Coordinate value ({val2}) is outside the min/max range ({min_val}, '
                                 f'{max_val}) for the dimension {coord_var}')
            index2 = (np.abs(vals - val2)).argmin()

            # check each option in case the index is the same or in case the coords were provided backwards
            if index1 == index2:
                slices.append(index1)
            elif index1 < index2:
                slices.append(slice(index1, index2))
            else:
                slices.append(slice(index2, index1))
        if self.engine != 'pygrib':
            tmp_file.close()
        return tuple(slices)

    def _create_spatial_mask_array(self, geom: str, ) -> np.ma:
        x = None
        y = None
        for a in self.dim_order:
            if a in SPATIAL_X_VARS:
                x = a
            if a in SPATIAL_Y_VARS:
                y = a

        sample_data = _open_by_engine(self.files[0], self.engine, self.xr_kwargs)
        x = _array_by_engine(sample_data, x)
        y = _array_by_engine(sample_data, y)
        if self.engine != 'pygrib':
            sample_data.close()

        # catch the case when people use improper 2d instead of proper 1d coordinate dimensions
        if x.ndim == 2:
            x = x[0, :]
        if y.ndim == 2:
            y = y[:, 0]

        # read the shapefile
        shp_file = geopandas.read_file(geom).to_crs(epsg=self.epsg)
        # creates a binary, boolean mask of the shapefile in it's crs over the affine transformation area
        mask = rasterio.features.geometry_mask(shp_file.geometry,
                                               (y.shape[0], x.shape[0],),
                                               affine.Affine(x[1] - x[0], 0, x.min(), 0, y[0] - y[1], y.max()),
                                               invert=True)
        return mask

    def _handle_time_steps(self, opened_file, file_path):
        if self.interp_units:  # convert the time variable array's numbers to datetime representations
            tvals = _array_by_engine(opened_file, self.t_var)
            if self.engine == 'xarray':
                return tvals
            if self.unit_str is None:
                return _delta_to_datetime(tvals, _attr_by_engine(opened_file, self.t_var, 'units'), self.origin_format)
            else:
                return _delta_to_datetime(tvals, self.unit_str, self.origin_format)

        elif self.strp_filename:  # strip the datetime from the file name
            return [datetime.datetime.strptime(os.path.basename(file_path), self.strp_filename), ]

        elif _check_var_in_dataset(opened_file, self.t_var):  # use the time variable if it exists
            tvals = _array_by_engine(opened_file, self.t_var)
            if isinstance(tvals, np.datetime64):
                return [tvals]
            if tvals.ndim == 0:
                return tvals
            else:
                dates = []
                for t in tvals:
                    dates.append(t)
                return dates
        else:
            return [os.path.basename(file_path), ]
