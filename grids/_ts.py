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
import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio.features
import requests
import xarray as xr
from pydap.cas.urs import setup_session

try:
    import pygrib
except ImportError:
    pygrib = None

from ._utils import _array_by_engine
from ._utils import _array_to_stat_list
from ._utils import _attr_by_engine
from ._utils import _check_var_in_dataset
from ._utils import _delta_to_datetime
from ._utils import _gen_stat_list

__all__ = ['TimeSeries', ]

ALL_STATS = ('mean', 'median', 'max', 'min', 'sum', 'std',)
RECOGNIZED_TIME_INTERVALS = ('years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds',)
SPATIAL_X_VARS = ('x', 'lon', 'longitude', 'longitudes', 'degrees_east', 'eastings',)
SPATIAL_Y_VARS = ('y', 'lat', 'latitude', 'longitudes', 'degrees_north', 'northings',)
ALL_ENGINES = ('xarray', 'opendap', 'auth-opendap', 'netcdf4', 'cfgrib', 'pygrib', 'h5py', 'rasterio',)
NETCDF_EXTENSIONS = ('.nc', '.nc4')
GRIB_EXTENSIONS = ('.grb', 'grb2', '.grib', '.grib2')
HDF_EXTENSIONS = ('.h5', '.hd5', '.hdf5')
GEOTIFF_EXTENSIONS = ('.gtiff', '.tiff', 'tif')


class TimeSeries:
    """
    Creates a time series of values from arrays contained in netCDF, grib, hdf, or geotiff formats. Values in the
    series are extracted by specifying coordinates of a point, range of coordinates, a spatial data file, or computing
    statistics for the entire array.

    Args:
        files (list): A list (even if len==1) of either absolute file paths to netcdf, grib, hdf5, or geotiff files or
            urls to an OPeNDAP service (but beware the data transfer speed bottleneck)
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
        user (str): a username used for authenticating remote datasets, if required by your remote data source
        pswd (str): a password used for authenticating remote datasets, if required by your remote data source
        session (requests.Session): a requests Session object preloaded with credentials/tokens for authentication
        xr_kwargs (dict): A dictionary of kwargs that you might need when opening complex grib files with xarray
        fill_value (int): The value used for filling no_data spaces in the source file's array. Default: -9999.0
        interp_units (bool): If your data conforms to the CF NetCDF standard for time data, choose True to
            convert the values in the time variable to datetime strings in the pandas output. The units string for the
            time variable of each file is checked separately unless you specify it in the unit_str parameter.
        unit_str (str): a CF Standard conforming string indicating how the spacing and origin of the time values.
            Only specify this if ALL files that you query will contain the same units string. This is helpful if your
            files do not contain a units string. Usually this looks like "step_size since YYYY-MM-DD HH:MM:SS" such as
            "days since 2000-01-01 00:00:00".
        origin_format (str): A datetime.strptime string for extracting the origin time from the units string.
        strp_filename (str): A datetime.strptime string for extracting datetimes from patterns in file names.

    Methods:
        point: Extracts a time series of values at a point for a given series of coordinate values
        bound: Extracts a time series of values with a bounding box for each requested statistic
        shape: Extracts a time series of values on a line or within a polygon for each requested statistic
        masks: Extracts a time series of values from the array for a given mask array for each requested statistic
        stats: Extracts a time series of values which are requested statistics to summarize the array values

    Example:
        import grids

        # collect the input information
        files = ['/path/to/file/1.nc', '/path/to/file/2.nc', '/path/to/file/3.nc', ]
        var = 'name_of_my_variable'
        dim_order = ('name', 'of', 'dimensions', 'of', 'variable')

        # combine these into an instance of the TimeSeries class
        series = grids.TimeSeries(files=files, var=var, dim_order=dim_order)
        # call the function to query the time series subset you're interested in
        point_time_series = series.point(coords*)

    Example:
        # current GFS 1/4 degree forecast time series for air temperature in Provo Utah
        files = ['https://tds.scigw.unidata.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best']
        var = 'Temperature_surface'
        dim_order = ('time1', 'lat', 'lon')

        series = TimeSeries(files=files, var=var, dim_order=dim_order)
        temp_forecast = series.point(None, 40.25, -111.65 + 360)

    """

    def __init__(self, files: list, var: str or int, dim_order: tuple, **kwargs):
        # parameters configuring how the data is interpreted
        self.files = files
        self.var = var
        self.dim_order = dim_order

        # optional parameters describing how to access the data
        self.engine = kwargs.get('engine', False)
        if not self.engine:
            f = files[0]
            if f.startswith('http') and 'nasa.gov' in f:  # reading from a nasa opendap server (requires auth)
                self.engine = 'auth-opendap'
            elif f.startswith('http'):  # reading from opendap
                self.engine = 'opendap'
            elif any(f.endswith(i) for i in NETCDF_EXTENSIONS):
                self.engine = 'netcdf4'
            elif any(f.endswith(i) for i in GRIB_EXTENSIONS):
                self.engine = 'cfgrib'
            elif any(f.endswith(i) for i in HDF_EXTENSIONS):
                self.engine = 'h5py'
            elif any(f.endswith(i) for i in GEOTIFF_EXTENSIONS):
                self.engine = 'rasterio'
            else:
                raise ValueError(f'Could not guess appropriate file reading ending, please specify it')
        else:
            assert self.engine in ALL_ENGINES, f'engine "{self.engine}" not recognized'
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

        # option parameters describing behavior for timeseries with vector data (cache to make scripts concise)
        self.behavior = kwargs.get('behavior', 'dissolved')
        self.organizedby = kwargs.get('organizedby', None)

        # optional authentication for remote datasets
        self.user = kwargs.get('user', None)
        self.pswd = kwargs.get('pswd', None)
        self.session = kwargs.get('session', False)
        if not self.session and self.user is not None and self.pswd is not None:
            a = requests.Session()
            a.auth = (self.user, self.pswd)
            self.session = a

        # validate that some parameters are compatible
        if self.engine == 'rasterio':
            assert isinstance(self.var, int), 'GeoTIFF variables must be integer band numbers'
            if not self.dim_order == ('y', 'x'):
                warnings.warn('For GeoTIFFs, the correct dim order is ("y", "x")')
                self.dim_order = ('y', 'x')

        elif self.engine == 'pygrib':
            if pygrib is None:
                raise ModuleNotFoundError('pygrib engine only available if optional pygrib dependency is installed')
            assert isinstance(self.var, int), 'pygrib engine variables must be integer band numbers'

    def __bool__(self):
        return True

    def __str__(self):
        string = 'grids.TimeSeries Object'
        for p in vars(self):
            if p == 'files':
                string += f'\n\t{p}: {len(self.__getattribute__(p))}'
            else:
                string += f'\n\t{p}: {self.__getattribute__(p)}'
        return string

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
        assert len(self.dim_order) == len(coordinates), 'Specify 1 coordinate for each dimension of the array'

        # make the return item
        results = dict(datetime=[], values=[])

        # map coordinates -> cell indices -> python slice() objects
        slices = self._map_coords_to_slice(coordinates)

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = self._open_data(file)
            results['datetime'] += list(self._handle_time_steps(opened_file, file))

            # extract the appropriate values from the variable
            vs = _array_by_engine(opened_file, self.var, slices)
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

    def bound(self, min_coordinates: tuple, max_coordinates: tuple) -> pd.DataFrame:
        """
        Args:
            min_coordinates (tuple): a tuple containing minimum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
            max_coordinates (tuple): a tuple containing maximum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        assert len(self.dim_order) == len(min_coordinates) == len(max_coordinates), \
            'Specify 1 min and 1 max coordinate for each dimension'

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
            opened_file = self._open_data(file)
            results['datetime'] += list(self._handle_time_steps(opened_file, file))

            # slice the variable's array, returns array with shape corresponding to dimension order and size
            vs = _array_by_engine(opened_file, self.var, slices)
            vs[vs == self.fill_value] = np.nan
            for stat in self.statistics:
                results[stat] += _array_to_stat_list(vs, stat)
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def shape(self, vector: str, behavior: str = None, organizedby: str = None) -> pd.DataFrame:
        """
        Applicable only to source data with 2 spatial dimensions and, optionally, a time dimension.

        Args:
            vector (str): path to any spatial polygon file, e.g. shapefile or geojson, which can be read by geopandas.
            behavior (str): sets how to generate masks. Options are:
                dissolved- treats all features as if they were 1 feature and masks the entire set of polygons
                areaweighted- handles all features as 1 but weights each feature by its area
                features- treats each feature as a separate entity, must specify an attribute shared by each feature to
                 use as a label for the time series results of each feature.
            organizedby: The name of the attribute in the vector data features to label the several outputs
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        if not len(self.dim_order) == 3:
            raise RuntimeError('You can only extract by polygon if the data is exactly 3 dimensional: time, y, x')

        # cache the behavior and organization parameters
        self.behavior = behavior if behavior is not None else self.behavior
        self.organizedby = organizedby if organizedby is not None else self.organizedby
        # todo self._assert_valid_arguments

        # make the return item
        results = dict(datetime=[])

        masks = self._create_spatial_mask_array(vector)

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = self._open_data(file)
            new_time_steps = list(self._handle_time_steps(opened_file, file))
            num_time_steps = len(new_time_steps)
            results['datetime'] += new_time_steps

            slices = [slice(None), ] * len(self.dim_order)
            time_index = self.dim_order.index(self.t_var)

            # slice the variable's array, returns array with shape corresponding to dimension order and size
            for i in range(num_time_steps):
                slices[time_index] = slice(i, i + 1)
                vals = _array_by_engine(opened_file, self.var, tuple(slices))
                vals = np.flip(vals, axis=0)
                for mask in masks:
                    masked_vals = np.where(mask[1], vals, np.nan).squeeze()
                    masked_vals[masked_vals == self.fill_value] = np.nan
                    for stat in self.statistics:
                        if f'{mask[0]}-{stat}' not in results.keys():
                            results[f'{mask[0]}-{stat}'] = []
                        results[f'{mask[0]}-{stat}'] += _array_to_stat_list(masked_vals, stat)

            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def masks(self, mask: np.array) -> pd.DataFrame:
        """
        Subsets the source arrays with any mask matching the dimensions of the source data. Useful when you want to
        generate your own mask.

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
            opened_file = self._open_data(file)
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
            opened_file = self._open_data(file)
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

        if self.engine == 'pygrib':
            revert_engine = self.engine
            self.engine = 'cfgrib'
        else:
            revert_engine = False

        tmp_file = self._open_data(self.files[0])

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

        tmp_file.close()

        if revert_engine:
            self.engine = revert_engine

        return tuple(slices)

    def _create_spatial_mask_array(self, vector: str, ) -> np.ma:
        x, y = None, None
        for a in self.dim_order:
            if a in SPATIAL_X_VARS:
                x = a
            elif a in SPATIAL_Y_VARS:
                y = a
        if x is None or y is None:
            raise ValueError('Unable to determine x and y dimensions')

        sample_data = self._open_data(self.files[0])
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
        vector_gdf = geopandas.read_file(vector)
        vector_gdf = vector_gdf.to_crs(epsg=4326)

        # set up the variables to creating and storing masks
        masks = []
        gridshape = (y.shape[0], x.shape[0],)
        affinetransform = affine.Affine(np.abs(x[1] - x[0]), 0, x.min(), 0, np.abs(y[1] - y[0]), y.min())

        # creates a binary, boolean mask of the shapefile
        # in it's crs, over the affine transform area, for a certain masking behavior
        if self.behavior == 'dissolved':
            masks.append(
                ('featuremask',
                 rasterio.features.geometry_mask(vector_gdf.geometry, gridshape, affinetransform, invert=True),)
            )
        elif self.behavior == 'features':
            for idx, row in vector_gdf.iterrows():
                masks.append(
                    (row[self.organizedby],
                     rasterio.features.geometry_mask(
                         geopandas.GeoSeries(row.geometry), gridshape, affinetransform, invert=True),)
                )
        return masks

    def _handle_time_steps(self, opened_file, file_path):
        if self.interp_units:  # convert the time variable array's numbers to datetime representations
            tvals = _array_by_engine(opened_file, self.t_var)
            if self.engine == 'xarray':
                return tvals
            if self.unit_str is None:
                return _delta_to_datetime(tvals, _attr_by_engine(opened_file, self.t_var, 'units'), self.origin_format)
            return _delta_to_datetime(tvals, self.unit_str, self.origin_format)

        elif self.strp_filename:  # strip the datetime from the file name
            return [datetime.datetime.strptime(os.path.basename(file_path), self.strp_filename), ]

        elif self.engine == 'pygrib':
            return [opened_file[self.var].validDate]

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

    def _open_data(self, path):
        if self.engine == 'xarray':
            return xr.open_dataset(path, backend_kwargs=self.backend_kwargs)
        elif self.engine == 'opendap':
            try:
                if self.session:
                    return xr.open_dataset(xr.backends.PydapDataStore.open(path, session=self.session))
                else:
                    return xr.open_dataset(path)
            except Exception as e:
                raise ConnectionRefusedError(f'Couldn\'t connect to dataset {path}. Does it exist? Need credentials?')
        elif self.engine == 'auth-opendap':
            return xr.open_dataset(xr.backends.PydapDataStore.open(
                path, session=setup_session(self.user, self.pswd, check_url=path)))
        elif self.engine == 'netcdf4':
            return nc.Dataset(path, 'r')
        elif self.engine == 'cfgrib':
            return xr.open_dataset(path, engine='cfgrib', backend_kwargs=self.backend_kwargs)
        elif self.engine == 'pygrib':
            a = pygrib.open(path)
            return a.read()
        elif self.engine == 'h5py':
            return h5py.File(path, 'r')
        elif self.engine == 'rasterio':
            return xr.open_rasterio(path)
        else:
            raise ValueError(f'Unable to open file, unsupported engine: {self.engine}')
