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
import geopandas as gpd
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

from ._coords import _map_coords_to_slice
from ._coords import _map_coord_to_index
from ._coords import _map_indices_to_slice

from ._utils import _assign_eng
from ._utils import _guess_time_var
from ._utils import _array_by_eng
from ._utils import _array_to_stat_list
from ._utils import _attr_by_eng
from ._utils import _check_var_in_dataset
from ._utils import _delta_to_datetime
from ._utils import _gen_stat_list
from ._utils import ALL_ENGINES
from ._utils import SPATIAL_X_VARS
from ._utils import SPATIAL_Y_VARS

__all__ = ['TimeSeries', ]


class TimeSeries:
    """
    Creates a time series of values from arrays contained in netCDF, grib, hdf, or geotiff formats. Values in the
    series are extracted by specifying coordinates of a point, range of coordinates, a spatial data file, or computing
    statistics for the entire array.

    Args:
        files (list): A list (even if len==1) of either absolute file paths to netcdf, grib, hdf5, or geotiff files or
            urls to an OPeNDAP service (but beware the data transfer speed bottleneck)
        variables (str or int or list or tuple): The name of the variable(s) to query as they are stored in the file
            (e.g. often 'temp' or 'T' instead of Temperature) or the band number if you are using grib files *and* you
            specify the engine as pygrib. If the var is contained in a group, include the group name as a unix style
            path e.g. 'group_name/var'
        dim_order (tuple): A tuple of the names of the dimensions for `var`, listed in order.

    Keyword Args:
        t_var (str): Name of the time variable if it is used in the files. grids will try to guess it if you do not
            specify and default to 'time'
        statistics (str or tuple): How to reduce arrays of values to a single scalar value for the timeseries.
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
        point: Extracts a time series of values at a point for a given coordinate pair
        multipoint: Extracts a time series of values for several points given a series of coordinate values
        bound: Extracts a time series of values with a bounding box for each requested statistic
        range: Alias for TimeSeries.bound()
        shape: Extracts a time series of values on a line or within a polygon for each requested statistic
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
        dim_order = ('time', 'lat', 'lon')

        series = TimeSeries(files=files, var=var, dim_order=dim_order)
        temp_forecast = series.point(None, 40.25, -111.65 + 360)

    """
    # core parameters from user
    files: list
    var: tuple
    dim_order: tuple
    engine: str

    # how to handle the time data
    t_var: str
    t_index: int
    # t_range: tuple
    interp_units: bool
    strp_filename: str
    unit_str: str
    origin_format: str

    # reducing arrays to numbers
    statistics: str or list or tuple or np.ndarray
    behavior: str
    labelby: str
    fill_value: int or float or bool

    # help opening data
    xr_kwargs: dict
    user: str
    pswd: str
    session: requests.session

    def __init__(self, files: list, var: str or int or list or tuple, dim_order: tuple, **kwargs):
        # parameters configuring how the data is interpreted
        self.files = (files,) if isinstance(files, str) else files
        self.variables = (var,) if isinstance(var, str) else var
        assert len(self.variables) >= 1, 'specify at least 1 variable'
        self.dim_order = dim_order

        # optional parameters describing how to access the data
        self.engine = kwargs.get('engine', _assign_eng(files[0]))
        assert self.engine in ALL_ENGINES, f'engine "{self.engine}" not recognized'
        self.xr_kwargs = kwargs.get('xr_kwargs', None)
        self.fill_value = kwargs.get('fill_value', -9999.0)

        # optional parameters modifying how the time data is interpreted
        self.t_var = kwargs.get('t_var', _guess_time_var(self.dim_order))
        self.t_index = self.dim_order.index(self.t_var)
        # self.t_range = kwargs.get('t_range', slice(None))
        self.interp_units = kwargs.get('interp_units', False)
        self.strp_filename = kwargs.get('strp_filename', False)
        self.unit_str = kwargs.get('unit_str', None)
        self.origin_format = kwargs.get('origin_format', '%Y-%m-%d %X')

        # optional parameter modifying which statistics to process
        self.statistics = _gen_stat_list(kwargs.get('statistics', ('mean',)))

        # option parameters describing behavior for timeseries with vector data (cache to make scripts concise)
        self.behavior = kwargs.get('behavior', 'dissolve')
        self.labelby = kwargs.get('labelby', None)

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
            assert isinstance(self.variables, int), 'GeoTIFF variables must be integer band numbers'
            if not self.dim_order == ('y', 'x'):
                warnings.warn('For GeoTIFFs, the correct dim order is ("y", "x")')
                self.dim_order = ('y', 'x')

        elif self.engine == 'pygrib':
            if pygrib is None:
                raise ModuleNotFoundError('pygrib engine only available if optional pygrib dependency is installed')
            assert isinstance(self.variables, int), 'pygrib engine variables must be integer band numbers'

    def __bool__(self):
        return True

    def __str__(self):
        string = 'grids.TimeSeries'
        for p in vars(self):
            if p == 'files':
                string += f'\n\t{p}: {len(self.__getattribute__(p))}'
            else:
                string += f'\n\t{p}: {self.__getattribute__(p)}'
        return string

    def __repr__(self):
        return self.__str__()

    def point(self,
              *coords: int or float or None, ) -> pd.DataFrame:
        """
        Extracts a time series at a point for a given series of coordinate values

        Args:
            coords (int or float or None): provide a coordinate value (integer or float) for each dimension of the
                array which you are creating a time series for. You need to provide exactly the same number of
                coordinates as there are dimensions
        Returns:
            pandas.DataFrame with an index, a column named datetime, and a column named values.
        """
        assert len(self.dim_order) == len(coords), 'Specify 1 coordinate for each dimension of the array'

        # make the return item
        results = dict(datetime=[])
        for var in self.variables:
            results[var] = []

        # map coordinates -> cell indices -> python slice() objects
        slices = self._gen_dim_slices(coords, 'point')

        # iterate over each file extracting the value and time for each
        for num, file in enumerate(self.files):
            # open the file
            opened_file = self._open_data(file)
            tsteps, tslices = self._handle_time(opened_file, file, (coords[self.t_index], coords[self.t_index]))
            results['datetime'] += list(tsteps)
            slices[self.t_index] = tslices
            for var in self.variables:
                # extract the appropriate values from the variable
                vs = _array_by_eng(opened_file, var, tuple(slices))
                if vs.ndim == 0:
                    if vs == self.fill_value:
                        vs = np.nan
                    results[var].append(vs)
                elif vs.ndim == 1:
                    vs[vs == self.fill_value] = np.nan
                    for v in vs:
                        results[var].append(v)
                else:
                    raise ValueError('Too many dimensions remain after slicing')
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def multipoint(self,
                   *coords: list,
                   labels: list = None, ) -> pd.DataFrame:
        """
        Extracts a time series at many points for a given series of coordinate values. Each point should have the same
        time coordinate and different coordinates for each other dimension.

        Args:
            coords (int or float or None): a list of coordinate tuples or a 2D numpy array. Each coordinate pair in
                the list should provide a coordinate value (integer or float) for each dimension of the array, e.g.
                len(coordinate_pair) == len(dim_order). See TimeSeries.point for more explanation.
            labels (list): an optional list of strings which label each of the coordinates provided. len(labels) should
                be equal to len(coords)
        Returns:
            pandas.DataFrame with an index, a column named datetime, and a column named values.
        """
        assert len(self.dim_order) == len(coords[0]), 'Specify 1 coordinate for each dimension of the array'
        if labels is None:
            labels = [f'point{i}' for i in range(len(coords))]
        assert len(labels) == len(coords), 'You must provide a label for each point or use auto numbering'

        datalabels = []
        for label in labels:
            for var in self.variables:
                datalabels.append(f'({var})_{label}')

        # make the return item
        results = dict(datetime=[])
        for datalabel in datalabels:
            results[datalabel] = []

        # map coordinates -> cell indices -> python slice() objects
        slices = self._gen_dim_slices(coords, 'multipoint')

        # iterate over each file extracting the value and time for each
        for file in self.files:
            opened_file = self._open_data(file)
            tsteps, tslices = self._handle_time(opened_file, file, (coords[self.t_index], coords[self.t_index]))
            results['datetime'] += list(tsteps)
            slices[self.t_index] = tslices
            for var in self.variables:
                for i, slc in enumerate(slices):
                    # extract the appropriate values from the variable
                    vs = _array_by_eng(opened_file, var, tuple(slc))
                    if vs.ndim == 0:
                        if vs == self.fill_value:
                            vs = np.nan
                        results[f'({var})_{labels[i]}'].append(vs)
                    elif vs.ndim == 1:
                        vs[vs == self.fill_value] = np.nan
                        for v in vs:
                            results[f'({var})_{labels[i]}'].append(v)
                    else:
                        raise ValueError('There are too many dimensions after slicing')
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def bound(self,
              min_coords: tuple,
              max_coords: tuple,
              statistics: str or tuple = None, ) -> pd.DataFrame:
        """
        Args:
            min_coords (tuple): a tuple containing minimum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
            max_coords (tuple): a tuple containing maximum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
            statistics (str or tuple): How to reduce arrays of values to a single scalar value for the time series.
                Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
                Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        assert len(self.dim_order) == len(min_coords) == len(max_coords), \
            'Specify 1 min and 1 max coordinate for each dimension'

        # handle the optional arguments
        self.statistics = _gen_stat_list(statistics) if statistics is not None else self.statistics

        # make the return item
        results = dict(datetime=[])
        # add a list for each stat requested
        for var in self.variables:
            for stat in self.statistics:
                results[f'({var})_{stat}'] = []

        # map coordinates -> cell indices -> python slice() objects
        slices = self._gen_dim_slices((min_coords, max_coords), 'range')

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = self._open_data(file)
            tsteps, tslices = self._handle_time(opened_file, file, (min_coords[self.t_index], max_coords[self.t_index]))
            results['datetime'] += list(tsteps)
            slices[self.t_index] = tslices
            for var in self.variables:
                # slice the variable's array, returns array with shape corresponding to dimension order and size
                vs = _array_by_eng(opened_file, var, tuple(slices))
                vs[vs == self.fill_value] = np.nan
                for stat in self.statistics:
                    results[f'({var})_{stat}'] += _array_to_stat_list(vs, stat)
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def range(self,
              min_coordinates: tuple,
              max_coordinates: tuple,
              statistics: str or tuple = None, ) -> pd.DataFrame:
        """
        Alias for TimeSeries.bound(). Refer to documentation for the bound method.
        """
        return self.bound(min_coordinates, max_coordinates, statistics)

    def shape(self,
              mask: str or np.ndarray,
              time_range: tuple = (None, None),
              behavior: str = None,
              labelby: str = None,
              statistics: str or tuple = None, ) -> pd.DataFrame:
        """
        Applicable only to source data with 2 spatial dimensions and, optionally, a time dimension.

        Args:
            mask (str): path to any spatial polygon file, e.g. shapefile or geojson, which can be read by gpd.
            time_range: a tuple of the min and max time range to query a time series for
            behavior (str): determines how the vector data is used to mask the arrays. Options are: dissolve, features
                - dissolve: treats all features as if they were 1 feature and masks the entire set of polygons in 1 grid
                - features: treats each feature as a separate entity, must specify an attribute shared by each feature
                with unique values for each feature used to label the resulting series
            labelby: The name of the attribute in the vector data features to label the several outputs
            statistics (str or tuple): How to reduce arrays of values to a single scalar value for the time series.
                Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%) or all.
                Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        if not len(self.dim_order) == 3:
            raise RuntimeError('You can only extract by polygon if the data is exactly 3 dimensional: time, y, x')

        # cache the behavior and organization parameters
        self.behavior = behavior if behavior is not None else self.behavior
        self.labelby = labelby if labelby is not None else self.labelby
        self.statistics = _gen_stat_list(statistics) if statistics is not None else self.statistics

        if isinstance(mask, str):
            masks = self._create_spatial_mask_array(mask)
        elif isinstance(mask, np.ndarray):
            masks = ['masked', mask]

        # make the return item
        results = dict(datetime=[])
        for mask in masks:
            for stat in self.statistics:
                for var in self.variables:
                    results[f'({var})_{mask[0]}-{stat}'] = []

        # slice data on all dimensions
        slices = [slice(None), ] * len(self.dim_order)

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = self._open_data(file)
            tsteps, tslices = self._handle_time(opened_file, file, (time_range[0], time_range[1]))
            results['datetime'] += list(tsteps)
            slices[self.t_index] = tslices
            num_time_steps = len(tsteps)

            for var in self.variables:
                # slice the variable's array, returns array with shape corresponding to dimension order and size
                for i in range(num_time_steps):
                    slices[self.t_index] = slice(i, i + 1)
                    vals = _array_by_eng(opened_file, var, tuple(slices))
                    vals = np.flip(vals, axis=0)
                    for mask in masks:
                        masked_vals = np.where(mask[1], vals, np.nan).squeeze()
                        masked_vals[masked_vals == self.fill_value] = np.nan
                        for stat in self.statistics:
                            results[f'({var})_{mask[0]}-{stat}'] += _array_to_stat_list(masked_vals, stat)

            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def stats(self,
              statistics: str or tuple = 'mean') -> pd.DataFrame:
        """
        Computes statistics for the entire array of data contained in each file.

        Args:
            statistics (str): Optional: the name of each of the statistics you want to be calculated for the array.
                Defaults to the value of TimeSeries.statistics or overrides with value specified. Options are
                mean, median, max, min, sum, std (standard deviation) and a percentile written as '25%'.
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        # set the specified statistics
        self.statistics = _gen_stat_list(statistics)

        # make the return item
        results = dict(datetime=[])
        # add a list for each stat requested
        for stat in self.statistics:
            for var in self.variables:
                results[f'({var})_{stat}'] = []

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = self._open_data(file)
            results['datetime'] += list(self._handle_time(opened_file, file))
            for var in self.variables:
                # slice the variable's array, returns array with shape corresponding to dimension order and size
                vals = _array_by_eng(opened_file, var)
                vals[vals == self.fill_value] = np.nan
                for stat in self.statistics:
                    if self.t_var in self.dim_order:
                        # roll axis brings the time dimension to the "front" so we iterate over it in a for loop
                        for time_step_array in np.rollaxis(vals, self.t_index):
                            results[f'({var})_{stat}'] += _array_to_stat_list(time_step_array, stat)
                    else:
                        results[f'({var})_{stat}'] += _array_to_stat_list(vals, stat)
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def _gen_dim_slices(self,
                        coords: tuple,
                        slice_style: str):
        if self.engine == 'pygrib':
            revert_engine = self.engine
            self.engine = 'cfgrib'
        else:
            revert_engine = False

        slices = []
        tmp_file = self._open_data(self.files[0])
        for index, dim in enumerate(self.dim_order):
            if dim == self.t_var:
                slices.append(None)
                continue
            vals = _array_by_eng(tmp_file, dim)

            if slice_style == 'point':
                slices.append(_map_coords_to_slice(vals, coords[index], coords[index], dim))
            elif slice_style == 'multipoint':
                for coord in coords:
                    slices.append(_map_coords_to_slice(vals, coord[index], coord[index], dim))
            elif slice_style == 'range':
                slices.append(_map_coords_to_slice(vals, coords[0][index], coords[1][index], dim))
            else:
                raise RuntimeError("Slice behavior not implemented")

        if revert_engine:
            self.engine = revert_engine
        return slices

    def _create_spatial_mask_array(self, vector: str, ) -> np.ma:
        # todo check here if the array needs to be flipped by looking if coord values increase or decrease
        x, y = None, None
        for a in self.dim_order:
            if a in SPATIAL_X_VARS:
                x = a
            elif a in SPATIAL_Y_VARS:
                y = a
        if x is None or y is None:
            raise ValueError('Unable to determine x and y dimensions')

        sample_data = self._open_data(self.files[0])
        x = _array_by_eng(sample_data, x)
        y = _array_by_eng(sample_data, y)
        if self.engine != 'pygrib':
            sample_data.close()

        # catch the case when people use improper 2d instead of proper 1d coordinate dimensions
        if x.ndim == 2:
            x = x[0, :]
        if y.ndim == 2:
            y = y[:, 0]

        # read the shapefile
        vector_gdf = gpd.read_file(vector)
        vector_gdf = vector_gdf.to_crs(epsg=4326)

        # set up the variables to creating and storing masks
        masks = []
        gridshape = (y.shape[0], x.shape[0],)
        affinetransform = affine.Affine(np.abs(x[1] - x[0]), 0, x.min(), 0, np.abs(y[1] - y[0]), y.min())

        # creates a binary/boolean mask of the shapefile
        # in the same crs, over the affine transform area, for a certain masking behavior
        if self.behavior == 'dissolve':
            masks.append(
                ('shape',
                 rasterio.features.geometry_mask(vector_gdf.geometry, gridshape, affinetransform, invert=True),)
            )
        elif self.behavior == 'features':
            assert self.labelby in vector_gdf.keys(), 'labelby parameter not found in attributes of the vector data'
            for idx, row in vector_gdf.iterrows():
                masks.append(
                    (row[self.labelby],
                     rasterio.features.geometry_mask(
                         gpd.GeoSeries(row.geometry), gridshape, affinetransform, invert=True),)
                )
        return masks

    def _handle_time(self, opened_file, file_path: str, time_range: tuple) -> tuple:
        if _check_var_in_dataset(opened_file, self.t_var):
            tvals = _array_by_eng(opened_file, self.t_var)
            if isinstance(tvals, np.datetime64):
                tvals = [tvals]
            if tvals.ndim == 0:
                ...
            else:
                tvals = [t for t in tvals]

            if self.interp_units:  # convert the time variable array's numbers to datetime representations
                if self.engine == 'xarray':
                    ...
                elif self.unit_str is None:
                    tvals = _delta_to_datetime(tvals, _attr_by_eng(opened_file, self.t_var, 'units'), self.origin_format)
                else:
                    tvals = _delta_to_datetime(tvals, self.unit_str, self.origin_format)
        elif self.strp_filename:  # strip the datetime from the file name
            tvals = [datetime.datetime.strptime(os.path.basename(file_path), self.strp_filename), ]
        elif self.engine == 'pygrib':
            tvals = [opened_file[self.variables].validDate, ]
        else:
            raise RuntimeError('Unable to find the correct time values. Is there a time variable?')

        tvals = np.array(tvals)
        time_slices = _map_coords_to_slice(tvals, time_range[0], time_range[1], 'time')
        return tvals[(time_slices, )], time_slices

    def _open_data(self, path):
        if self.engine == 'xarray':
            return xr.open_dataset(path, backend_kwargs=self.xr_kwargs)
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
            return xr.open_dataset(path, engine='cfgrib', backend_kwargs=self.xr_kwargs)
        elif self.engine == 'pygrib':
            a = pygrib.open(path)
            return a.read()
        elif self.engine == 'h5py':
            return h5py.File(path, 'r')
        elif self.engine == 'rasterio':
            return xr.open_rasterio(path)
        else:
            raise ValueError(f'Unable to open file, unsupported engine: {self.engine}')
