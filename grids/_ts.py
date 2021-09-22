"""
Author: Riley Hales
Copyright: Riley Hales, RCH Engineering, 2021
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
import rasterio.features as riof
import requests
import xarray as xr
from pydap.cas.urs import setup_session

try:
    import pygrib
except ImportError:
    pygrib = None

from ._coords import _map_coords_to_slice

from ._utils import _assign_eng
from ._utils import _guess_time_var
from ._utils import _array_by_eng
from ._utils import _array_to_stat_list
from ._utils import _attr_by_eng
from ._utils import _delta_to_time
from ._utils import _gen_stat_list

from ._consts import ALL_ENGINES
from ._consts import SPATIAL_X_VARS
from ._consts import SPATIAL_Y_VARS

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
        stats (str or tuple): How to reduce arrays of values to a single scalar value for the timeseries.
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
    stats: str or list or tuple or np.ndarray
    behavior: str
    label_attr: str
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
        self.t_var_in_dims = self.t_var in self.dim_order
        self.t_index = self.dim_order.index(self.t_var) if self.t_var_in_dims else False

        # optional parameters modifying how to interpret the spatial variables
        self.x_var = kwargs.get('x_var', None)
        self.y_var = kwargs.get('y_var', None)
        if self.x_var is None:
            for a in self.dim_order:
                if a in SPATIAL_X_VARS:
                    self.x_var = a
        if self.y_var is None:
            for a in self.dim_order:
                if a in SPATIAL_Y_VARS:
                    self.y_var = a

        # self.t_range = kwargs.get('t_range', slice(None))
        self.interp_units = kwargs.get('interp_units', False)
        self.strp_filename = kwargs.get('strp_filename', False)
        self.unit_str = kwargs.get('unit_str', None)
        self.origin_format = kwargs.get('origin_format', '%Y-%m-%d %X')

        # optional parameter modifying which statistics to process
        self.stats = _gen_stat_list(kwargs.get('stats', ('mean',)))

        # option parameters describing behavior for timeseries with vector data (cache to make scripts concise)
        self.behavior = kwargs.get('behavior', 'dissolve')
        self.label_attr = kwargs.get('label_attr', None)

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
            tsteps, tslices = self._handle_time(opened_file, file, (coords, coords))
            results['datetime'] += list(tsteps)
            slices[self.t_index] = tslices if self.t_var_in_dims else slices[self.t_index]
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
                datalabels.append(f'{var}_{label}')

        # make the return item
        results = dict(datetime=[])
        for datalabel in datalabels:
            results[datalabel] = []

        # map coordinates -> cell indices -> python slice() objects
        slices = self._gen_dim_slices(coords, 'multipoint')

        # iterate over each file extracting the value and time for each
        for file in self.files:
            opened_file = self._open_data(file)
            tsteps, tslices = self._handle_time(opened_file, file, (coords[0], coords[0]))
            results['datetime'] += list(tsteps)
            for var in self.variables:
                for i, slc in enumerate(slices):
                    slc[self.t_index] = tslices if self.t_var_in_dims else slc[self.t_index]
                    # extract the appropriate values from the variable
                    vs = _array_by_eng(opened_file, var, tuple(slc))
                    if vs.ndim == 0:
                        if vs == self.fill_value:
                            vs = np.nan
                        results[f'{var}_{labels[i]}'].append(vs)
                    elif vs.ndim == 1:
                        vs[vs == self.fill_value] = np.nan
                        for v in vs:
                            results[f'{var}_{labels[i]}'].append(v)
                    else:
                        raise ValueError('There are too many dimensions after slicing')
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def bound(self,
              min_coords: tuple,
              max_coords: tuple,
              stats: str or tuple = None, ) -> pd.DataFrame:
        """
        Args:
            min_coords (tuple): a tuple containing minimum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
            max_coords (tuple): a tuple containing maximum coordinates of a bounding box range- coordinates given
                in order of the dimensions of the source arrays.
            stats (str or tuple): How to reduce arrays of values to a single scalar value for the time series.
                Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%), all, or values.
                Values returns a flattened list of all values in query range for plotting or computing other stats.
                Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        assert len(self.dim_order) == len(min_coords) == len(max_coords), \
            'Specify 1 min and 1 max coordinate for each dimension'

        # handle the optional arguments
        self.stats = _gen_stat_list(stats) if stats is not None else self.stats

        # make the return item
        results = dict(datetime=[])
        # add a list for each stat requested
        for var in self.variables:
            for stat in self.stats:
                results[f'{var}_{stat}'] = []

        # map coordinates -> cell indices -> python slice() objects
        slices = self._gen_dim_slices((min_coords, max_coords), 'range')

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = self._open_data(file)
            tsteps, tslices = self._handle_time(opened_file, file, (min_coords, max_coords))
            results['datetime'] += list(tsteps)
            slices[self.t_index] = tslices if self.t_var_in_dims else slices[self.t_index]
            for var in self.variables:
                # slice the variable's array, returns array with shape corresponding to dimension order and size
                vs = _array_by_eng(opened_file, var, tuple(slices))
                vs[vs == self.fill_value] = np.nan
                for stat in self.stats:
                    results[f'{var}_{stat}'] += _array_to_stat_list(vs, stat)
            if self.engine != 'pygrib':
                opened_file.close()

        # return the data stored in a dataframe
        return pd.DataFrame(results)

    def range(self,
              min_coordinates: tuple,
              max_coordinates: tuple,
              stats: str or tuple = None, ) -> pd.DataFrame:
        """
        Alias for TimeSeries.bound(). Refer to documentation for the bound method.
        """
        return self.bound(min_coordinates, max_coordinates, stats)

    def shape(self,
              mask: str or np.ndarray,
              time_range: tuple = (None, None),
              behavior: str = None,
              label_attr: str = None,
              feature: str = None,
              stats: str or tuple = None, ) -> pd.DataFrame:
        """
        Applicable only to source data with exactly 2 spatial dimensions, x and y, and a time dimension.

        Args:
            mask (str): path to any spatial polygon file, e.g. shapefile or geojson, which can be read by gpd.
            time_range: a tuple of the min and max time range to query a time series for
            behavior (str): determines how the vector data is used to mask the arrays. Options are: dissolve, features
                - dissolve: treats all features as if they were 1 feature and masks the entire set of polygons in 1 grid
                - features: treats each feature as a separate entity, must specify an attribute shared by each feature
                with unique values for each feature used to label the resulting series
            label_attr: The name of the attribute in the vector data features to label the several outputs
            feature: A value of the label_attr attribute for 1 or more features found in the provided shapefile
            stats (str or tuple): How to reduce arrays of values to a single scalar value for the time series.
                Options include: mean, median, max, min, sum, std, a percentile (e.g. 25%), all, or values.
                Values returns a flattened list of all values in query range for plotting or computing other stats.
                Provide a list of strings (e.g. ['mean', 'max']), or a comma separated string (e.g. 'mean,max,min')
        Returns:
            pandas.DataFrame with an index, a datetime column, and a column named for each statistic specified
        """
        if not len(self.dim_order) == 3:
            raise RuntimeError('You can only extract by polygon if the data is exactly 3 dimensional: time, y, x')

        # cache the behavior and organization parameters
        self.behavior = behavior if behavior is not None else self.behavior
        self.label_attr = label_attr if label_attr is not None else self.label_attr
        self.stats = _gen_stat_list(stats) if stats is not None else self.stats

        if isinstance(mask, str):
            masks = self._create_spatial_mask_array(mask, feature)
        elif isinstance(mask, np.ndarray):
            masks = ['masked', mask]
        else:
            raise ValueError('Unusable data provided for the "mask" argument')

        # make the return item
        results = dict(datetime=[])
        for mask in masks:
            for stat in self.stats:
                for var in self.variables:
                    results[f'{var}_{mask[0]}_{stat}'] = []

        # slice data on all dimensions
        slices = [slice(None), ] * len(self.dim_order)

        # iterate over each file extracting the value and time for each
        for file in self.files:
            # open the file
            opened_file = self._open_data(file)
            tsteps, tslices = self._handle_time(opened_file, file, (time_range[0], time_range[1]))
            results['datetime'] += list(tsteps)
            slices[self.t_index] = tslices if self.t_var_in_dims else slices[self.t_index]
            num_time_steps = len(tsteps)

            for var in self.variables:
                # slice the variable's array, returns array with shape corresponding to dimension order and size
                for i in range(num_time_steps):
                    slices[self.t_index] = slice(i, i + 1)
                    vals = _array_by_eng(opened_file, var, tuple(slices))
                    for mask in masks:
                        masked_vals = np.where(mask[1], vals, np.nan).squeeze()
                        masked_vals[masked_vals == self.fill_value] = np.nan
                        for stat in self.stats:
                            results[f'{var}_{mask[0]}_{stat}'] += _array_to_stat_list(masked_vals, stat)

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

        if slice_style in ('point', 'range'):
            for index, dim in enumerate(self.dim_order):
                if dim == self.t_var:
                    slices.append(None)
                    continue
                vals = _array_by_eng(tmp_file, dim)
                if slice_style == 'point':
                    slices.append(_map_coords_to_slice(vals, coords[index], coords[index], dim))
                else:
                    slices.append(_map_coords_to_slice(vals, coords[0][index], coords[1][index], dim))
        elif slice_style == 'multipoint':
            for index, dim in enumerate(self.dim_order):
                if dim == self.t_var:
                    slices.append([None, ] * len(coords))
                    continue
                vals = _array_by_eng(tmp_file, dim)
                dim_slices = []
                for coord in coords:
                    dim_slices.append(_map_coords_to_slice(vals, coord[index], coord[index], dim))
                slices.append(dim_slices)
            slices = np.transpose(slices)
        else:
            raise RuntimeError("Slice behavior not implemented")

        if revert_engine:
            self.engine = revert_engine
        return slices

    def _create_spatial_mask_array(self, vector: str, feature: str) -> np.ma:
        if self.x_var is None or self.y_var is None:
            raise ValueError('Unable to determine x and y dimensions')
        sample_data = self._open_data(self.files[0])
        x = _array_by_eng(sample_data, self.x_var)
        y = _array_by_eng(sample_data, self.y_var)
        if self.engine != 'pygrib':
            sample_data.close()

        # catch the case when people use improper 2d instead of proper 1d coordinate dimensions
        if x.ndim == 2:
            x = x[0, :]
        if y.ndim == 2:
            y = y[:, 0]

        # check if you need to vertically invert the array mask (if y vals go from small to large)
        # or if you need to transpose the mask (if the dimensions go x then y, should be y then x- think of the shape)
        invert = y[-1] > y[0]
        transpose = self.dim_order.index(self.x_var) < self.dim_order.index(self.y_var)

        # read the shapefile
        vector_gdf = gpd.read_file(vector)
        vector_gdf = vector_gdf.to_crs(epsg=4326)

        # set up the variables to create and storing masks
        masks = []
        # what is the shape of the grid to be masked
        gshape = (y.shape[0], x.shape[0],)
        # calculate the affine transformation of the grid to be masked
        aff = affine.Affine(np.abs(x[1] - x[0]), 0, x.min(), 0, np.abs(y[1] - y[0]), y.min())

        # creates a binary/boolean mask of the shapefile
        # in the same crs, over the affine transform area, for a certain masking behavior
        if self.behavior == 'dissolve':
            m = riof.geometry_mask(vector_gdf.geometry, gshape, aff, invert=invert)
            if transpose:
                m = np.transpose(m)
            masks.append(('shape', m))
        elif self.behavior == 'feature':
            assert self.label_attr in vector_gdf.keys(), \
                'label_attr parameter not found in attributes list of the vector data'
            assert feature is not None, \
                'Provide a value for the feature argument to query for certain features'
            vector_gdf = vector_gdf[vector_gdf[self.label_attr] == feature]
            assert not vector_gdf.empty, f'No features have value "{feature}" for attribute "{self.label_attr}"'
            m = riof.geometry_mask(vector_gdf.geometry, gshape, aff, invert=invert)
            if transpose:
                m = np.transpose(m)
            masks.append((feature, m))

        elif self.behavior == 'features':
            assert self.label_attr in vector_gdf.keys(), \
                'label_attr parameter not found in attributes list of the vector data'
            for idx, row in vector_gdf.iterrows():
                m = riof.geometry_mask(gpd.GeoSeries(row.geometry), gshape, aff, invert=invert)
                if transpose:
                    m = np.transpose(m)
                masks.append((row[self.label_attr], m))
        return masks

    def _handle_time(self, opened_file, file_path: str, time_range: tuple) -> tuple:
        if self.strp_filename:  # strip the datetime from the file name
            tvals = [datetime.datetime.strptime(os.path.basename(file_path), self.strp_filename), ]
        elif self.engine == 'pygrib':
            tvals = [opened_file[self.variables].validDate, ]
        else:
            tvals = _array_by_eng(opened_file, self.t_var)
            if isinstance(tvals, np.datetime64):
                tvals = [tvals]
            if tvals.ndim == 0:
                ...
            else:
                tvals = [t for t in tvals]

            # convert the time variable array's numbers to datetime representations
            if self.interp_units:
                if self.engine == 'xarray':
                    ...
                elif self.unit_str is None:
                    tvals = _delta_to_time(tvals, _attr_by_eng(opened_file, self.t_var, 'units'), self.origin_format)
                else:
                    tvals = _delta_to_time(tvals, self.unit_str, self.origin_format)

        tvals = np.array(tvals)

        # if the variable depends on time then there should be a coordinate provided for it
        if self.t_var_in_dims:
            t1 = time_range[0]
            t2 = time_range[1]

            if isinstance(t1, list) or isinstance(t1, tuple):
                t1 = t1[self.t_index]
            if isinstance(t2, list) or isinstance(t2, tuple):
                t2 = t2[self.t_index]
        # otherwise, no time coordinates provided.
        else:
            t1 = None
            t2 = None

        time_slices = _map_coords_to_slice(tvals, t1, t2, 'time')
        return tvals[(time_slices,)], time_slices

    def _open_data(self, path):
        if self.engine == 'xarray':
            return xr.open_dataset(path, backend_kwargs=self.xr_kwargs)
        elif self.engine == 'opendap':
            try:
                if self.session:
                    return xr.open_dataset(xr.backends.PydapDataStore.open(path, session=self.session))
                else:
                    return xr.open_dataset(path)
            except ConnectionRefusedError as e:
                raise e
            except Exception as e:
                print('Unexpected Error')
                raise e
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
