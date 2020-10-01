.. toctree::
	:maxdepth: 3
	:numbered:

********************
Temporal Informatics
********************

.. image:: https://img.shields.io/pypi/v/temporal_informatics
	:target: https://pypi.org/project/temporal_informatics
	:alt: PYPI Version
.. image:: https://readthedocs.org/projects/temporal-informatics/badge/?version=latest
	:target: https://temporal-informatics.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

Tools for extracting time series subsets from n-dimensional arrays in NetCDF, GRIB, HDF, and GeoTIFF formats. Time series
can be extracted for:

#. Points - by specifying the coordinates of the point in terms of the dimensions of the array
#. Ranges or Bounding Boxes - by specifying the minimum and maximum coordinates for each dimension
#. Spatial data - if the rasters are spatial data and the appropriate dimensions are specified
#. Masks - any irregularly shaped subset of the array which you can create a binary mask array for
#. Statistical summaries - of the entire array

.. code-block:: bash

	pip install temporal_informatics

Example Usage
*************

.. code-block:: python

	import glob
	import temporal_informatics as tin

	# example using GLDAS datasets (list of absolute file paths)
	files = sorted(glob.glob('/path/to/my/spatial/data/GLDAS*.nc4'))
	# the temperature variable's name (string)
	var = 'Tair_f_inst'
	# the order of variables in the (tuple of strings)
	dim_order = ('time', 'lat', 'lon')

	# create a TimeSeries class with your data points
	ts = tin.TimeSeries(files, var, dim_order)
	# set option optional behavior for extracting series (see kwargs)
	ts.interp_units = True

	# get a time series for a point (args are coordinates of the point in order of the dim_order)
	# None -> all time values, 10 -> latitude ('lat') = 10, 15 -> longitude ('lon') = 15
	a = ts.point(None, 10, 15)
	print(a)

	# get a time series for a range of values (args are 2 tuples of coordinates
	# None -> all time values, between 10-15 latitude, between 15 and 20 longitude
	b = ts.range((None, 10, 15), (None, 15, 20))
	print(b)

	# get a time series within a shapefile's boundaries
	# define raster data's EPSG, geometry will be reprojected to this before masking the rasters (see kwargs)
	ts.epsg = 4326
	c = ts.spatial('/path/to/my/shapefile.shp')
	print(c)

	# get a time series of the averages of the entire array
	# stats defaults to computing mean, set stats to compute using ts.statistics (see kwargs)
	ts.statistics = ('mean', 'median', 'max', 'min', 'sum', 'std',)
	d = ts.stats()
	print(d)

Handling Time values
********************
Datetime values are extracted in one of 4 ways (controlled by the :code:`interp_units`, :code:`units_str`,
:code:`origin_format`, and :code:`strp_format` parameters), in this order of preference:

#. When :code:`interp_units` is True, interpret the time variable values as datetimes using time's units attribute.
	Override the file using the :code:`units_str` kwarg and the :code:`origin_format` kwarg if the date doesn't use
	YYYY-MM-DD HH:MM:SS format.
#. When a pattern is specified with :code:`strp_filename`, a datetime extracted from the filename is applied to all
   values coming from that dataset.
#. If a time variable exists, its numerical values are used without further interpretation.
#. The string file name is used if there is no time variable and no other options were provided.

TimeSeries
**********

.. automodule:: temporal_informatics
	:members: TimeSeries

Speed Test Results
******************

.. csv-table::
	:file: test_data/speed_test_times.csv
	:header-rows: 1
