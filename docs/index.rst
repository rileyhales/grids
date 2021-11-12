.. toctree::
    :maxdepth: 3
    :numbered:

*********************
The Grids Python Tool
*********************

.. image:: https://img.shields.io/pypi/v/grids
    :target: https://pypi.org/project/grids
    :alt: PYPI Version
.. image:: https://readthedocs.org/projects/tsgrids/badge/?version=latest
    :target: https://grids.rileyhales.com/en/latest/?badge=latest
    :alt: Documentation Status

Tools for extracting time series subsets from n-dimensional arrays in NetCDF, GRIB, HDF, and GeoTIFF formats. Time series
can be extracted for:

#. Points - by specifying the coordinates of the point in terms of the dimensions of the array
#. Ranges or Bounding Boxes - by specifying the minimum and maximum coordinates for each dimension
#. Spatial data - if the rasters are spatial data and the appropriate dimensions are specified

Citing Grids
************

If you use Grids in a project, please cite

- Our journal article at MDPI Water. doi: `10.3390/w13152066 <https://doi.org/10.3390/w13152066>`_
- The source code through Zenodo. doi: `10.5281/zenodo.5225437 <https://doi.org/10.5281/zenodo.5225437>`_

Installation
************
.. code-block:: bash

    pip install grids

Some of the dependencies for grids depend on system libraries and binaries which are not installed using a pip install.
The easiest solution is to conda install the dependency whose system dependencies you need e.g. cfgrib or rasterio.
You should not need to do this often.

.. code-block:: bash

    # example conda install to get system dependencies
    conda install -c conda-forge cfgrib rasterio netcdf4
    pip install grids

Interactive Demo
****************
View a live `demo python notebook <https://colab.research.google.com/gist/rileyhales/79761303df16127e0195e11425fc2d9d/grids-gist-demo.ipynb>`_ using Google Colaboratory and GitHub Gists.

Find a copy of the notebook on `GitHub Gists <https://gist.github.com/rileyhales/79761303df16127e0195e11425fc2d9d>`_.

TimeSeries Class Documentation
******************************

.. automodule:: grids
    :members: TimeSeries

Handling Time values
********************
Datetime values are extracted in one of 4 ways (controlled by the :code:`interp_units`, :code:`units_str`,
:code:`origin_format`, and :code:`strp_format` parameters), in this order of preference:

#. When :code:`interp_units` is True, interpret the time variable values as datetimes using time's units attribute.
    Override the file's units attribute or provide a missing one with the :code:`units_str` kwarg and the
    :code:`origin_format` kwarg if the date doesn't use YYYY-MM-DD HH:MM:SS format.
#. When a pattern is specified with :code:`strp_filename`, a datetime extracted from the filename is applied to all
   values coming from that dataset.
#. If a time variable exists, its numerical values are used without further interpretation.
#. The string file name is used if there is no time variable and no other options were provided.

Speed Test Results
******************

.. csv-table::
    :file: speed_test_times.csv
    :header-rows: 1
