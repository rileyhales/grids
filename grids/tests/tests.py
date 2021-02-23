import grids
# some other dependencies which you need to get a sample from esri
import tempfile
import os
import json
import requests
import xarray as xr

print(grids.__version__)

a = xr.open_dataset("https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best")
print(a)
exit()
# point_timeseries = myseries.point(None, 41.9, 12.5)
# print(point_timeseries)

# boundingbox_timeseries = myseries.bound((None, 40, 12), (None, 41, 13))
# print(boundingbox_timeseries)
