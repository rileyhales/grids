import grids
print(grids.__version__)

# some other dependencies which you need to get a sample shapefile from esri
import tempfile
import os
import json
import requests

import xarray as xr


# data = ["https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"]
# var = 'Temperature_surface'
# dim_order = ('time', 'lat', 'lon', )
# myseries = grids.TimeSeries(files=data, var=var, dim_order=dim_order)

a = xr.open_dataset("https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best")
print(a)
exit()
# point_timeseries = myseries.point(None, 41.9, 12.5)
# print(point_timeseries)

boundingbox_timeseries = myseries.bound((None, 40, 12), (None, 41, 13))
print(boundingbox_timeseries)

# italy_url = 'https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/World__Countries_Generalized_analysis_trim/FeatureServer/0/query?f=pgeojson&outSR=4326&where=NAME+%3D+%27Italy%27'
# italy_json = requests.get(url=italy_url).json()
# italy_path = os.path.join(tempfile.gettempdir(), 'italy.json')
# with open(italy_path, 'w') as file:
#   file.write(json.dumps(italy_json))
#
# series.epsg = 4326
#
# italy_timeseries = myseries.shape(italy_path)