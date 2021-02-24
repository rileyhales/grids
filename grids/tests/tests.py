import grids
import requests
import os
import tempfile
import json

print(grids.__version__)

files = ["https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"]
var = 'Temperature_surface'
dim_order = ('time1', 'lat', 'lon', )
myseries = grids.TimeSeries(files=files, var=var, dim_order=dim_order)
myseries.t_var = 'time1'

# point_timeseries = myseries.point(None, 41.9, 12.5)
# print(point_timeseries)
# boundingbox_timeseries = myseries.bound((None, 40, 12), (None, 41, 13))
# print(boundingbox_timeseries)

italy_url = 'https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/World__Countries_Generalized_analysis_trim/FeatureServer/0/query?f=pgeojson&outSR=4326&where=NAME+%3D+%27Italy%27'
italy_json = requests.get(url=italy_url).json()
italy_path = os.path.join(tempfile.gettempdir(), 'italy.json')
with open(italy_path, 'w') as file:
    file.write(json.dumps(italy_json))
myseries.epsg = 4326
a = myseries.shape(italy_path)
print(a)