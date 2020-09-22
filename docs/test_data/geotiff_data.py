import datetime
import glob

import pandas as pd

import temporal_informatics as tin

files = sorted(glob.glob('/Users/riley/spatialdata/geotiff_data/*.tif'))
print(len(files))
var = 0
dim_order = ('y', 'x')
ts = tin.TimeSeries(files=files, var=var, dim_order=dim_order, strp_filename='GLDAS_NOAH025_3H.A%Y%m%d.%H00.021.nc4.tif', epsg=4326)

t1 = datetime.datetime.now()
a = ts.point(10, 10)
t2 = datetime.datetime.now()
b = ts.range((10, 10), (20, 20))
t3 = datetime.datetime.now()
c = ts.spatial('/Users/riley/spatialdata/shapefiles/utah/utah.shp')
t4 = datetime.datetime.now()

pd.DataFrame({
    'point': ((t2 - t1).total_seconds(),),
    'range': ((t3 - t2).total_seconds(),),
    'shape': ((t4 - t3).total_seconds(),),
}).to_csv('rasterio.csv')
