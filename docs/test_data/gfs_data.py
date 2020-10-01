import datetime
import glob

import pandas as pd

import tin as tin

files = sorted(glob.glob('/Users/riley/Downloads/gfs_20200101.grb2') * 500)
var = '4lftx'
dim_order = ('latitude', 'longitude')
ts = tin.TimeSeries(files=files, var=var, dim_order=dim_order, strp_filename='gfs_%Y%m%d.grb2', epsg=4326)
ts.xr_kwargs = {'filter_by_keys': {'typeOfLevel': 'surface'}}

for engine in ('cfgrib', ):
    ts.engine = engine

    print(f'working on engine: {engine}')

    t1 = datetime.datetime.now()
    ts.point(10, 10)
    t2 = datetime.datetime.now()
    ts.range((10, 10), (20, 20))
    t3 = datetime.datetime.now()
    ts.spatial('/Users/riley/spatialdata/shapefiles/utah/utah.shp')
    t4 = datetime.datetime.now()

    pd.DataFrame({
        'point': ((t2 - t1).total_seconds(),),
        'range': ((t3 - t2).total_seconds(),),
        'shape': ((t4 - t3).total_seconds(),),
    }).to_csv(f'{engine}.csv')
