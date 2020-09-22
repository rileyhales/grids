import datetime
import glob

import pandas as pd

import temporal_informatics as tin

files = sorted(glob.glob('/Users/riley/spatialdata/thredds/gldas/raw/*.nc4'))
var = 'Tair_f_inst'
dim_order = ('time', 'lat', 'lon')
ts = tin.TimeSeries(files=files, var=var, dim_order=dim_order, interp_units=True, epsg=4326)

for engine in ('xarray', 'netcdf4', 'h5py', ):
    ts.engine = engine

    print(f'working on engine: {engine}')

    t1 = datetime.datetime.now()
    ts.point(None, 10, 10)
    t2 = datetime.datetime.now()
    ts.range((None, 10, 10), (None, 20, 20))
    t3 = datetime.datetime.now()
    ts.spatial('/Users/riley/spatialdata/shapefiles/utah/utah.shp')
    t4 = datetime.datetime.now()

    pd.DataFrame({
        'point': ((t2 - t1).total_seconds(),),
        'range': ((t3 - t2).total_seconds(),),
        'shape': ((t4 - t3).total_seconds(),),
    }).to_csv(f'{engine}.csv')
