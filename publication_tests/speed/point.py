import grids
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import datetime
import glob


files = list(sorted(glob.glob('/Users/rchales/data/gldas/*.nc4'))) * 100
times = {}

for engine in ('xarray', 'netcdf4'):
    start = datetime.datetime.utcnow()
    var = 'Tair_f_inst'
    dim_order = ('time', 'lat', 'lon', )
    ts = grids.TimeSeries(files, var, dim_order, engine='netcdf4')
    ts.point(None, 40, -110)
    end = datetime.datetime.utcnow()
    times[f'grids_{engine}'] = (end - start).total_seconds()


# native xarray speeds
start = datetime.datetime.utcnow()
values = []
dates = []
for file in files:
    a = xr.open_dataset(file)
    var = 'Tair_f_inst'
    x_val = -110
    y_val = 40
    x_array_idx = (np.abs(a['lon'].values - x_val)).argmin()
    y_array_idx = (np.abs(a['lat'].values - y_val)).argmin()
    values.append(float(a[var][:, y_array_idx, x_array_idx][:]))
    dates.append(a['time'][:][0].values)
df = pd.DataFrame(values, index=times)
end = datetime.datetime.utcnow()
times['native_xarray'] = (end - start).total_seconds()


# native netcdf4 speeds
start = datetime.datetime.utcnow()
values = []
dates = []
for file in files:
    a = nc.Dataset(file)
    var = 'Tair_f_inst'
    x_val = -110
    y_val = 40
    x_array_idx = (np.abs(a['lon'][:] - x_val)).argmin()
    y_array_idx = (np.abs(a['lat'][:] - y_val)).argmin()
    values.append(float(a[var][:, y_array_idx, x_array_idx]))
    dates.append(a['time'][:][0].values)
df = pd.DataFrame(values, index=times)
end = datetime.datetime.utcnow()
times['native_netcdf'] = (end - start).total_seconds()

print(times)
pd.DataFrame(times).to_csv('comparison.csv')
