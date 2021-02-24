import grids
print(grids.__version__)

files = ["https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"]
var = 'Temperature_surface'
dim_order = ('time1', 'lat', 'lon', )
myseries = grids.TimeSeries(files=files, var=var, dim_order=dim_order)
myseries.t_var = 'time1'

point_timeseries = myseries.point(None, 41.9, 12.5)
print(point_timeseries)

boundingbox_timeseries = myseries.bound((None, 40, 12), (None, 41, 13))
print(boundingbox_timeseries)
