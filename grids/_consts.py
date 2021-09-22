ALL_ENGINES = ('xarray', 'opendap', 'auth-opendap', 'netcdf4', 'cfgrib', 'pygrib', 'h5py', 'rasterio',)
ALL_STATS = ('mean', 'median', 'max', 'min', 'sum', 'std',)
BOX_STATS = ('max', '75%', 'median', 'mean', '25%', 'min')

T_VARS = ('time', )
SPATIAL_X_VARS = ('x', 'lon', 'longitude', 'longitudes', 'degrees_east', 'eastings',)
SPATIAL_Y_VARS = ('y', 'lat', 'latitude', 'longitudes', 'degrees_north', 'northings',)

NETCDF_EXTENSIONS = ('.nc', '.nc4')
GRIB_EXTENSIONS = ('.grb', 'grb2', '.grib', '.grib2')
HDF_EXTENSIONS = ('.h5', '.hd5', '.hdf5')
GEOTIFF_EXTENSIONS = ('.gtiff', '.tiff', 'tif')
