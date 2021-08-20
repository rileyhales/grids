"""
Copyright Riley Chad Hales 2021
All Rights Reserved
"""
# import datetime
# import os
# import shutil

import pandas as pd
# import requests
# import glob
# import grids

import plotly.graph_objs as go


def calc_most_recent_forecast_datetime(fcrange: str, previous: bool = False, steps_back: int = 1):
    start_date = datetime.datetime.utcnow()

    # compute previous forecast times if necessary
    if previous:
        # go back the number of hours between model runs
        if fcrange == 'short':
            start_date = start_date - datetime.timedelta(hours=1 * steps_back)
        else:
            start_date = start_date - datetime.timedelta(hours=6 * steps_back)

    start_hour = f'{start_date.hour:02}' if fcrange == 'short' else f'{start_date.hour // 6 * 6:02}'

    return start_date.strftime('%Y%m%d'), start_hour


def get_forecast_data(fcrange: str,
                      path: str,
                      start_date: str = None,
                      start_hour: str = None,
                      retry: bool = True,
                      retry_count: int = 0,
                      max_retry: int = 5) -> None:
    http_base = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/prod/nwm.'
    download_list = []

    # determine the start datetime
    if start_date is None or start_hour is None:
        start_date, start_hour = calc_most_recent_forecast_datetime(fcrange, previous=False)

    # setting some variable used to generate the list of filenames and paths on NOMADS
    if fcrange == 'short':
        ens_members = ('',)
        timesteps = [f'{i + 1:03}' for i in range(18)]
        dir = f'{fcrange}_range'
        ens_prefix = ''
    elif fcrange == 'medium':
        ens_members = (1, 2, 3, 4, 5, 6, 7)
        timesteps = [f'{(i + 1) * 3:03}' for i in range(20)]
        dir = f'{fcrange}_range_mem'
        ens_prefix = '_'
    else:  # elif fcrange == 'long':
        # ens_members = (1, 2, 3, 4)
        ens_members = (4,)
        timesteps = [f'{(i + 1) * 6:03}' for i in range(120)]
        dir = f'{fcrange}_range_mem'
        ens_prefix = '_'

    # download the first 20 timesteps of all 4 ensemble memebers
    for ens_num in ens_members:
        for time_step in timesteps:
            file_name = f'nwm.t{start_hour}z.{fcrange}_range.channel_rt{ens_prefix}{ens_num}.f{time_step}.conus.nc'
            url = f'{http_base}{start_date}/{dir}{ens_num}/{file_name}'
            download_list.append((file_name, url,))

    # download the urls
    try:
        for file_name, url in download_list:
            print(f'Downloading: {url}')
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(os.path.join(path, file_name), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
    except requests.HTTPError as e:
        print(f'Error retrieving {fcrange} range data for start date {start_date} and start hour {start_hour}')
        print(f'-->   Error encountered: {e}\n')
        if r.status_code == 404:
            print(f'-->   The data is not available, this forecast simulation may not be available through nomads yet.')
        if retry and retry_count < max_retry:
            retry_count += 1
            prev_start, prev_hour = calc_most_recent_forecast_datetime(fcrange, previous=True, steps_back=retry_count)
            print(f'Automatically retrying with the previous forecast day ({prev_start}) and time ({prev_hour})')
            get_forecast_data(fcrange, path=path, start_date=prev_start, start_hour=prev_hour,
                              retry=retry, retry_count=retry_count, max_retry=max_retry)
        else:
            raise RuntimeError('Unable to get data from Nomads. Try again later.')
    except Exception as e:
        print('Unexpected error. Aborting program')
        raise e

    return


# set path to download the nwm data
# path_to_nwm_directory = "/Users/rchales/nwm_data"

# if os.path.exists(path_to_nwm_directory):
#     shutil.rmtree(path_to_nwm_directory)
# os.mkdir(path_to_nwm_directory)

# download the nwm forecast data defaulting to the most recent forecast time
# get_forecast_data('long', path=path_to_nwm_directory)


# for member_number in (4, ):
#     # create the grids time series object used to execute queries
#     list_of_nwm_files = sorted(glob.glob(os.path.join(path_to_nwm_directory, f'*channel_rt_{member_number}*.nc')))
#     streamflow_variable = 'streamflow'
#     dim_order = ('feature_id', )
#     nwm_timeseries = grids.TimeSeries(list_of_nwm_files, streamflow_variable, dim_order, engine='xarray')
#
#     # query stream flow for a specific station
#     series = nwm_timeseries.point(14352926)
#     series.index = pd.to_datetime(series['datetime'])
#     del series['datetime']
#     series.set_axis([f'member_{member_number}', ], axis=1, inplace=True)
#     series.to_csv(f'member_{member_number}.csv')

master_df = pd.read_csv('national_water_model_extracted_timeseries.csv', index_col=0)
master_df.index = pd.to_datetime(master_df.index)

plot_lines = []
for i in (1, 2, 3, 4):
    plot_lines.append(
        go.Scatter(
            x=master_df.index,
            y=master_df[f'member_{i}'],
            line=dict(color='blue', width=5),
            showlegend=False
        )
    )
for i in (1, 2, 3, 4):
    plot_lines.append(
        go.Scatter(
            x=master_df.index,
            y=master_df[f'member_{i}'],
            line=dict(color='red', dash='dot'),
            showlegend=False
        )
    )
plot = go.Figure(plot_lines)
plot.update_xaxes(title='Datetime (UTC)')
plot.update_yaxes(title='Forecasted Discharge (ft^3/sec)')
plot.write_image('new_timeseries_plot.png')
plot.show()
