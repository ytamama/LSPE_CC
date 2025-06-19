"""
Takes input LSPE daily ASCII data and converts it to hourly SAC files

Adapted from original code by Francesco Civilini (see https://github.com/civilinifr/thermal_mq_analysis)

"""

# Import packages
import pandas as pd
import numpy as np
import glob
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper
import os
from datetime import datetime, timedelta
from obspy.io.sac.sactrace import SACTrace
import time

# Import functions
from despike import *

##############################################################################################

def interp_data(in_time, in_data, out_time):
    """
    From P01_ascii2sac.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis
    
    Interpolates the input data to a new target samplerate
    :param in_time: [Vector] Input time
    :param in_data: [Vector] Input data
    :param out_time: [Vector] New target samplerate
    :return: [Vector] Interpolated data
    
    """
    # Remove the mean and cosine taper (due to the large amount of data, only do 0.05% taper on each side)
    trace_nomean = in_data - np.nanmean(in_data)
    n = len(trace_nomean)
    taper_function = cosine_taper(n, p=0.001)
    trace_taper = trace_nomean * taper_function

    # Interpolate the data
    f = interp1d(in_time, trace_taper)
    out_data_interp = f(out_time)

    return out_data_interp



def cut_data(input_interp_time, input_interp_data, sta_name, input_file_start_time,
             input_year, input_jday, out, target_delta=1/117.6, ds=1):
    """
    Adapted from P01_ascii2sac.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis
    
    Cuts the input interpolated data into hourly time segments and despikes them

    :param input_interp_time: [Vector] Input interpolated time
    :param input_interp_data: [Vector] Input interpolated seismic data
    :param sta_name: [String] Station name
    :param input_jday: [Int] Julian day
    :param input_year: [Int] Input year
    :param input_file_start_time: [Datetime] Start time of day (entire file)
    :param out: [String] Folder where we output the SAC file
    :param target_delta: [Scalar] Time difference between samples
    :param ds: [Int] Whether to despike the data. 1 if yes (default). 0 if no. 

    :return: Nothing is returned
    """
    progress_month = '{:02d}'.format(input_file_start_time.month)
    progress_day = '{:02d}'.format(input_file_start_time.day)
    clock_start_time = time.time()

    # print('------------------------------------')
    # print(f'Working on {sta_name} {input_year}-{progress_month}-{progress_day}...')

    # Set the hourly start and end times (in seconds)
    start_times = []
    for start_hour_index in np.arange(24):
        start_times.append(start_hour_index * 3600)

    # Find the index corresponding to the hourly segments
    for start_hour_index in np.arange(24):
        if start_hour_index == 23:
            end_time = input_interp_time[-1]
        else:
            end_time = start_times[start_hour_index + 1]

        indices = np.intersect1d(np.where(input_interp_time >= start_times[start_hour_index]),
                                 np.where(input_interp_time < end_time))

        hour_type = '{:01d}'.format(start_hour_index)
        output_filename = f'{input_year}{progress_month}{progress_day}_17{sta_name}_{hour_type}_ID'

        # If there isn't any data for the time period, skip this hour
        if len(indices) == 0:
            print(f'{output_filename} : No data')
            continue

        # Set up time variables for this hour that we can call to
        hourly_time = input_interp_time[indices]
        hourly_data = input_interp_data[indices]
        hour_start_time = input_file_start_time + timedelta(seconds=hourly_time[0])

        # Despike the data? 
        if ds == 1:
            despiked_data = despike_YT(hourly_data)

        # Save to sac
        # Setup the SAC start header
        header = {'kstnm': 'ST17', 'kcmpnm': sta_name, 'nzyear': int(input_year),
                  'nzjday': input_jday, 'nzhour': hour_start_time.hour, 'nzmin': hour_start_time.minute,
                  'nzsec': hour_start_time.second, 'nzmsec': hour_start_time.microsecond/1000, 'delta': target_delta}
        # Correction: nzmsec should be entered in MILLISECONDS, not microseconds
        print(header)

        if ds == 1:
            # Replace infs and NaNs with zeroes
            despiked_data = np.nan_to_num(despiked_data, nan=0)
            despiked_data[np.isinf(despiked_data)] = 0
            sac = SACTrace(data=despiked_data, **header)
        else:
            # Replace infs and NaNs with zeroes
            hourly_data = np.nan_to_num(hourly_data, nan=0)
            hourly_data[np.isinf(hourly_data)] = 0
            sac = SACTrace(data=hourly_data, **header)  
        sac.write(f'{out}{output_filename}')
        print(f'{output_filename} : Completed')

    elapsed_time = time.time() - clock_start_time

    if ds == 1:
        print(
            f'Finished despiking {sta_name} {input_year}-{progress_month}-{progress_day}! '
            f'(Elapsed time: {np.round(elapsed_time / 60.0, decimals=1)} minutes)')
        
    else:
        print(
            f'Finished converting {sta_name} {input_year}-{progress_month}-{progress_day}! '
            f'(Elapsed time: {np.round(elapsed_time / 60.0, decimals=1)} minutes)')

    return



def process_ascii(input_file, output_directory, target_delta=1/117.6, ds = 1):
    """
    From P01_ascii2sac.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis
    
    Processes the input ASCII data into resampled, despiked sac traces.
    The input ASCII traces are one day files. The output is in hours.

    :param input_file: [String] Input file
    :param target_delta: [Float] Time between each sample for resampled trace in SECONDS
    :param ds: [Int] Whether to despike the data. 1 if yes (default). 0 if no. 
    :return:
    """

    # Read in the data (one day)
    df = pd.read_csv(input_file, delimiter=' ', header=None, encoding='unicode_escape')
    data_time = df[0].values - np.floor(df[0][0])
    geo1 = df[1].values
    geo2 = df[2].values
    geo3 = df[3].values
    geo4 = df[4].values
    
    # Replace NaNs and infs with zeroes
    # Geophone 1
    geo1 = np.nan_to_num(geo1, nan=0)
    geo1[np.isinf(geo1)] = 0
    # Geophone 2
    geo2 = np.nan_to_num(geo2, nan=0)
    geo2[np.isinf(geo2)] = 0
    # Geophone 3
    geo3 = np.nan_to_num(geo3, nan=0)
    geo3[np.isinf(geo3)] = 0
    # Geophone 4
    geo4 = np.nan_to_num(geo4, nan=0)
    geo4[np.isinf(geo4)] = 0

    # Create a resampled time vector
    # Time is currently a float between 0 and 1 representing in DAYS. So it must be converted to SECONDS
    data_time = data_time * 24*60*60
    time_interp = np.arange(np.ceil(data_time[0]), np.floor(data_time[-1]), target_delta)
    geo1_interp = interp_data(data_time, geo1, time_interp)
    geo2_interp = interp_data(data_time, geo2, time_interp)
    geo3_interp = interp_data(data_time, geo3, time_interp)
    geo4_interp = interp_data(data_time, geo4, time_interp)

    # Convert the start and end times into proper datestrings
    file_bn = os.path.basename(input_file)
    year = file_bn[0:4]
    month = file_bn[4:6]
    day = file_bn[6:8]
    file_start = datetime(int(year), int(month), int(day), 0, 0, 0)
    start_time = file_start + timedelta(seconds=time_interp[0])

    # Figure out the julian day corresponding to the year/date
    tt = start_time.timetuple()
    jday = tt.tm_yday

    # Create the output folder
    out_folder = f'{output_directory}{year}{month}{day}/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Split the data into hour segments and save the output
    cut_data(time_interp, geo1_interp, 'Geo1', file_start, year, jday, out_folder, target_delta, ds)
    cut_data(time_interp, geo2_interp, 'Geo2', file_start, year, jday, out_folder, target_delta, ds)
    cut_data(time_interp, geo3_interp, 'Geo3', file_start, year, jday, out_folder, target_delta, ds)
    cut_data(time_interp, geo4_interp, 'Geo4', file_start, year, jday, out_folder, target_delta, ds)

    return


