"""
Code to calculate the rock and regolith temperature of the Apollo 17 site at the time of a moonquake, as well as where that moonquake falls within one instance of the diurnal temperature cycle. 

Temperatures from Molaro et al. (2017)

"""
# Import libraries
import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta
import os
from scipy.signal import *

############################################################################################################

def load_temp_cats(catdir):
    '''
    Function to load the temperature model by Molaro et al. (2017), as well 
    as a table showing the boundaries of each named interval (e.g. sunrise, 
    sunset, night).
    
    INPUT
    catdir : Directory containing the temperature model
    
    OUTPUTS
    temp_df : Table containing the lunar temperature model by Molaro et al. 
              (2017)
    int_df : Table containing the times + names of time intervals (e.g. 
             sunrise, sunset, night) within the temperature cycle 
    
    '''
    temp_df = pd.read_csv(catdir + 'longterm_thermal_data_Molaro.csv')
    int_df = pd.read_csv(catdir + 'thermally_defined_night_sunrise_sunset_times.csv')
    int_df = int_df.sort_values(by=['start times'],ignore_index=True)
    
    # Return
    return temp_df, int_df


def moonquake_temperature(arrtime,temp_df,int_df):
    '''
    Function to interpolate for the temperature of the regolith and rock at the time 
    of arrival of a moonquake, from the temperature model by Molaro et al. (2017)
    
    INPUTS
    arrtime : Datetime object, containing the arrival time of a moonquake at a 
              geophone
    temp_df : Dataframe containing the temperature model from Molaro et al. 
              (2017)
    int_df : Dataframe containing the start times of the temperature 
             intervals: night, sunrise, and sunset
                  
    OUTPUTS
    reg_temp : Interpolated temperature of the regolith at the time of the moonquake, in K
    reg_temp_slope : Slope of the regolith temperature curve at the time of the moonquake
    rock_temp : Interpolated temperature of lunar rock at the time of the moonquake, in K
    rock_temp_slope : Slope of the rock temperature curve at the time of the moonquake
    
    '''
    # Retrieve temperature data from dataframe
    temp_reg = np.array(temp_df.T_reg_K.tolist())
    temp_rock = np.array(temp_df.T_rock_K.tolist())
    time_ymd = [datetime.strptime(mqdate, '%Y-%m-%d %H:%M:%S') for mqdate in (temp_df['Time_YMD']).tolist()]
    
    # Obtain time steps and temperatures surrounding this moonquake
    # Before
    times_bef = [x for x in time_ymd if x <= arrtime]
    befind = len(times_bef)-1
    beftime_s = (time_ymd[befind] - arrtime).total_seconds()
    reg_temp_bef = temp_reg[befind]
    rock_temp_bef = temp_rock[befind]
    
    # After
    times_aft = [x for x in time_ymd if x >= arrtime]
    aftind = len(time_ymd) - len(times_aft)
    afttime_s = (time_ymd[aftind] - arrtime).total_seconds()
    reg_temp_aft = temp_reg[aftind]
    rock_temp_aft = temp_rock[aftind]

    # Interpolate temperatures to moonquake arrival
    reg_temp = np.interp(0, [beftime_s,afttime_s], [reg_temp_bef, reg_temp_aft])
    rock_temp = np.interp(0, [beftime_s,afttime_s], [rock_temp_bef, rock_temp_aft])

    # Estimate the derivative of the temperature curve 
    reg_temp_slope = (reg_temp_aft - reg_temp_bef)/(afttime_s - beftime_s)
    rock_temp_slope = (rock_temp_aft - rock_temp_bef)/(afttime_s - beftime_s)
    
    # Output temperature
    return reg_temp, reg_temp_slope, rock_temp, rock_temp_slope



def get_time_wrt_interval(arrtime,temp_df,int_df):
    '''
    Function to figure out where a moonquake falls within one diurnal cycle 
    of the temperature model by Molaro et al. (2017). 

    INPUTS
    arrtime : Datetime object, containing the arrival time of a moonquake at a geophone
    temp_df : Dataframe containing the temperature model from Molaro et al. (2017)
    int_df : Dataframe containing the start times of the temperature intervals: night, sunrise, and sunset
                  
    OUTPUTS
    arrtime_int_s : "Where" a moonquake falls within one temperature cycle, in seconds
    arrtime_int_days : Same as above, but in decimal days
    diurnal_interval : "night", "sunrise," or "sunset"
    
    '''
    # Retrieve temperature data from dataframe
    time_ymd = [datetime.strptime(mqdate, '%Y-%m-%d %H:%M:%S') for mqdate in (temp_df['Time_YMD']).tolist()]
    
    # Retrieve start times of night intervals
    night_cat = int_df[int_df['interval_name'].str.contains('night')]
    night_cat = night_cat.reset_index()
    night_start_times = [datetime.strptime(mqdate, '%Y-%m-%d %H:%M:%S') for mqdate in (night_cat['start times']).tolist()]

    # Determine interval number 
    night_start_earlier = []
    for m in np.arange(len(night_start_times)):
        start_time = night_start_times[m]
        if start_time <= arrtime:
            night_start_earlier.append(start_time)
    # 
    interval_num = len(night_start_earlier)
    evt_int_df = int_df[int_df['interval_name'].str.contains(str(interval_num))]

    # Starting time of present interval
    interval_start_time = datetime.strptime(evt_int_df.iloc[0]['start times'],'%Y-%m-%d %H:%M:%S')

    # Obtain time relative to interval 
    arrtime_int_s = (arrtime - interval_start_time).total_seconds()
    arrtime_int_days = (arrtime_int_s) / (60*60*24)
    
    # End times of night, sunrise, and sunset
    night_end_time = datetime.strptime(evt_int_df.iloc[0]['end times'],'%Y-%m-%d %H:%M:%S')
    night_end_days = ((night_end_time - interval_start_time).total_seconds()) / (60*60*24)
    sunrise_end_time = datetime.strptime(evt_int_df.iloc[1]['end times'],'%Y-%m-%d %H:%M:%S')
    sunrise_end_days = ((sunrise_end_time - interval_start_time).total_seconds()) / (60*60*24)
    if interval_num < 9:
        sunset_end_time = datetime.strptime(evt_int_df.iloc[2]['end times'],'%Y-%m-%d %H:%M:%S')
        sunset_end_days = ((sunset_end_time - interval_start_time).total_seconds()) / (60*60*24)

        # Check whether moonquake occurs at night, sunrise, or sunset
        if arrtime_int_days <= night_end_days:
            diurnal_interval = 'night' + str(interval_num)
        elif arrtime_int_days <= sunrise_end_days:
            diurnal_interval = 'sunrise' + str(interval_num)
        else:
            diurnal_interval = 'sunset' + str(interval_num)
            
    else:
    # Check whether moonquake occurs at night or sunrise
        if arrtime_int_days <= night_end_days:
            diurnal_interval = 'night' + str(interval_num)
        elif arrtime_int_days <= sunrise_end_days:
            diurnal_interval = 'sunrise' + str(interval_num)
    
    return arrtime_int_s, arrtime_int_days, diurnal_interval



def get_one_cycle(temp_df,int_df):
    '''
    Function to isolate one instance of a diurnal temperature cycle from the model of Molaro et al. (2017)
    
    INPUTS
    temp_df : Dataframe containing the temperature model from Molaro et al. (2017)
    int_df : Dataframe containing the start times of the temperature intervals: night, sunrise, and sunset
                  
    OUTPUTS
    time_days : Time, in decimal days
    temps_K_reg : Temperature of the regolith corresponding to each time step, in K
    temps_K_rock : Temperature of the rock corresponding to each time step, in K
    
    '''
    # Retrieve bounds for one interval
    starttime = int_df.iloc[0]['start times']
    endtime = int_df.iloc[2]['end times']
    
    # Retrieve data for one cycle
    temp_df_cycle = temp_df.loc[(temp_df.Time_YMD >= starttime) & (temp_df.Time_YMD <= endtime)]
    # Retrieve temperature data 
    temps_K_reg = np.array(temp_df_cycle.T_reg_K.tolist())
    temps_K_rock = np.array(temp_df_cycle.T_rock_K.tolist())
    times_ymd = [datetime.strptime(mqdate, '%Y-%m-%d %H:%M:%S') for mqdate in (temp_df_cycle['Time_YMD']).tolist()]

    # Obtain time differences relative to starting time
    starttime = min(times_ymd)
    time_seconds = [(temp_time - starttime).total_seconds() for temp_time in times_ymd]
    time_days = np.array(time_seconds) / (60*60*24)
    
    # Return outputs
    return time_days, temps_K_reg, temps_K_rock 
    
    