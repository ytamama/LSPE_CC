"""
Yuri Tamama

Code to calculate the regolith and rock temperatures corresponding to 
Grade A, B, C, and D moonquakes

"""
# Import packages
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import glob
import sys

# Import functions
fxndir = '../functions/'
sys.path.insert(0,fxndir)
from moon2data import *
from moonquake_temperatures import *

########################################################################

# Load catalog
mqdir = '../catalogs/final_catalogs/'
cat = pd.read_csv(mqdir + 'A17_moonquakes_catalog_HQ_final.csv')
cat.drop(list(cat.filter(regex='Unnamed|index')), axis=1, inplace=True)

# Load temperature cycle
tempdir = '../catalogs/temperature/'
temp_df, int_df = load_temp_cats(tempdir)

# Iteratively calculate temperature of each moonquake
reg_temps = []
rock_temps = []
interval_day_numbers = []
diurnal_intervals = []
for m in np.arange(0,len(cat)):
    
    # Arrival time
    moonrow = cat.iloc[m]
    arrtime = datetime.strptime(moonrow.picktime_SNR, '%Y-%m-%d %H:%M:%S.%f')

    # Interpolate for regolith and rock temperature and slope
    fxnouts = moonquake_temperature(arrtime,temp_df,int_df)
    reg_temp = fxnouts[0]
    rock_temp = fxnouts[2]
    
    # Calculate where moonquake falls in one instance of temperature cycle
    arrtime_int_s, arrtime_int_days, diurnal_interval = get_time_wrt_interval(arrtime,temp_df,int_df)

    # Append results
    reg_temps.append(reg_temp)
    rock_temps.append(rock_temp)
    interval_day_numbers.append(arrtime_int_days)
    diurnal_intervals.append(diurnal_interval)
# 
cat['Regolith_Temp_K'] = np.array(reg_temps)
cat['Rock_Temp_K'] = np.array(rock_temps)
cat['interval_day_number'] = np.array(interval_day_numbers)
cat['diurnal_interval'] = np.array(diurnal_intervals)
cat.to_csv(mqdir + 'A17_moonquakes_catalog_HQ_final.csv')

