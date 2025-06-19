"""
Yuri Tamama

Code to estimate the duration of moonquake signals

"""
# Import packages
import pandas as pd
from obspy import read,UTCDateTime
from datetime import datetime, timedelta
import numpy as np
import os
import glob
import sys

# Import functions
fxndir = '../functions/'
sys.path.insert(0,fxndir)
from moon2data import *

########################################################################

# INPUTS
minfreq = 3
maxfreq = 35
befwin = 10
aftwin = 600
parentdir = '/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'

# Load picks
mqdir = '../catalogs/final_catalogs/'
cat = pd.read_csv(mqdir + 'A17_moonquakes_catalog_HQ_final.csv')
cat.drop(list(cat.filter(regex='Unnamed|index')), axis=1, inplace=True)

# Iterate through picked arrivals and estimate PGV
durations = []
for r in np.arange(0,len(cat)):

    # Obtain seismogram
    row = cat.iloc[r]
    arrtime = datetime.strptime(row.picktime_SNR,'%Y-%m-%d %H:%M:%S.%f')
    geonum = row.geophone
    st = moon2sac(arrtime,geonum,befwin,aftwin,minfreq,maxfreq,parentdir)
    tr_times = st.traces[0].times() - befwin
    tr_data = st.traces[0].data
    
    # Calculate envelope 
    data_env = obspy.signal.filter.envelope(tr_data)
    
    # Isolate preceding noise
    winlen = 5
    stepby = 0.1
    noise_times = tr_times[(tr_times < -5) & (tr_times >= -5 - winlen)]
    noise_env = data_env[(tr_times < -5) & (tr_times >= -5 - winlen)]
    mean_noise = np.mean(noise_env)
    
    # Isolate signal
    signal_times = tr_times[tr_times > 0]
    signal_env = data_env[tr_times > 0]
    
    # Slide a moving window through the signal and compare amplitude with noise
    starttime = signal_times[0]
    endtime = signal_times[0] + winlen
    signal_end_time = -1
    while endtime < np.max(signal_times):
        # Obtain window
        window = signal_env[(signal_times < endtime) & (signal_times >= starttime)]
        mean_window = np.mean(window)
        if mean_window < mean_noise:
            signal_end_time = starttime
            break
        else:
            starttime += stepby
            endtime += stepby
    if signal_end_time < 0:
        signal_end_time = aftwin
    
    # Append
    durations.append(signal_end_time)

# 
cat['duration_s'] = durations
cat.to_csv(mqdir + 'A17_moonquakes_catalog_HQ_final.csv')

