"""
Yuri Tamama

Code to calculate the SNR of each moonquake, encompassing a 
60 second window before and after the picked arrival

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
sys.path.insert(0,'../functions/')
from moon2data import *

#########################################################################################

# INPUTS
minfreq = 3
maxfreq = 35
befwin = 60
aftwin = 60
parentdir = '/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'

# Load picks
mqdir = '../catalogs/quality_control/'
cat = pd.read_csv(mqdir + 'A17_moonquakes_catalog_nodupes.csv')
cat.drop(list(cat.filter(regex='Unnamed|index')), axis=1, inplace=True)

# Iteratively calculate SNR
snrs = []
for r in np.arange(0,len(cat)):
    
    # Obtain RMSE
    row = cat.iloc[r]
    arrtime = datetime.strptime(row.picktime,'%Y-%m-%d %H:%M:%S.%f')
    geonum = row.geophone
    
    # Obtain seismogram
    st = moon2sac(arrtime,geonum,befwin,aftwin,minfreq,maxfreq,parentdir)
    trtimes = st.traces[0].times() - befwin
    trdata = st.traces[0].data
    
    # Calculate envelope
    data_env = obspy.signal.filter.envelope(trdata)

    # Calculate SNR of entire signal
    noise_env = data_env[trtimes < 0]
    noise_env_mean = np.mean(noise_env)
    signal_env = data_env[trtimes > 0]
    signal_env_mean = np.mean(signal_env)
    snr = signal_env_mean / noise_env_mean
    snrs.append(snr)

cat['SNR'] = snrs
cat.to_csv(mqdir + 'A17_moonquakes_catalog_nodupes.csv',index=False)

