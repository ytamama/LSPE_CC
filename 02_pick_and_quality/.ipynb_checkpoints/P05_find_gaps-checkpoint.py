"""
Yuri Tamama

Find gaps in each waveform and the maximum length across all gaps

"""
# Import packages
import pandas as pd
from obspy import read,UTCDateTime
from datetime import datetime, timedelta
import numpy as np
import os
import glob
import sys
import obspy.signal
import random

# Import functions
fxndir = '../functions/'
sys.path.insert(0,fxndir)
from moon2data import *

########################################################################

# Load moonquake catalog
mqdir = '../catalogs/quality_control/'
combined_cat = pd.read_csv(mqdir + 'A17_moonquakes_catalog_nodupes.csv')
combined_cat.drop(list(combined_cat.filter(regex='Unnamed|index')), axis=1, inplace=True)

# Inputs to obtain waveforms
parentdir = '/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'
minfreq = 3
maxfreq = 35
befwin = 60
aftwin = 60

# Iterate through waveforms and calculate maximum gap length
maxgaps = []
for r in np.arange(0,len(combined_cat)):
    row = combined_cat.iloc[r]
    arrtime = datetime.strptime(row.picktime,'%Y-%m-%d %H:%M:%S.%f')
    geonum = row.geophone
    evid = row.evid

    # Obtain seismogram
    st = moon2sac(arrtime,geonum,befwin,aftwin,minfreq,maxfreq,parentdir)
    trtimes = st.traces[0].times()
    trdata = st.traces[0].data

    # Obtain envelope
    data_env = obspy.signal.filter.envelope(trdata)
    
    # Iterate through data
    maxgap = 0
    flag = 0
    t1 = 0
    t2 = 0
    for t in np.arange(0,len(data_env)):
        val = data_env[t]
        if (val < 0.00001) & (flag == 0):
            t1 = trtimes[t]
            t2 = t1
            flag = 1
        elif (val < 0.00001) & (flag == 1):
            t2 = trtimes[t]

        elif (val > 0.00001) & (flag == 1):
            flag = 0
            gap = t2 - t1
            if gap > maxgap:
                maxgap = gap
    #
    maxgaps.append(maxgap)

# Save
combined_cat['max_gap_len_s'] = maxgaps
combined_cat.to_csv(mqdir + 'A17_moonquakes_catalog_nodupes.csv')

