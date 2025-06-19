"""
Yuri Tamama

Code to calculate the peak ground velocity of each moonquake, encompassing a 
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
fxndir = '../functions/'
sys.path.insert(0,fxndir)
from moon2data import *

########################################################################

# INPUTS
minfreq = 3
maxfreq = 35
befwin = 0
parentdir = '/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'

# Load picks
mqdir = '../catalogs/final_catalogs/'
cat = pd.read_csv(mqdir + 'A17_moonquakes_catalog_HQ_final.csv')
cat.drop(list(cat.filter(regex='Unnamed|index')), axis=1, inplace=True)

# Iterate through picked arrivals and estimate PGV
pgvs = []
for r in np.arange(0,len(cat)):

    # Obtain seismogram
    row = cat.iloc[r]
    arrtime = datetime.strptime(row.picktime_SNR, '%Y-%m-%d %H:%M:%S.%f')
    geonum = row.geophone
    aftwin = row.emergence_s + 5
    st = moon2sac(arrtime,geonum,befwin,aftwin,minfreq,maxfreq,parentdir)
    tr_data = st.traces[0].data
    
    # Calculate PGV
    pgv = np.max(np.abs(tr_data))
    pgvs.append(pgv)
# 
cat['PGV'] = pgvs
cat.to_csv(mqdir + 'A17_moonquakes_catalog_HQ_final.csv')


