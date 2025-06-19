"""
Yuri Tamama

Code to estimate the arrival and rise time of moonquakes using the 
signal-to-noise ratio (SNR) function

"""
# Imports
import pandas as pd
from obspy import read,UTCDateTime
from datetime import datetime, timedelta
import numpy as np
import os
from joblib import Parallel, delayed
import sys
import obspy
import obspy.signal
from scipy import optimize

# Import functions
sys.path.insert(0,'../functions/')
from moon2data import *
from pick_and_quality import *

########################################################################

# Load catalog of isolated and repeating moonquakes
mqdir1 = '../catalogs/FC_catalogs/'
cat = pd.read_csv(mqdir1 + 'GradeABCD_avg_arrivals_catalog.csv')
cat.drop(list(cat.filter(regex='Unnamed|index')), axis=1, inplace=True)
evids_all = np.unique(cat.evid.tolist())

# Inputs to obtain waveforms
minfreq = 3
maxfreq = 35
befwin = 60
aftwin = 60
parentdir = '/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'
winlen = 10
stepby = 0.1

# Directory to save results (replace with your own directory)
savedir = '/data/ytamama/Apollo17/catalogs/picks/'

# Run code in parallel
num_cores = 15
Parallel(n_jobs=num_cores)(delayed(pick_wrapper)(evid,cat,befwin,aftwin,minfreq,maxfreq,parentdir,winlen,stepby,savedir) for evid in evids_all)

# Add results to existing catalog
picktimes = []
risetimes = []
emergences = []
rmses = []
evids = []
grades = []
geophones = []
geonums = np.array([1, 2, 3, 4])
for evid in evids_all:

    # Obtain dataframe corresponding to event
    fname = f'{savedir}{evid}_picks.csv'
    pick_cat = pd.read_csv(fname)
    
    # Iterate through geophones
    for num in geonums:
        pickrow = pick_cat.loc[pick_cat.geophone == num].reset_index().iloc[0]
 
        # Obtain pick time, rise time, emergence, etc.
        evids.append(pickrow.evid)
        grades.append(pickrow.grade)
        geophones.append(pickrow.geophone)
        picktimes.append(pickrow.picktime)
        risetimes.append(pickrow.risetime)
        emergences.append(pickrow.emergence_s)
        rmses.append(pickrow.rmse)

# Append to moonquake catalog
d = {'evid':evids, 'grade':grades, 'geophone':geophones, 'picktime':picktimes, 
     'risetime':risetimes, 'emergence_s':emergences, 'rmse_gauss':rmses}
combined_cat = pd.DataFrame(data = d)
mqdir2 = '../catalogs/quality_control/'
combined_cat.to_csv(mqdir2 + 'A17_moonquakes_catalog.csv')

