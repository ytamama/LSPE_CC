"""
Yuri Tamama

Code to cross-correlate the Grade A, B, C, and D Apollo 17 moonquakes, with the records of the same moonquake but on other geophones 

"""
# Import packages
import pandas as pd
from obspy import read,UTCDateTime
from datetime import datetime, timedelta
import numpy as np
import os
import glob
from joblib import Parallel, delayed
import sys

# Import functions
fxndir = '../functions/'
sys.path.insert(0,fxndir)
from moon2data import *
from cc_moonquakes import *

########################################################################

# INPUTS
minfreq = 3
maxfreq = 35
befwin = 60
aftwin = 60
# Max. lag
maxshift = 20

# Location of catalogs
mqdir1 = '../catalogs/quality_control/'
# Directory containing moonquakes (replace with where you store the data)
parentdir = '/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'
# Output for each file (replace to where you'd like to store the output)
output_directory = '/data/ytamama/Apollo17/catalogs/cc_geophones/'

# Load catalog of picked arrivals 
mooncat_full = pd.read_csv(mqdir1 + 'A17_moonquakes_catalog_nodupes.csv')
mooncat_full.drop(list(mooncat_full.filter(regex='Unnamed|index')), axis=1, inplace=True)
evids = np.unique(mooncat_full.evid.tolist())

# Iterate through events -- parallelize code
num_cores = 15
demean = 0
env = 1
Parallel(n_jobs=num_cores)(delayed(cc_A17_geophones_wrapper)(mooncat_full, evid, befwin, aftwin, minfreq, maxfreq, parentdir, output_directory, demean, maxshift, env) for evid in evids)

# Combine files
all_files = glob.glob(f'{output_directory}*.csv')
for f in np.arange(0,len(all_files)):
    fname = all_files[f]
    align_df = pd.read_csv(fname)
    align_df = align_df[['evid','cc_12','dt_12','cc_13','dt_13','cc_14','dt_14','cc_23','dt_23','cc_34','dt_34','cc_24','dt_24','minfreq','maxfreq','grade','grade_new']]
    if f == 0:
        align_df_all = align_df
    else:
        align_df_all = pd.concat([align_df_all, align_df])
        
# Save catalog
mqdir2 = '../catalogs/cc_geophones/'
align_df_all.to_csv(mqdir2 + 'A17_cc_geophones_catalog.csv')
    
    