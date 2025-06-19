"""
Yuri Tamama

Code to cross-correlate high-quality records of Grade A, B, C, and D moonquakes 
with respect to one another to classify them into families 

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
# Length of signal 
befwin = 60
aftwin = 60
# Max. lag
maxshift = 30
# Geophone numbers
geonums = np.array([1, 2, 3, 4])

# Location of event catalog
mqdir = '../catalogs/quality_control/'
# Directory to save alignment catalogs
aligndir = '/data/ytamama/Apollo17/catalogs/cc_ABCD/alignment_cc/'
if not os.path.exists(aligndir):
    os.mkdir(aligndir)
# Directory containing moonquakes
parentdir = '/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'

# Load catalog of high-quality picks
mooncat_full = pd.read_csv(mqdir + 'A17_moonquakes_catalog_nodupes_HQ.csv')
mooncat_full.drop(list(mooncat_full.filter(regex='Unnamed|index')), axis=1, inplace=True)

# Event IDs
evids = np.unique(mooncat_full.evid.tolist())

# Parallelize 
num_cores = 25
env = 1
demean = 0
Parallel(n_jobs=num_cores)(delayed(cc_A17_wrapper)(mooncat_full, evid, befwin, aftwin, minfreq, maxfreq, parentdir, aligndir, demean, maxshift, env) for evid in evids)

