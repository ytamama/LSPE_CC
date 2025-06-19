"""
Yuri Tamama

Code to estimate the incident azimuth of high-quality moonquakes, based on arrival times 
picked using the SNR function

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
from calc_azimuth import *

########################################################################

# Retrieve Grade AA and BB moonquakes across isolated and repeating events
mqdir = '../catalogs/final_catalogs/'
cat = pd.read_csv(mqdir + 'A17_moonquakes_catalog_HQ_final.csv')
cat_HQ = cat.loc[(cat.grade_new == 'AA') | (cat.grade_new == 'BB')]
evids_AABB = np.unique(cat_HQ.evid.tolist())

# Directory to save results
output_directory = '/data/ytamama/Apollo17/catalogs/cc_azimuths/'

# Number of iterations for gradient descent
num_iter = 20

# Iterate through events and calculate azimuth from picked onset times
num_cores = 15
Parallel(n_jobs=num_cores)(delayed(azimuth_wrapper_picks)(cat_HQ, evid, num_iter, output_directory) for evid in evids_AABB)
