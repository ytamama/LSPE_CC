"""
Yuri Tamama

Code to estimate the incident azimuth of high-quality moonquakes, based on the lag between geophones obtained via cross-correlation

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

# Load catalog of correlation coefficients between geophones
geodir = '../catalogs/cc_geophones/'
cat_lag = pd.read_csv(geodir + 'A17_cc_geophones_catalog_quality.csv')

# Isolate only Grade AA and BB events, and high-quality cross correlations
cat_lag_HQ = cat_lag.loc[cat_lag.quality == 'HQ-HQ'].reset_index()
cat_lag_HQ = cat_lag_HQ.loc[(cat_lag_HQ.grade_new == 'AA') | (cat_lag_HQ.grade_new == 'BB')]
evids_AABB = np.unique(cat_lag_HQ.evid.tolist())

# Directory to save results
output_directory = '/data/ytamama/Apollo17/catalogs/cc_azimuths/'

# Number of iterations for gradient descent
num_iter = 20

# Iterate through events and calculate azimuth from lags
num_cores = 15
Parallel(n_jobs=num_cores)(delayed(azimuth_wrapper_lag)(cat_lag, evid, num_iter, output_directory) for evid in evids_AABB)

