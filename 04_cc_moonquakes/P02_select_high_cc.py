"""
Yuri Tamama

Code to select moonquakes above a specified correlation coefficient, for each 
cross-correlation catalog

"""
# Import packages
import pandas as pd
from obspy import read,UTCDateTime
from datetime import datetime, timedelta
import numpy as np
import os
import glob
import sys
from joblib import Parallel, delayed

# Import functions
fxndir = '../functions/'
sys.path.insert(0,fxndir)
from moon2data import *
from cc_moonquakes import *

########################################################################

# Similarity threshold for moonquakes to group into a family
cc_thresh = 0.90

# Directory containing catalogs containing the results of our cross-correlations
aligndir = '/data/ytamama/Apollo17/catalogs/cc_ABCD/alignment_cc/'
# Directory to save catalog of moonquakes with high enough correlation coefficient
selectdir = '/data/ytamama/Apollo17/catalogs/cc_ABCD/select_high_cc_' + str(round(cc_thresh,2)) + '/'
if not os.path.exists(selectdir):
    os.mkdir(selectdir)
    
# Iterate through alignment catalogs
alignfiles = glob.glob(f'{aligndir}Align*.csv')

# Parallelize code
num_cores = 20
Parallel(n_jobs=num_cores)(delayed(select_highCC)(alignfile, cc_thresh, selectdir) for alignfile in alignfiles)

