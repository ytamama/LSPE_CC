"""
Yuri Tamama

Code to make sure moonquakes are classified into only one family

"""
# Import packages
import pandas as pd
from obspy import read,UTCDateTime
from datetime import datetime, timedelta
import numpy as np
import os

# Import functions
import sys
fxndir = '../functions/'
sys.path.insert(0,fxndir)
from moon2data import *
from one_family_each import *

########################################################################

# Similarity threshold for moonquakes to group into a family
cc_thresh = 0.90

# Directory to save catalog of moonquakes with high enough correlation coefficient
selectdir = '/data/ytamama/Apollo17/catalogs/cc_ABCD/select_high_cc_' + str(round(cc_thresh,2)) + '/'

# Directory to save moonquake catalogs, after we handle those classified into multiple families
sepdir = '/data/ytamama/Apollo17/catalogs/cc_ABCD/select_high_cc_nodupes_' + str(round(cc_thresh,2)) + '/'
if not os.path.exists(sepdir):
    os.mkdir(sepdir)
    
# Ensure each moonquake is assigned to only one family
one_family_each(selectdir,sepdir)

