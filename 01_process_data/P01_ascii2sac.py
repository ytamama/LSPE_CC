"""
Yuri Tamama

Takes input LSPE daily ASCII data and converts it to hourly SAC files. 
The data are in uncompressed volts

Adapted from original code by Francesco Civilini (see https://github.com/civilinifr/thermal_mq_analysis) 

"""

# Import packages
import pandas as pd
import numpy as np
import glob
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper
import os
from datetime import datetime, timedelta
from obspy.io.sac.sactrace import SACTrace
import time
from joblib import Parallel, delayed

# Import functions 
import sys
sys.path.insert(0,'../functions/')
from ascii2sac import *

###############################################################################################################

# Set up directories
# Directories containing ASCII data (modify to where you store the data!)
data_dir = '/data/ytamama/Apollo17/LSPE_data/'
input_folder = '/data/ytamama/Apollo17/LSPE_data/daily_ascii_raw/'

# Main
# Number of CPU cores to use in the processing
num_cores = 15

# Time between each sample in resampled trace
target_delta = 1/117.6

# Despike the data? Yes
ds = 1

# Find a list of files
output_directory = f'{data_dir}sac_volts_ds/'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

input_files = []
dayfiles = glob.glob(f'{input_folder}*.txt')
for dayname in dayfiles:
    input_files.append(dayname)

# Create the folders before parallelizing
for input_data in input_files:
    day_actual = os.path.basename(input_data).split('_')[0]
    if not os.path.exists(f'{output_directory}{day_actual}'):
        os.mkdir(f'{output_directory}{day_actual}')

if num_cores == 1:
    # Non-parallel version of the code
    for input_data in input_files:
        process_ascii(input_data, output_directory, target_delta, ds)
else:
    # Parallel version of the code
    Parallel(n_jobs=num_cores)(delayed(process_ascii)(input_data, output_directory, target_delta, ds) for input_data in input_files)


