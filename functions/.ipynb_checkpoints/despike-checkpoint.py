"""
Functions to despike Apollo 17 Lunar Seismic Profiling Experiment data 

Adapted from original code by Francesco Civilini (see https://github.com/civilinifr/thermal_mq_analysis)

"""
# Import packages
import pandas as pd
import numpy as np
import glob
from scipy.signal import find_peaks
import os

######################################################################################################

# Francesco's code to despike signals
def despike(input_d, med_multiplier=5.):
    """
    Francesco's despiking routine. See https://github.com/civilinifr/thermal_mq_analysis
    
    Despikes the input data according to Renee's 2005 paper (Median despiker).
    Note: This routine does NOT do a bandpass.

    :param input_d: [Vector] Input data to despike
    :param med_multiplier: [Scalar] The median multiplier threshold (excludes greater values)
    :return: [Vector] Despiked data
    """

    # Compute a running median on the data
    input_d_copy = np.copy(input_d)
    window_size = 589
    if window_size % 2 == 0:
        window_size = window_size + 1
    med = running_median(input_d_copy, window_size)

    # Find values greater than 5 times the running median
    indices_to_remove = []
    for ind in np.arange(len(input_d_copy)):
        if input_d_copy[ind] > abs(med[ind] * med_multiplier) or input_d_copy[ind] < -1 * abs(med[ind] * med_multiplier):
            indices_to_remove.append(ind)

    # Change data values for those indices to zero
    input_d_copy[indices_to_remove] = 0

    return input_d_copy


def running_median(seq, win):
    """
    From P01_ascii2sac.py by Francesco Civilini
    see https://github.com/civilinifr/thermal_mq_analysis
    
    Computes a running median on the input data
    :param seq: [Vector] Input data to find the running median
    :param win: [Integer] Window size in samples
    :return: [Vector] The running median
    """
    medians = []
    window_middle = int(np.ceil(win/2))

    for ind in np.arange(len(seq)):

        if ind <= window_middle:
            medians.append(np.median(abs(seq[0:win])))

        if ind >= len(seq)-window_middle:
            medians.append(np.median(abs(seq[len(seq)-win:len(seq)])))

        if window_middle < ind < len(seq)-window_middle:
            medians.append(np.median(abs(seq[ind-int(np.floor(win/2)):ind+int(np.floor(win/2))])))

    return np.array(medians)


def median_abs_peak_amp(input_data):
    '''
    Function to calculate the median absolute peak amplitude of a sequence of data
    
    INPUT
    input_data : Sequence of input data
    
    OUTPUT
    med_abs_amp : Median absolute peak amplitude of the input data
    
    '''
    input_abs = np.abs(input_data)
    peakinds,_ = find_peaks(input_abs)
    peakvals = input_abs[peakinds]
    med_abs_amp = np.median(peakvals)
    return med_abs_amp


def despike_YT(input_data, med_multiplier=4., window_size = 589):
    '''
    Function to despike a signal by removing signals with an amplitude of 5 times that of the median peak amplitude of each window
    
    Modification of despike() by Francesco Civilini in P01_ascii2sac.py
    See https://github.com/civilinifr/thermal_mq_analysis
    
    '''
    # Copy data
    input_data_copy = np.copy(input_data)
    
    # Adjust window size if necessary
    if window_size % 2 == 0:
        window_size = window_size + 1
        
    # Iterate through windows
    startind = 0
    endind = window_size
    despiked_data = []
    keepLooping = True
    while keepLooping == True:
        
        # Check if we're at the end of the data
        if endind >= len(input_data_copy):
            endind = len(input_data_copy)
            # This is the last iteration
            keepLooping = False
        
        # Isolate window of data
        data_window = input_data_copy[startind:endind]
        data_window_abs = np.abs(data_window)
        
        # Calculate median peak amplitude
        med_amp = median_abs_peak_amp(data_window_abs)
        
        # Set signals exceeding the median peak amplitude by the multiplier to 0
        data_window_ds = data_window
        data_window_ds[data_window_abs >= med_multiplier*med_amp] = 0
        despiked_data = np.concatenate([despiked_data,data_window_ds])
        
        # Iterate to next indices
        startind += window_size
        endind += window_size

    # Return
    return despiked_data

