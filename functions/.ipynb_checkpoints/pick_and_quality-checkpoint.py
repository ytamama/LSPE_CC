"""
Functions to estimate the arrival and rise time of moonquakes, and get a sense of 
the signal quality

"""
# Imports
import pandas as pd
from obspy import read,UTCDateTime
from datetime import datetime, timedelta
import numpy as np
import os
import glob
import sys
import obspy
import obspy.signal
from scipy import optimize

# Import functions
from moon2data import *

########################################################################

def snr_window(data,times,winlen,stepby):
    ''' 
    Function to calculate the SNR across two moving windows
    
    INPUTS
    data : Array containing data
    times : Array containing time steps corresponding to each point in data
    winlen : Length of each window, in seconds
    stepby : Number of seconds to step by to reach the next window
    
    OUTPUTS
    snr_times : Time step corresponding to the boundary between each pair of windows
    snr_vals : SNR of each pair of windows
    '''
    starttime1 = np.min(times)
    starttime2 = starttime1 + winlen
    snr_times = []
    snr_vals = []
    
    # Iterate through windows
    while starttime2 < np.max(times)-winlen:
        
        # Obtain "noise" and "signal" windows
        win1 = data[(times >= starttime1) & (times < starttime1 + winlen)]
        win2 = data[(times >= starttime2) & (times < starttime2 + winlen)]
        
        # Calculate SNR between windows
        snr = np.mean(win2) / np.mean(win1)
        
        # Append results 
        snr_vals.append(snr)
        snr_times.append(starttime2)
        
        # Iterate to next window
        starttime1 += stepby
        starttime2 += stepby
        
    # Return outputs
    snr_times = np.array(snr_times)
    snr_vals = np.array(snr_vals)
    return snr_times, snr_vals


# Gaussian function
def gaussian(x, amp, mean, sd, height):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sd ** 2)) + height


# Fit a Gaussian
def fit_gaussian(data,times):
    try:
        # Normalize data
        data_norm = data - np.min(data)
        data_norm = data_norm / np.max(data_norm)
        
        # Set bounds and initial conditions
        fxn_bounds = ((0, -1*np.inf, -1*np.inf, -1*np.inf), (np.inf, np.inf, np.inf, np.inf))
        fxn_p0 = [np.max(data_norm), times[np.argmax(data_norm)], 1, np.percentile(data_norm,10)]
        
        # Fit a Gaussian to normalized data
        popt, _ = optimize.curve_fit(gaussian, times, data_norm, bounds=fxn_bounds, p0 = fxn_p0)
        data_gauss_norm = gaussian(times, *popt)
        # Note: 
        # popt[0]: Amplitude, popt[1]: Mean, popt[2]: Standard deviation, popt[3]: Height
        
        # Calculate RMSE between normalized curves
        residuals = data_gauss_norm - data_norm
        residuals_sq = np.square(residuals)
        mse = np.mean(residuals_sq)
        rmse = np.sqrt(mse)
        
    except:
        rmse = -1
        data_gauss_norm = []
        data_norm = []
        popt = []
        
    # Return
    return rmse, data_gauss_norm, data_norm, popt


# Wrapper function 
def pick_wrapper(evid,mqtbl,befwin,aftwin,minfreq,maxfreq,parentdir,winlen,stepby,savedir):
    '''
    Wrapper function to estimate the pick, rise time, and quality of 
    a moonquake
    
    INPUTS
    evid : Event ID
    mqtbl : Dataframe containing the average arrival time of each moonquake
    befwin : # of seconds before avg. arrival to start the seismogram
    aftwin : # of seconds after avg. arrival to end the seismogram
    minfreq : Min. frequency of bandpass
    maxfreq : Max. frequency of bandpass
    parentdir : Directory to obtain seismic data
    savedir : Directory to save results
    winlen : Length of each window used to calculate the signal to noise ratio (SNR)
    stepby : Amount of seconds we increment each window
    
    OUTPUT
    CSV file, containing a dataframe with the following columns:
    
    evid : Event ID
    grade : Grade of event
    geophone : Geophone
    picktime : Approx. time of first arrival
    risetime : Time when the smoothed signal envelope reaches its maximum
    emergence : Difference, in seconds, between risetime and picktime
    rmse : Root mean squared error between the moving window SNR function 
           and a fitted Gaussian. Both are normalized with respect to 
           max. amplitude. 
    '''
    # Initialize outputs
    evids = []
    geophones = []
    grades = []
    geophones = []
    picktimes = []
    risetimes = []
    emergences = []
    rmses = []
    amps = []
    means = []
    sds = []
    heights = []
    
    # Obtain rough arrival time
    evtrow = mqtbl.loc[mqtbl.evid == evid].iloc[0]
    arrtime = datetime.strptime(evtrow.mean_arrival_time,'%Y-%m-%dT%H:%M:%S.%f')
    arrtime_str = datetime.strftime(arrtime,'%Y-%m-%d %H:%M:%S.%f')
    evtgrade = evtrow.grade
    
    # Iterate through geophones
    geophones = [1, 2, 3, 4]
    for geonum in geophones:

        # Obtain waveforms and calculate envelope
        st = moon2sac(arrtime,geonum,befwin,aftwin,minfreq,maxfreq,parentdir)
        times = st.traces[0].times() - befwin
        data = st.traces[0].data
        data_env = obspy.signal.filter.envelope(data)
        # Subtract minimum
        data_env = data_env - np.min(data_env)
    
        #######################
        # Determine signal quality
    
        # Calculate SNR function
        snrtimes, snrvals = snr_window(data_env,times,winlen,stepby)

        # Pick where we reach peak SNR
        pick_s = snrtimes[np.argmax(snrvals)]
        picktime = arrtime + timedelta(seconds = pick_s)
        picktime_str = datetime.strftime(picktime,'%Y-%m-%d %H:%M:%S.%f')
   
        # Rise time is when envelope reaches its maximum, after the pick time
        times_signal = times[times >= pick_s]
        data_env_signal = data_env[times >= pick_s]
        rise_s = times_signal[np.argmax(data_env_signal)]
        risetime = arrtime + timedelta(seconds = rise_s)
        risetime_str = datetime.strftime(risetime,'%Y-%m-%d %H:%M:%S.%f')
    
        # Emergence: difference between rise time and onset time
        emergence = (risetime - picktime).total_seconds()
    
        # Fit Gaussian to SNR function and obtain parameters of Gaussian 
        rmse, snrvals_gauss_norm, snrvals_norm, params = fit_gaussian(snrvals,snrtimes)
        if len(params) > 0:
            amp = params[0]
            mean = params[1]
            sd = np.abs(params[2])
            height = params[3]
        else:
            amp = -1
            mean = -1
            sd = -1
            height = -1

        # Save results
        evids.append(evid)
        grades.append(evtgrade)
        picktimes.append(picktime_str)
        risetimes.append(risetime_str)
        emergences.append(emergence)
        rmses.append(rmse)
        amps.append(amp)
        means.append(mean)
        sds.append(sd)
        heights.append(height)

        
    #######################
    # Save results to dataframe
    d = {'evid':evids, 'grade':grades, 'geophone':geophones, 'picktime':picktimes, 
         'risetime':risetimes, 'emergence_s':emergences, 'rmse':rmses, 'amplitude':amps, 
         'mean_s':means, 'sd_s':sds, 'height':heights}
    df = pd.DataFrame(data = d)
    df.to_csv(f'{savedir}{evid}_picks.csv',index=False)

