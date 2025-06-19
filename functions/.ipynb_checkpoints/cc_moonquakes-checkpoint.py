"""
Codes to cross-correlate moonquakes and classify them into families 

"""
# Import packages
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy import signal
import obspy
import obspy.signal
import os
import glob

# Import functions
from moon2data import *

########################################################################

def cc_traces(signal1,arrtime1,signal2,arrtime2,maxshift=20,fs=117.6):
    '''
    Function to cross-correlate two signals, each with a time corresponding to 
    a seismic pick, and determine the time lag such that the similarity between 
    the two traces is maximized. 
    
    INPUTS
    signal1 : Time-series data of one signal
    arrtime1 : Datetime object representing the pick for signal1
    signal2 : Time-series data of the other signal
    arrtime2 : Datetime object representing the pick for signal2
    maxshift : Maximum number of seconds we allow for the lag on either side
    fs : Sampling frequency, in Hz
    
    OUTPUTS
    dt : Time lag, in number of seconds, of signal 2 relative to signal 1
    cc_max : Maximum correlation coefficient between the signals, normalized 
            with respect to amplitude
    arrtime2_shift : Arrival time of the pick for signal 2, shifted by the 
                     lag determined from cross-correlation. 
    
    '''  
    # Cross-correlate signals
    corr_fxn = signal.correlate(signal1,signal2)
    
    # Normalize cross-correlation function  
    norm = (np.sum(signal1 ** 2) * np.sum(signal2 ** 2)) ** 0.5
    # If the normalization factor is zero:
    if norm == 0:
        corr_fxn_norm = corr_fxn
        corr_fxn_norm[:] = 0
    else:
        corr_fxn_norm = corr_fxn / norm
        
    # Obtain and limit the lag
    lags = signal.correlation_lags(signal1.size, signal2.size)
    dts = lags*(1/fs)
    dts_lim = dts[(dts >= -1*maxshift) & (dts <= maxshift)]
    corr_fxn_norm_lim = corr_fxn_norm[(dts >= -1*maxshift) & (dts <= maxshift)]
    dt = dts_lim[(np.argmax(corr_fxn_norm_lim))]
    
    # Obtain max. normalized correlation coefficient
    cc_max = np.max(corr_fxn_norm_lim)
    
    # Shift pick time of second signal
    arrtime2_shift = arrtime2 - timedelta(seconds = dt)
    
    # Return outputs
    return dt, cc_max, arrtime2_shift
    
    

def cc_traces_nolag(signal1,signal2):
    '''
    Function to cross-correlate two signals with no lag between them
    
    INPUTS
    signal1 : Time-series data of one signal
    signal2 : Time-series data of the other signal
    
    OUTPUTS
    cc_nolag : Correlation coefficient between the signals with zero lag, 
               normalized with respect to amplitude
    
    ''' 
    # Cross-correlate signals
    corr_fxn = signal.correlate(signal1,signal2)
    
    # Normalize cross-correlation function  
    norm = (np.sum(signal1 ** 2) * np.sum(signal2 ** 2)) ** 0.5
    # If the normalization factor is zero:
    if norm == 0:
        corr_fxn_norm = corr_fxn
        corr_fxn_norm[:] = 0
    else:
        corr_fxn_norm = corr_fxn / norm
        
    # Determine lag allowing maximum similarity
    lags = signal.correlation_lags(signal1.size, signal2.size)
    
    # Obtain correlation coefficient at zero lag
    cc_nolag = corr_fxn_norm[lags == 0][0]
    return cc_nolag
    
    
    
def select_highCC(alignfile,cc_thresh,savedir):
    '''
    Function to identify detections whose correlation coefficients exceed the 
    threshold value
    
    INPUTS
    alignfile : Full path to the file containing the correlation coefficients of all 
                moonquakes with respect to the reference, across all four 
                geophones
    cc_thresh : Correlation coefficient threshold, at or above which we 
                consider the moonquake to be in the same family as the 
                reference
    savedir : Directory to save the dataframe containing the events that pass 
              the threshold
    
    '''
    # Read alignment catalog
    align_df = pd.read_csv(alignfile)
    align_df = align_df[['evid','evid_ref','geophone','mod_arrival_time','corr_coeffs','dt_values','minfreq','maxfreq','grade','grade_new']]
    
    # Obtain events where the threshold is met at at least one geophone
    align_df_pass = align_df.loc[align_df.corr_coeffs >= cc_thresh]
    evids_pass = np.unique(align_df_pass.evid.tolist())

    # Moonquakes with high enough CCs
    align_df_pass = align_df[align_df['evid'].isin(evids_pass)]
    align_df_pass = align_df_pass.reset_index()
    align_df_pass = align_df_pass[align_df_pass.columns.drop(list(align_df_pass.filter(regex='Unnamed:|index')))] 
    
    # Save dataframe
    alignfile_basename = os.path.basename(alignfile)
    align_df_pass.to_csv(savedir + alignfile_basename)
    
    

def cc_A17_wrapper(mooncat_full, evid_ref, befwin, aftwin, minfreq, maxfreq, parentdir, output_directory, demean=1, maxshift=20, env=1):
    '''
    Wrapper function to cross-correlate one moonquake, recorded by the Apollo 17  
    geophones, with the remaining moonquakes in Francesco's catalog, recorded on the 
    same instrument. Conduct the cross-correlations across all Grade A, B,
    C, and D moonquakes -- substituting missing detections with the mean arrival 
    time across the existing detections. 
    
    Return the correlation coefficients between each pair of moonquakes to 
    quantify the similarity between the selected event and the remaining 
    moonquakes. 
    
    INPUTS
    mooncat_full : Dataframe of high-quality records of Grade A through D moonquakes
    evid_ref : String containing the ID of the selected moonquake 
    befwin : Number of seconds before the arrival time at which we begin each 
             seismogram. 
    aftwin : Number of seconds after the arrival time at which we end each 
             seismogram.  
    minfreq : Minimum frequency of our filtering window, in Hz
              If not imposing a lower bound, input ''
    maxfreq : Maximum frequency of our filtering window, in Hz
              If not imposing an upper bound, input ''
    parentdir : Directory where the data are stored. 
    output_directory : Directory to save the resulting alignment dataframe 
    demean : Do we want to remove the mean from the waveforms first? 
             0 for no
             1 for yes
    maxshift : Maximum number of seconds we allow for the lag on either side
    env : Do we want to cross-correlate the waveforms or seismogram envelope? 
          0 for waveforms
          1 for envelope
          
    OUTPUT
    align_df : Dataframe containing the results of cross-correlation, as the 
               following columns:
    
    evid : Event ID
    evid_ref : Event ID of the moonquake used as the reference, with which all other moonquakes 
               are cross-correlated against
    geophone : Geophone 
    mod_arrival_time : Arrival time, corrected with the lag maximizing correlation 
                       coefficient
    corr_coeff : Correlation coefficients, normalized by amplitude
    dt : Lag for each event relative to the "reference," in seconds. 
    minfreqs : Minimum frequency of filtering window
    maxfreqs : Maximum frequency of filtering window
    grade : Grade of moonquake (A, B, C, or D)
    
    '''
    # Initialize arrays for cross-correlation results
    evids = []
    evid_refs = []
    geophones = []
    mod_arrival_times = []
    corr_coeffs = []
    dt_vals = []
    minfreqs = []
    maxfreqs = []
    grades = []
    grades_new = []
    
    # Obtain rows containing "reference" moonquake and identify geophones
    rows = mooncat_full.loc[mooncat_full.evid == evid_ref]
    geonums = np.array(rows.geophone.tolist())
    
    # Iterate through geophones
    for geonum in geonums:
        
        # Obtain pick time of moonquake
        row = rows.loc[rows.geophone == geonum].iloc[0]
        arrtime_ref = datetime.strptime(row.picktime,'%Y-%m-%d %H:%M:%S.%f')
        refgrade = row.grade
        refgrade_new = row.grade_new
        
        # Obtain seismogram for selected "reference" moonquake 
        st_ref = moon2sac(arrtime_ref,geonum,befwin,aftwin,minfreq,maxfreq,parentdir)
        data_ref = st_ref.traces[0].data
        # Demean signal
        if demean == 1:
            data_ref = data_ref - np.mean(data_ref)
        
        # Cross-correlating waveform or envelope? 
        if env == 1:
            data_ref = obspy.signal.filter.envelope(data_ref)
            
        # Obtain section of HQ dataframe corresponding to this geophone
        mooncat_geoX = mooncat_full.loc[mooncat_full.geophone == geonum].reset_index()
    
        # Iterate through remaining moonquakes in the catalog
        for n in np.arange(0,len(mooncat_geoX)):
            
            # Arrival time of selected waveform, from Francesco's catalog
            moonrow = mooncat_geoX.iloc[n]
            evid_test = moonrow.evid
            
            # If autocorrelating -- don't bother. The correlation coefficient will be 1 anyways
            if evid_test == evid_ref:
                evids.append(evid_test)
                evid_refs.append(evid_ref)
                geophones.append(geonum)
                mod_arrival_times.append(arrtime_ref)
                corr_coeffs.append(1)
                dt_vals.append(0)
                if minfreq == '':
                    minfreqs.append(-1)
                else:
                    minfreqs.append(minfreq)
                if maxfreq == '':
                    maxfreqs.append(-1)
                else:
                    maxfreqs.append(maxfreq)
                grades.append(refgrade)
                grades_new.append(refgrade_new)
                
                
            # If cross-correlating with another event:
            else:
                # Obtain seismogram
                arrtime_test = datetime.strptime(moonrow.picktime,'%Y-%m-%d %H:%M:%S.%f')
                st_test = moon2sac(arrtime_test,geonum,befwin,aftwin,minfreq,maxfreq,parentdir)
                data_test = st_test.traces[0].data
                # Demean signal?
                if demean == 1:
                    data_test = data_test - np.mean(data_test)

                # Waveform or envelope?
                if env == 1:
                    data_test = obspy.signal.filter.envelope(data_test)

                # Cross-correlate reference and "test" seismogram to obtain normalized correlation coefficient and lag
                dt, cc_max, arrtime_test_shift = cc_traces(data_ref,arrtime_ref,data_test,arrtime_test,maxshift,117.6)
                
                # Save results
                evids.append(evid_test)
                evid_refs.append(evid_ref)
                geophones.append(geonum)
                mod_arrival_times.append(arrtime_test_shift)
                corr_coeffs.append(cc_max)
                dt_vals.append(dt)
                if minfreq == '':
                    minfreqs.append(-1)
                else:
                    minfreqs.append(minfreq)
                if maxfreq == '':
                    maxfreqs.append(-1)
                else:
                    maxfreqs.append(maxfreq)
                grades.append(moonrow.grade)
                grades_new.append(moonrow.grade_new)
    
    
    # Compile results into table
    align_df = pd.DataFrame(data = {'evid': evids, 
                                    'evid_ref':evid_refs,
                                    'geophone': geophones, 
                                    'mod_arrival_time': mod_arrival_times, 
                                    'corr_coeffs': corr_coeffs, 
                                    'dt_values': dt_vals, 
                                    'minfreq': minfreqs, 
                                    'maxfreq': maxfreqs,
                                    'grade': grades,
                                    'grade_new': grades_new})

    # Save dataframe
    align_df.to_csv(output_directory + f'AlignArrivals_CC_REF-{evid_ref}.csv',index=False)

    
    
def cc_A17_geophones_wrapper(mooncat_full, evid, befwin, aftwin, minfreq, maxfreq, parentdir, output_directory, demean=1, maxshift=20, env=1):
    '''
    Wrapper function to cross-correlate the seismogram of one moonquake, recorded by 
    an Apollo 17 geophone, with the records of the same moonquake but on another geophone to 
    determine the similarity and lag between them. Conduct the cross-correlations across 
    all Grade A, B, C, and D moonquakes, substituting missing detections with the mean 
    arrival time across the existing detections. 
    
    Return the correlation coefficients between each pair of geophones to 
    quantify the similarity of records across geophones. 
    
    Note that correlation coefficients are normalized for amplitude. 
    
    INPUTS
    mooncat_full : Combined dataframe of catalogued Grade A through D moonquakes.
    evid : String containing the ID of the selected moonquake 
    befwin : Number of seconds before the arrival time at which we begin the seismogram
    aftwin : Number of seconds after the arrival time at which we end the seismogram
    minfreq : Minimum frequency of our filtering window, in Hz
              If not imposing a lower bound, input ''
    maxfreq : Maximum frequency of our filtering window, in Hz
              If not imposing an upper bound, input ''
    parentdir : Directory where the data are stored
    output_directory : Directory to save resulting dataframe
    demean : Do we want to remove the mean from the waveforms first? 
             0 for no
             1 for yes
    maxshift : Maximum number of seconds we allow for the lag on either side
    env : Do we want to cross-correlate the waveforms or seismogram envelope? 
          0 for waveforms
          1 for envelope
                 
    OUTPUT
    align_df : Dataframe containing the results of cross-correlation, as the 
               following columns:
    
               evid : Event ID
               cc_12 : Normalized correlation coefficient between geophones 1 and 2
               dt_12 : Lag between geophones 1 and 2
               cc_13 : Normalized correlation coefficient between geophones 1 and 3
               dt_13 : Lag between geophones 1 and 3
               cc_14 : Normalized correlation coefficient between geophones 1 and 4
               dt_14 : Lag between geophones 1 and 4
               cc_23 : Normalized correlation coefficient between geophones 2 and 3
               dt_23 : Lag between geophones 2 and 3
               cc_34 : Normalized correlation coefficient between geophones 3 and 4
               dt_34 : Lag between geophones 3 and 4
               cc_24 : Normalized correlation coefficient between geophones 2 and 4
               dt_24 : Lag between geophones 2 and 4
               minfreqs : Minimum frequency of filtering window
               maxfreqs : Maximum frequency of filtering window
               grades : Grade of moonquake 
               grades_new : New grade of moonquake
    
    '''
    # Combinations of geophones
    geonum_combos =  np.array([[1,2], [1,3], [1,4], [2,3], [3,4], [2,4]])
    
    # Initialize outputs
    evids = []
    ccs_12 = []
    dts_12 = []
    ccs_13 = []
    dts_13 = []
    ccs_14 = []
    dts_14 = []
    ccs_23 = []
    dts_23 = []
    ccs_34 = []
    dts_34 = []
    ccs_24 = []
    dts_24 = []
    minfreqs = []
    maxfreqs = []
    grades = []
    grades_new = []
    
    # Obtain one of the pick times for event
    evtrow = mooncat_full.loc[mooncat_full.evid == evid].iloc[0]
    arrtime = datetime.strptime(evtrow.picktime,'%Y-%m-%d %H:%M:%S.%f')
    
    # Iterate through each combination of geophones
    for combo in geonum_combos:

        # Obtain record on first geophone
        geonumA = combo[0]
        stA = moon2sac(arrtime,geonumA,befwin,aftwin,minfreq,maxfreq,parentdir)
        dataA = stA.traces[0].data
        # Demean signal?
        if demean == 1:
            dataA = dataA - np.mean(dataA)
        
        # Waveform or envelope?
        if env == 1:
            dataA = obspy.signal.filter.envelope(dataA)
        
        # Obtain record on second geophone
        geonumB = combo[1]
        stB = moon2sac(arrtime,geonumB,befwin,aftwin,minfreq,maxfreq,parentdir)
        dataB = stB.traces[0].data
        # Demean signal?
        if demean == 1:
            dataB = dataB - np.mean(dataB)
        
        # Waveform or envelope?
        if env == 1:
            dataB = obspy.signal.filter.envelope(dataB)
            
        # Cross-correlate and find the lag
        dt, cc, arrtime_shift = cc_traces(dataA,arrtime,dataB,arrtime,maxshift,117.6)
        
        # Save results
        if (geonumA == 1) & (geonumB == 2):
            ccs_12.append(cc)
            dts_12.append(dt)
        elif (geonumA == 1) & (geonumB == 3):
            ccs_13.append(cc)
            dts_13.append(dt)
        elif (geonumA == 1) & (geonumB == 4):
            ccs_14.append(cc)
            dts_14.append(dt)
        elif (geonumA == 2) & (geonumB == 3):
            ccs_23.append(cc)
            dts_23.append(dt)
        elif (geonumA == 3) & (geonumB == 4):
            ccs_34.append(cc)
            dts_34.append(dt)
        elif (geonumA == 2) & (geonumB == 4):
            ccs_24.append(cc)
            dts_24.append(dt)

    # 
    evids.append(evid)
    minfreqs.append(minfreq)
    maxfreqs.append(maxfreq)
    grades.append(evtrow.grade)
    grades_new.append(evtrow.grade_new)
            
    # Construct and save final dataframe
    align_df = pd.DataFrame(data = {'evid': evids, 
                                    'cc_12': ccs_12, 
                                    'dt_12': dts_12,
                                    'cc_13': ccs_13, 
                                    'dt_13': dts_13,
                                    'cc_14': ccs_14, 
                                    'dt_14': dts_14,
                                    'cc_23': ccs_23, 
                                    'dt_23': dts_23,
                                    'cc_34': ccs_34,
                                    'dt_34': dts_34,
                                    'cc_24': ccs_24,
                                    'dt_24': dts_24,
                                    'minfreq': minfreqs, 
                                    'maxfreq': maxfreqs,
                                    'grade': grades,
                                    'grade_new': grades_new})
    
    # Save dataframe
    align_df.to_csv(output_directory + evid + '_cc_geophones.csv',index=False)
    return align_df

