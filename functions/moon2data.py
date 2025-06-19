"""
Functions to retrieve data containing a lunar seismic event

"""
# Import statements
from datetime import datetime, timedelta
import numpy as np
from obspy import read,UTCDateTime
import os
import pandas as pd
import glob
from scipy.interpolate import CubicSpline

########################################################################

def moon2sac(evttime,geonum,befwin,aftwin,minfreq,maxfreq,parentdir='/data/ytamama/Apollo17/LSPE_data/sac_volts_ds/'):
    '''
    Function to obtain the SAC file containing a lunar seismic event, 
    recorded in a catalog. 
    
    INPUTS
    evttime : Datetime object, containing the time of the event's arrival
    geonum : ID number of the geophone (1, 2, 3, or 4), that recorded this event. 
    befwin : Number of seconds before the event's arrival at which we begin the 
             seismogram. 
    aftwin : Number of seconds after the event's arrival at which we end the 
             seismogram. 
    minfreq : Minimum frequency to which we filter our seismogram, using a 
              bandpass filter. Set this value to '' if we don't wish to filter 
              our seismogram. 
    maxfreq : Maximum frequency to which we filter our seismogram, using a 
              bandpass filter. Set this value to '' if we don't wish to filter 
              our seismogram. 
    parentdir : Name of the parent directory containing the lunar data, inputted
                as a string
    
    OUTPUT
    evtst : Stream object, containing a seismogram of the lunar seismic event 
            during the specified time window. If the data needed for this 
            stream is not available, evtst is returned as an empty string. 
    
    '''
    
    # Construct name of SAC file of event's current hour 
    if evttime.month<10:
        evtmonstr='0'+str(evttime.month)
    else:
        evtmonstr=str(evttime.month)
    if evttime.day<10:
        evtdaystr='0'+str(evttime.day)
    else:
        evtdaystr=str(evttime.day) 
    evtymdstr=str(evttime.year)+evtmonstr+evtdaystr
    evtdir=parentdir + evtymdstr + '/'
    evtsac=evtymdstr + '_17Geo' + str(geonum) + '_' + str(evttime.hour) + '_ID'
    evtsac=evtdir+evtsac
    #
    if (not (os.path.exists(evtsac))) & (evttime.hour < 10):
        evtsac=evtymdstr + '_17Geo' + str(geonum) + '_0' + str(evttime.hour) + '_ID'
        evtsac=evtdir+evtsac
        
    
    # Construct names of SAC files for hour before and after, if necessary
    # Hour before
    beftime=evttime-timedelta(seconds=befwin)
    if beftime.hour!=evttime.hour:
        if beftime.month<10:
            befmonstr='0'+str(beftime.month)
        else:
            befmonstr=str(beftime.month)
        if beftime.day<10:
            befdaystr='0'+str(beftime.day)
        else:
            befdaystr=str(beftime.day)
        befymdstr=str(beftime.year)+befmonstr+befdaystr
        beffiledir=parentdir + befymdstr + '/' 
        sacbef=befymdstr + '_17Geo' + str(geonum) + '_' + str(beftime.hour) + '_ID'
        sacbef=beffiledir+sacbef
        #
        if (not (os.path.exists(sacbef))) & (beftime.hour < 10):
            sacbef=befymdstr + '_17Geo' + str(geonum) + '_0' + str(beftime.hour) + '_ID'
            sacbef=beffiledir+sacbef

    # Hour after
    afttime=evttime+timedelta(seconds=aftwin)
    if afttime.hour!=evttime.hour:
        if afttime.month<10:
            aftmonstr='0'+str(afttime.month)
        else:
            aftmonstr=str(afttime.month)
        if afttime.day<10:
            aftdaystr='0'+str(afttime.day)
        else:
            aftdaystr=str(afttime.day)
        aftymdstr=str(afttime.year)+aftmonstr+aftdaystr
        aftfiledir=parentdir + aftymdstr + '/' 
        sacaft=aftymdstr + '_17Geo' + str(geonum) + '_' + str(afttime.hour) + '_ID'
        sacaft=aftfiledir+sacaft
        #
        if (not (os.path.exists(sacaft))) & (afttime.hour < 10):
            sacaft=aftymdstr + '_17Geo' + str(geonum) + '_0' + str(afttime.hour) + '_ID'
            sacaft=aftfiledir+sacaft


    # Check if the SAC files we need exist
    # Just 1 SAC file
    if (beftime.hour==evttime.hour) & (afttime.hour==evttime.hour):
        # If file exists, read
        if os.path.exists(evtsac):
            evtst=read(evtsac)
                
        # Otherwise, return an empty string
        else:
            evtst=''
    
    # Files for this hour and hour before
    elif (beftime.hour!=evttime.hour) & (afttime.hour==evttime.hour):
        # If files exist, read
        if (os.path.exists(evtsac)) & (os.path.exists(sacbef)):
            evtst=read(evtsac)
            evtst+=read(sacbef)
            evtst.sort(['starttime'])
            # Merge
            evtst.merge(method=0,fill_value='interpolate')

        # Otherwise, skip this event
        else:
            evtst=''
    
    # Files for this hour and hour after
    elif (beftime.hour==evttime.hour) & (afttime.hour!=evttime.hour):
        # If files exist, read
        if (os.path.exists(evtsac)) & (os.path.exists(sacaft)):
            evtst=read(evtsac)
            evtst+=read(sacaft)
            evtst.sort(['starttime'])
            # Merge
            evtst.merge(method=0,fill_value='interpolate')
       
        # Otherwise, skip this event
        else:
            evtst=''
    
    # Files for this hour, hour before, and hour after
    elif (beftime.hour!=evttime.hour) & (afttime.hour!=evttime.hour):
        # If files exist, read
        if (os.path.exists(evtsac)) & (os.path.exists(sacbef)) & (os.path.exists(sacaft)):
            evtst=read(evtsac)
            evtst+=read(sacbef)
            evtst+=read(sacaft)
            evtst.sort(['starttime'])
            # Merge
            evtst.merge(method=0,fill_value='interpolate')
                
        # Otherwise, skip this event
        else:
            evtst=''


    # If we do NOT have the data to construct this seismogram, exit the function:
    if evtst=='':
        return evtst
    
    # Otherwise:
    else:
        # Filter and detrend the data if requested:
        if (minfreq!='') | (maxfreq!=''):
            
            # De-mean the signal
            evtst.traces[0].detrend('demean')
            
            # Remove linear trend
            evtst.traces[0].detrend('linear')
            
            # Butterworth-bandpass
            if (minfreq!='') & (maxfreq!=''):
                evtst.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
                
            # Butterworth-lowpass (filter out above maxfreq)
            elif (maxfreq!=''):
                evtst.filter('lowpass',freq=maxfreq)
            
            # Butterworth-highpass (filter out below minfreq)
            elif (minfreq!=''):
                evtst.filter('highpass',freq=minfreq)

            # Trim the SAC files to the window containing the event
            beftime_utc=UTCDateTime(beftime)
            afttime_utc=UTCDateTime(afttime)
            evtst.trim(starttime=beftime_utc,endtime=afttime_utc)
            
            # Remove mean and linear trend
            evtst.traces[0].detrend('demean')
            evtst.traces[0].detrend('linear')
        
        
        # Otherwise, return the raw seismogram
        else:
            # Trim the SAC files to the window containing the event
            beftime_utc=UTCDateTime(beftime)
            afttime_utc=UTCDateTime(afttime)
            evtst.trim(starttime=beftime_utc,endtime=afttime_utc) 
        
    # Return stream
    return evtst


def moon2ascii(evttime,geonum,befwin,aftwin,parentdir='/data/ytamama/Apollo17/LSPE_data/hourly_ascii_raw/'):
    '''
    Function to obtain read an ASCII file of raw lunar data, recorded by 
    Apollo 17, and retrieving a section containing a lunar seismic event. 
    
    INPUTS
    evttime : Datetime object, containing the time of the event's arrival
    geonum : ID number of the geophone (1, 2, 3, or 4), that recorded this event. 
    befwin : Number of seconds before the event's arrival at which we begin the 
             seismogram. 
    aftwin : Number of seconds after the event's arrival at which we end the 
             seismogram. 
    parentdir : Name of the parent directory containing the lunar data, inputted
                as a string
    
    OUTPUT
    evttimes : Time, in seconds, of the data relative to the inputted arrival time
    evtdata : Lunar seismic data surrounding the selected event's arrival time
    '''

    # Construct name of the ASCII file containing the data for the hour of 
    # the moonquake
    if evttime.month<10:
        evtmonstr = '0'+str(evttime.month)
    else:
        evtmonstr = str(evttime.month)
    if evttime.day<10:
        evtdaystr = '0'+str(evttime.day)
    else:
        evtdaystr = str(evttime.day) 
    evtymdstr = str(evttime.year) + evtmonstr + evtdaystr
    # Directory
    evtdir = parentdir + evtymdstr + '/'
    # Name of file
    evthr = str(evttime.hour)
    evtfile = evtdir + evtymdstr + '_HR' + str(evthr) + '_geo' + str(geonum) + '.csv'

    # Construct names of data files for the hour before and after if necessary
    # Hour before
    beftime = evttime - timedelta(seconds = befwin)
    if beftime.hour != evttime.hour:
        if beftime.month<10:
            befmonstr = '0'+str(beftime.month)
        else:
            befmonstr = str(beftime.month)
        if beftime.day<10:
            befdaystr = '0'+str(beftime.day)
        else:
            befdaystr = str(beftime.day)
        befymdstr = str(beftime.year) + befmonstr + befdaystr
        # Directory
        befdir = parentdir + befymdstr + '/'
        # Name of file
        befhr = str(beftime.hour)
        beffile = befdir + befymdstr + '_HR' + str(befhr) + '_geo' + str(geonum) + '.csv'

    # Hour after
    afttime = evttime + timedelta(seconds=aftwin)
    if afttime.hour != evttime.hour:
        if afttime.month<10:
            aftmonstr = '0'+str(afttime.month)
        else:
            aftmonstr = str(afttime.month)
        if afttime.day<10:
            aftdaystr = '0'+str(afttime.day)
        else:
            aftdaystr = str(afttime.day)
        aftymdstr = str(afttime.year) + aftmonstr + aftdaystr
        # Directory
        aftdir = parentdir + aftymdstr + '/'
        # Name of file
        afthr = str(afttime.hour)
        aftfile = aftdir + aftymdstr + '_HR' + str(afthr) + '_geo' + str(geonum) + '.csv'


    # Check if the data files we need exist
    # Just 1 data file
    if (beftime.hour==evttime.hour) & (afttime.hour==evttime.hour):
        # If file exists, read
        if os.path.exists(evtfile):
            evtdf = pd.read_csv(evtfile)
                
        # Otherwise, return an empty string
        else:
            evtdf = ''
    
    # Files for current hour and hour before
    elif (beftime.hour!=evttime.hour) & (afttime.hour==evttime.hour):
        # If files exist, read
        if (os.path.exists(evtfile)) & (os.path.exists(beffile)):
            evtdf = pd.read_csv(evtfile)
            befdf = pd.read_csv(beffile)
            # Combine dataframes
            evtdf = pd.concat([befdf,evtdf])
            
        # Still read if the file for the current hour exists
        elif (os.path.exists(evtfile)):
            evtdf = pd.read_csv(evtfile)

        # Otherwise, skip this event
        else:
            evtdf=''
    
    # Files for this hour and hour after
    elif (beftime.hour==evttime.hour) & (afttime.hour!=evttime.hour):
        # If files exist, read
        if (os.path.exists(evtfile)) & (os.path.exists(aftfile)):
            evtdf = pd.read_csv(evtfile)
            aftdf = pd.read_csv(aftfile)
            # Combine dataframes
            evtdf = pd.concat([evtdf,aftdf])
            
        # Still read if the file for the current hour exists
        elif (os.path.exists(evtfile)):
            evtdf = pd.read_csv(evtfile)
       
        # Otherwise, skip this event
        else:
            evtdf=''
    
    # Files for this hour, hour before, and hour after
    elif (beftime.hour!=evttime.hour) & (afttime.hour!=evttime.hour):
        # If files exist, read
        if (os.path.exists(evtfile)) & (os.path.exists(beffile)) & (os.path.exists(aftfile)):
            evtdf = pd.read_csv(evtfile)
            befdf = pd.read_csv(beffile)
            aftdf = pd.read_csv(aftfile)
            # Combine dataframes
            evtdf = pd.concat([befdf,evtdf,aftdf])
            
        # Still read if files for current hour and previous hour exist:
        elif (os.path.exists(evtfile)) & (os.path.exists(beffile)):
            evtdf = pd.read_csv(evtfile)
            befdf = pd.read_csv(beffile)
            # Combine dataframes
            evtdf = pd.concat([befdf,evtdf])
            
        # Still read if files for current hour and next hour exist:
        elif (os.path.exists(evtfile)) & (os.path.exists(aftfile)):
            evtdf = pd.read_csv(evtfile)
            aftdf = pd.read_csv(aftfile)
            # Combine dataframes
            evtdf = pd.concat([evtdf,aftdf])
            
        # Still read if the file for the current hour exists
        elif (os.path.exists(evtfile)):
            evtdf = pd.read_csv(evtfile)
                
        # Otherwise, skip this event
        else:
            evtdf=''


    # If we do NOT have the data to construct this seismogram, exit the function:
    if len(evtdf) == 0:
        evttimes = []
        evtdata = []
    
    # Otherwise:
    else:
        # Trim the data
        beftimestr = datetime.strftime(beftime,format = '%Y-%m-%d %H:%M:%S.%f')
        afttimestr = datetime.strftime(afttime,format = '%Y-%m-%d %H:%M:%S.%f')
        evtdf_trim = evtdf.loc[(evtdf.time >= beftimestr) & (evtdf.time <= afttimestr)]
        
        # Times
        evttimes_str = evtdf_trim.time.values
        evttimes_dt = [datetime.strptime(j,'%Y-%m-%d %H:%M:%S.%f') for j in evttimes_str]
        evttimes = [(j - evttime).total_seconds() for j in evttimes_dt]
        
        # Data
        evtdata = evtdf_trim.data_volts.values 

    # Return times and data
    return evttimes, evtdata


def lunar_spline(input_times, input_data, deltat):
    '''
    Function to resample and interpolate lunar seismic data to a constant 
    sampling rate, using cubic spline interpolation. 
    
    INPUTS
    input_times : Array of seconds corresponding to the inputted data
    input_data : Array of lunar seismic data corresponding to input_times
    deltat : Time step with which to resample the data
    
    OUTPUT
    output_times : Array of interpolated times, at the new sampling rate
    output_data : Array of interpolated data, at the new sampling rate
    '''
    # Construct a cubic spline
    lunar_spl = CubicSpline(input_times, input_data)
    
    # Construct array of interpolated times
    output_times = np.arange(min(input_times), max(input_times), deltat)
    
    # Interpolate data
    output_data = lunar_spl(output_times)
    
    # Return 
    return output_times, output_data


