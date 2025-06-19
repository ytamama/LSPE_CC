"""
Code to conduct Schuster tests of periodicity. Originally written in MATLAB 
by Thomas Ader but translated into Python by Yuri Tamama.

Original code: http://www.tectonics.caltech.edu/resources/schuster_spectrum/

Ader, T.J. & Avouac, J.-P., 2013. Detecting periodicities and declustering in earthquake catalogs using the Schuster spectrum, application to Himalayan seismicity. Earth and Planetary Science Letters, 377–378, 97–105. doi:10.1016/j.epsl.2013.06.032

"""
# Import packages
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import glob

########################################################################

def Schuster_test_log(t_seq,periods):
    '''
    INPUTS:
    t_seq: The timeseries on which we want to test any periodicity
    periods: Vector containing all the periods to be tested
    
    OUTPUT: 
    log_prob: Vector of the same length as 'period' giving for each period the
              log of probability that the distance covered by the Schuster walk is due to a
              random walk. The smaller the probability, the more likely it is that
              there is a periodicity in the data at the given period.

    Original code in MATLAB by Thomas Ader
    Copyright 2010-2011 Tectonics Observatory
    Created 06/10/2012
    Modified 05/01/2012
    
    Translated for use in Python by Yuri Tamama
    '''
    log_prob = np.zeros(len(periods))
    num_periods = len(periods)
    t_seq = t_seq - np.min(t_seq)
    
    # Iterate through periods
    for r in np.arange(0,num_periods):
        
        # Periods
        T = periods[r]
        # Round # of cycles
        tlim = np.max(t_seq) - np.mod(np.max(t_seq),T)  
        # Select a round # of cycles from the timeseries
        t = t_seq[t_seq <= tlim]
        
        # Phase all times from timeseries
        phase = np.mod(t,T)*2*np.pi/T
    
        # Where Schuster walk ends
        end_walk = [np.sum(np.cos(phase)),np.sum(np.sin(phase))]
        
        # Distance from the origin
        D = np.linalg.norm(end_walk)
        
        # Probability to reach the same point via random walk
        log_prob[r] = -D**2/len(t)

    # Return
    return log_prob