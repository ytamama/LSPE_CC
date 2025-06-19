"""
Functions to calculate the azimuth of Apollo 17 thermal moonquakes using stochastic gradient descent 

Adapted from original code by Francesco Civilini (see https://github.com/civilinifr/thermal_mq_analysis)

"""

# Import packages
import pickle
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math
import datetime
import random

############################################################################################################

def tt_misfit(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the total travel time misfit across all geophone stations
    
    From P11_find_azimuth.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """
    # Initialize the vector
    sta_misfit = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):

        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_misfit[sta_ind] = (ts + (np.sqrt(((xs-xi)**2)+((ys-yi)**2))/v) - ti)**2
    misfit = np.sum(sta_misfit)

    return misfit


def comp_dt_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dt
    
    From P11_find_azimuth.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dt_grad = np.zeros(len(new_rel_time))
    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit'
        sta_dt_grad[sta_ind] = (2*(ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti))

    dt_grad = np.sum(sta_dt_grad)

    return dt_grad


def comp_dx_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dx
    
    From P11_find_azimuth.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dx_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_dx_grad[sta_ind] = (2*(xs-xi)*(ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti))/(v*(np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2))))

    dx_grad = np.sum(sta_dx_grad)

    return dx_grad


def comp_dy_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dy
    
    From P11_find_azimuth.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dy_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_dy_grad[sta_ind] = (2 * (ys - yi) * (ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti)) / (
                    v * (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2))))

    dy_grad = np.sum(sta_dy_grad)

    return dy_grad



def model(x_vector, y_vector, new_rel_time, avg_velocity, geo_locs, xs, ys):
    """
    Stochastic gradient descent, without creating figures for each iteration
    
    From P11_find_azimuth.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis
    
    :param x_vector: [vector] Distance x (from Geo3) parameter space in meters
    :param y_vector: [vector] Distance y (from Geo3) parameter space in meters
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :param avg_velocity: [float] Average expected wave velocity
    :param geo_locs : [vector] Geophone coordinates
    :param xs: [float] Source x location 
    :param ys: [float] Source y location 
    :return:
    """
    # Set the number of iterations and the learning rate
    # True number of iterations is one less than displayed
    num_iterations = 500001
    lr_t = 0.05
    lr_x = 200.0
    lr_y = lr_x

    # Set a misfit improvement cutoff value. If the mean improvement of the past number of iterations is below this, we are probably ok with stopping.
    iteration_num_cutoff = 10000
    iteration_value_cutoff = 0.1

    # We will want ot save the misfit, but the plot step of 100 is too large. 1000 is fine.

    # Initialize the parameters randomly
    ts = 0
    misfit_vector = []
    iteration_vector = []
    ts_vector = []
    xs_vector = []
    ys_vector = []
    theta_vector = []

    # We set the location of Geophone 3 to be the origin
    x3 = 0
    y3 = 0

    # Iterate through algorithm
    for iteration in np.arange(num_iterations):

        # Do a forward propagation of the traces
        misfit = tt_misfit(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)

        # Compute theta
        theta_deg = 90 - np.degrees(np.arctan2((ys - y3), (xs - x3)))
        if theta_deg < 0:
            theta_deg += 360

        if iteration > iteration_num_cutoff + 1:
            if abs(theta_vector[iteration - iteration_num_cutoff] - theta_deg) < iteration_value_cutoff:
                break

        # Append results of our calculations
        iteration_vector.append(iteration)
        misfit_vector.append(misfit)
        ts_vector.append(ts)
        xs_vector.append(xs)
        ys_vector.append(ys)
        theta_vector.append(theta_deg)

        # Compute the gradient using the analytical derivative
        # To make things clear, we will create a new function for each parameter
        dt = comp_dt_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)
        dx = comp_dx_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)
        dy = comp_dy_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)

        # Perturb source location using learning rate
        ts = ts - (lr_t * dt)
        xs = xs - (lr_x * dx)
        ys = ys - (lr_y * dy)

        
    # Create variables for the first source values
    xs_start = xs_vector[0]
    ys_start = ys_vector[0]
    ts_start = ts_vector[0]

    # Final source location
    xs1 = xs_vector[-1]
    ys1 = ys_vector[-1]

    # Calculate theta at final iteration
    final_theta_deg = 90 - np.degrees(np.arctan2((ys1 - y3), (xs1 - x3)))
    if final_theta_deg < 0:
        final_theta_deg += 360
    
    # Final misfit
    final_misfit = misfit_vector[-1]
    final_iteration = iteration_vector[-1]

    return final_theta_deg, xs_start, ys_start, ts_start, xs_vector[-1], ys_vector[-1], ts_vector[-1], final_misfit, final_iteration


    
def azimuth_wrapper_lag(cc_cat, evid, num_iter, output_directory):
    '''
    Wrapper code for estimating the incident azimuth of the inputted 
    moonquake, based on the lags obtained between geophones.
    
    Adapted from P11_find_azimuth.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis
    
    INPUTS
    cc_cat : Catalog of correlation coefficients between geophones
    output_directory : Directory to save azimuth results
    evid : Event ID
    num_iter : Number of iterations to conduct gradient descent
    
    '''
    # Assume average velocity for this area
    avg_velocity = 45.0

    # Set locations of each geophone relative to geophone 3, the center of the array (in meters)
    geo1_loc = np.array([45.578, 34.973])
    geo2_loc = np.array([-53.06, 19.045])
    geo3_loc = np.array([0, 0])
    geo4_loc = np.array([12.596, -55.485])
    geo_locs_all = [geo1_loc, geo2_loc, geo3_loc, geo4_loc]
    
    # Obtain HQ-HQ cross correlations for that event
    cc_cat_HQ = cc_cat.loc[cc_cat.quality == 'HQ-HQ'].reset_index()
    evtrows_HQ = cc_cat_HQ.loc[cc_cat_HQ.evid == evid].reset_index()
    evtrows_HQ = evtrows_HQ[['evid','pair','cc','dt','quality','grade_new']]
    
    # Which geophones are high-quality?
    geophones_HQ = []
    for r in np.arange(0,len(evtrows_HQ)):
        row = evtrows_HQ.iloc[r]
        pair = row.pair
        geophones_HQ.append(int(np.floor(pair/10)))   # First geophone
        geophones_HQ.append(int(row.pair % 10))  # Second geophone
    geophones_HQ = np.sort(np.unique(geophones_HQ))

    ### Obtain lags with respect to one high-quality geophone
    
    # Initialize output
    geo_locs = []
    rel_times = []
    geophones = []
    
    # Select reference geophone
    geonum_ref = min(geophones_HQ)

    # Iterate through high-quality geophones
    for geonum in geophones_HQ:
        
        # Geophone number and location
        geophones.append(geonum)
        geo_locs.append(geo_locs_all[geonum - 1])

        # Lag time
        # Autocorrelation
        if geonum == geonum_ref:
            # Set reference time as zero
            rel_times.append(0)

        # Cross-correlation with different geophone 
        else:
            # Which pair? (Note: geophones in ascending order)
            pair = int(f'{min([geonum_ref, geonum])}{max([geonum_ref, geonum])}')
            row = evtrows_HQ.loc[evtrows_HQ.pair == pair].iloc[0]
            # Reference compared to current geophone
            if geonum_ref < geonum:
                rel_times.append(-1*row['dt'])
            # Current geophone compared to reference
            elif geonum_ref > geonum:
                rel_times.append(row['dt'])

    # Adjust reference time so that earliest is zero
    rel_times = rel_times - np.min(rel_times)
    
    # Set up parameter space for gradient descent
    x_vector = np.arange(-2000, 2001)
    y_vector = np.arange(-2000, 2001)

    # Initialize outputs
    evids = []
    theta_vector = []
    xs_start_vector, ys_start_vector, ts_start_vector = [], [], []
    xs_fin_vector, ys_fin_vector, ts_fin_vector = [], [], []
    misfit_fin_vector = []
    final_iteration_vector = []

    # Random starting locations
    np.random.seed(seed=1)
    xs_vector = np.random.choice(x_vector, num_iter)
    ys_vector = np.random.choice(y_vector, num_iter)

    # Conduct gradient descent
    for start_iter in np.arange(num_iter):

        # Calculate incident azimuth
        theta, xs_start, ys_start, ts_start, xs_fin, ys_fin, ts_fin, misfit_fin, final_iteration = \
             model(x_vector, y_vector, rel_times, avg_velocity, geo_locs, xs_vector[start_iter], ys_vector[start_iter])
        evids.append(evid)
        theta_vector.append(theta)
        xs_start_vector.append(xs_start)
        ys_start_vector.append(ys_start)
        ts_start_vector.append(ts_start)
        xs_fin_vector.append(xs_fin)
        ys_fin_vector.append(ys_fin)
        ts_fin_vector.append(ts_fin)
        misfit_fin_vector.append(misfit_fin)
        final_iteration_vector.append(final_iteration)
    
    # Combine into a pandas dataframe and save
    combined_data = list(zip(evids, theta_vector, xs_start_vector, ys_start_vector, ts_start_vector,
                         xs_fin_vector, ys_fin_vector, ts_fin_vector, misfit_fin_vector, final_iteration_vector))
    df = pd.DataFrame(combined_data, columns=['evid','theta', 'xs_start', 'ys_start', 'ts_start',
                                          'xs_fin', 'ys_fin', 'ts_fin', 'misfit_fin', 'final_iteration'])
    
    # Save dataframe
    df.to_csv(output_directory + evid + '_azimuth_results_lag.csv',index=False)
    
    
    
def azimuth_wrapper_picks(pick_cat, evid, num_iter, output_directory):
    '''
    Wrapper code for estimating the incident azimuth of the inputted 
    moonquake, based on the arrival times picked using the 
    signal-to-noise ratio (SNR) function
    
    Adapted from P11_find_azimuth.py by Francesco Civilini
    See https://github.com/civilinifr/thermal_mq_analysis
    
    INPUTS
    pick_cat : Catalog containing high-quality, re-calculated arrival times. 
               Each moonquake should have at least 3 high-quality detections
    output_directory : Directory to save azimuth results
    evid : Event ID
    num_iter : Number of iterations to conduct gradient descent
    
    '''
    # Set average velocity for this area
    avg_velocity = 45.0

    # Set locations of each geophone relative to geophone 3, the center of the array (in meters)
    geo1_loc = np.array([45.578, 34.973])
    geo2_loc = np.array([-53.06, 19.045])
    geo3_loc = np.array([0, 0])
    geo4_loc = np.array([12.596, -55.485])
    geo_locs_all = [geo1_loc, geo2_loc, geo3_loc, geo4_loc]

    # Compile arrival times at each geophone
    evtrows = pick_cat.loc[pick_cat.evid == evid].reset_index()
    geo_locs = []
    pick_times = []
    geophones = []
    for r in np.arange(0,len(evtrows)):
        row = evtrows.iloc[r]
        picktime = datetime.datetime.strptime(row.picktime_SNR,'%Y-%m-%d %H:%M:%S.%f')
        pick_times.append(picktime)
        geophone = row.geophone
        geophones.append(row.geophone)
        geo_locs.append(geo_locs_all[row.geophone-1])

    # Calculate relative arrival times between geophones
    # Set earliest time to zero
    earliest_arrival = np.min(pick_times)
    rel_times = []
    for picktime in pick_times:
        time_diff = (picktime - earliest_arrival).total_seconds()
        rel_times.append(time_diff)
    
    # Set up parameter space for gradient descent
    x_vector = np.arange(-2000, 2001)
    y_vector = np.arange(-2000, 2001)

    # Initialize outputs
    evids = []
    theta_vector = []
    xs_start_vector, ys_start_vector, ts_start_vector = [], [], []
    xs_fin_vector, ys_fin_vector, ts_fin_vector = [], [], []
    misfit_fin_vector = []
    final_iteration_vector = []

    # Random starting locations
    np.random.seed(seed=1)
    xs_vector = np.random.choice(x_vector, num_iter)
    ys_vector = np.random.choice(y_vector, num_iter)

    # Conduct gradient descent
    for start_iter in np.arange(num_iter):

        # Calculate incident azimuth
        theta, xs_start, ys_start, ts_start, xs_fin, ys_fin, ts_fin, misfit_fin, final_iteration = \
             model(x_vector, y_vector, rel_times, avg_velocity, geo_locs, xs_vector[start_iter], ys_vector[start_iter])
        evids.append(evid)
        theta_vector.append(theta)
        xs_start_vector.append(xs_start)
        ys_start_vector.append(ys_start)
        ts_start_vector.append(ts_start)
        xs_fin_vector.append(xs_fin)
        ys_fin_vector.append(ys_fin)
        ts_fin_vector.append(ts_fin)
        misfit_fin_vector.append(misfit_fin)
        final_iteration_vector.append(final_iteration)
    
    # Combine into a pandas dataframe and save
    combined_data = list(zip(evids, theta_vector, xs_start_vector, ys_start_vector, ts_start_vector,
                         xs_fin_vector, ys_fin_vector, ts_fin_vector, misfit_fin_vector, final_iteration_vector))
    df = pd.DataFrame(combined_data, columns=['evid','theta', 'xs_start', 'ys_start', 'ts_start',
                                          'xs_fin', 'ys_fin', 'ts_fin', 'misfit_fin', 'final_iteration'])
    
    # Save dataframe
    df.to_csv(output_directory + evid + '_azimuth_results_SNR.csv',index=False)
    

