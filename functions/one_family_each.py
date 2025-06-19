# Import packages
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import glob

########################################################################

def one_family_each(ccdir,savedir):
    '''
    Function to sort the cross-correlation catalogs such that each moonquake 
    is classified into only one family:
        
    INPUTS
    ccdir : Directory containing the cross-correlation catalogs, whose
            contents have already been narrowed down to moonquakes with 
            correlation coefficients exceeding a certain threshold.       
    savedir : Directory in which we want to save our newly sorted 
              moonquake family catalogs
              
    OUTPUT
    cake : A dummy variable, which will yield a value of 'good' if the code
           runs        

    '''
    # Assemble list of moonquake catalogs
    mq_files = glob.glob(f'{ccdir}Align*.csv')
    
    # Sort files by how many moonquakes are included in each catalog AND by grade of reference event
    mq_file_list = []
    num_mqs_list = []
    refgrades_list = []
    for mq_file in mq_files:
        
        # Load file
        mq_df = pd.read_csv(mq_file)
        
        # Number of moonquakes
        num_mqs = len(np.unique(mq_df.evid.tolist()))
        
        # Append to lists
        if num_mqs > 0:
            mq_file_list.append(mq_file)
            num_mqs_list.append(num_mqs)
            
            # Grade of reference
            refrows = mq_df.loc[mq_df.evid == mq_df.evid_ref]
            refgrade = refrows.grade_new.tolist()[0]
            refgrades_list.append(refgrade)
    
    # Sort files by grade AND number of moonquakes in each catalog
    d = {'mq_files': mq_file_list, 'ref_grade_new':refgrades_list, 'num_mqs': num_mqs_list}
    mq_files_df = pd.DataFrame(data = d)
    mq_files_df = mq_files_df.sort_values(by=['ref_grade_new', 'num_mqs'], ascending=[True, False])
    mq_file_list = mq_files_df.mq_files.tolist()

    # Initialize list of event IDs that have already been classified 
    evids_classified = []
    
    # Iterate through catalogs
    for mq_file in mq_file_list:
        
        # Read catalog
        mq_df = pd.read_csv(mq_file) 
        mq_df = mq_df[mq_df.columns.drop(list(mq_df.filter(regex='Unnamed:|index')))]
        
        # If no moonquakes have been classified yet:
        if len(evids_classified) == 0:
            
            # Add event IDs to classified events list
            evids_classified = np.unique(mq_df.evid.tolist())
        
            # Save dataframe! 
            # File name
            basefile = os.path.basename(mq_file)
            mq_file_new = savedir + basefile[0:-4] + '_separated.csv'
            # Save! 
            mq_df.to_csv(mq_file_new,index=False)
           
        
        # Check if we already classified moonquakes
        else:
            
            # If reference moonquake has not yet been classified into another 
            # family:
            evid_ref = mq_df.evid_ref.tolist()[0]
            if evid_ref not in evids_classified:
                
                # Remove moonquakes that have already been classified elsewhere
                mq_df_new = mq_df.loc[~mq_df['evid'].isin(evids_classified)]
                mq_df_new = mq_df_new[mq_df_new.columns.drop(list(mq_df_new.filter(regex='Unnamed:|index')))]
                mq_df_new = mq_df_new.reset_index()
                mq_df_new = mq_df_new[mq_df_new.columns.drop(list(mq_df_new.filter(regex='Unnamed:|index')))]
    
                # Save dataframe!
                if len(mq_df_new) > 0:
                    
                    # File name
                    basefile = os.path.basename(mq_file)
                    mq_file_new = savedir + basefile[0:-4] + '_separated.csv'
                    # Save! 
                    mq_df_new.to_csv(mq_file_new,index=False)
                    
                    # Add event IDs to classified events list
                    evids_classified = np.concatenate((evids_classified, np.unique(mq_df_new.evid.tolist())))

            
    cake = 'good'
    return cake



