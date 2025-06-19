## Welcome

Thank you for your interest in my Python repository (LSPE_CC) accompanying my paper Tamama et al., 2025, submitted to JGR: Planets. Please find brief explanations of the contents of this repository. Please also note that the directories containing the codes are listed in the numerical order in which they should be run. 

### LSPE_CC/

- #### catalogs/

    Catalogs pertaining to the moonquakes recorded by the Apollo 17 Lunar Seismic Profiling Experiment (LSPE)
    - **azimuths/** - Results of stochastic gradient descent algorithm to calculate azimuth
    - **cc_geophones/** - Results of cross-correlating records between geophones for each event
    - **cc_moonquakes_sample/** - Results of cross-correlating the records of one event with all others in the moonquake catalog, for a few example events. 
    - **coordinates/** - Coordinates of the geophones, Lunar Module (LM), and boulders at the Apollo 17 site, estimated from Haase et al. (2019), and source-receiver distances between the geophones and possible seismic sources
    - **FC_catalogs/** - Catalogs compiled by and/or derived from Civilini et al. (2021) and Civilini et al. (2023)
    - **final_catalogs/** - High-quality seismic events
        - **geo3_geo4_events/** - Geophone 3 and Geophone 4 events
        - **Isolated_vs_repeating/** - Isolated and repeating events
        - **LM_vs_boulder_vs_other/** - LM-quakes, boulder-quakes, and other thermal moonquakes 
    - **quality_control/** - Results of intermediate steps in applying quality control algorithms to moonquakes 
    - **temperature/** - Temperatures and timestamps within the day-to-day temperature cycle at the Apollo 17 site, from Molaro et al. (2017)
    
- #### functions/ 
    Functions to process and filter lunar seismic data, calculate incident azimuth, cross-correlate moonquakes, etc. 

- #### 01_process_data/
    Process the ASCII data into hourly SAC files and combine the catalogs of Civilini et al. (2021) and Civilini et al. (2023)

- #### 02_pick_and_quality/
    Evaluate the quality of moonquake seismograms and refine the arrival times of moonquakes on each geophone

- #### 03_cc_geophones/
    Cross-correlate records of the same moonquake but on different geophones

- #### 04_cc_moonquakes/
    Cross-correlate records of different moonquakes

- #### 05_calc_event_properties/
    Calculate properties of each moonquake such as peak ground velocity (PGV), emergence, incident azimuth, duration, and temperature / time-of-day of moonquakes

- #### 06_repeating_vs_isolated/
    Plot PGV, emergence, duration, temperature / time-of-day, and incident azimuth of isolated and repeating moonquakes

- #### 07_plot_families/
    Plot PGV, emergence, duration, temperature / time-of-day, incident azimuth, waveforms, and frequency spectra of moonquake families

- #### 08_boulder_quakes/ 
    Identify and plot boulder-quakes, as well as Geophone 3 and 4 events

- #### 09_LM_events/
    Identify and plot LM events

- #### 10_known_distances/
    Calculate distances between possible seismic sources and each geophone and plot possible relationships between amplitude and source-receiver distance of boulderquakes

- #### 11_other_thermal_events/
    Identify and plot waveforms and characteristics of other thermal moonquakes

- #### 12_other_figures/
    Plot additional figures in the manuscript
