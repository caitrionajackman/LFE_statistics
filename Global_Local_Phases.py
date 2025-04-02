import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap 
import matplotlib.dates as mdates
from mpl_toolkits import axes_grid1
from bisect import bisect_left
import configparser
from tqdm import tqdm
from scipy.io import readsav
from matplotlib.patches import Circle, PathPatch
import os
from LFE_statistics import *

# Planetary Period Oscillations Function - need updated .sav file
def SavePPO(file_path, LFE_df, data_directory, file_name):
    print("Finding LFE Phase")

    print(f"Loading {file_path}")
    ppo_df = readsav(file_path)


    south_time = ppo_df["south_model_time"] # minutes since 2004-01-01 00:00:00
    south_phase = ppo_df["south_mag_phase"]

    north_time = ppo_df["north_model_time"]
    north_phase = ppo_df["north_mag_phase"]

    doy2004_0 = pd.Timestamp(2004, 1, 1)

    lfe_south_phase_indices = []
    lfe_north_phase_indices = []
    for i, lfe in tqdm(LFE_df.iterrows(), total=len(LFE_df)):

        lfe_start_time = lfe["start"] # pandas timestamp
        lfe_start_doy2004 = (pd.Timestamp(lfe_start_time) - doy2004_0).total_seconds() / 60 / 60 / 24 # days since 2004-01-01 00:00:00

        # Find minimum time difference
        south_index = (np.abs(south_time - lfe_start_doy2004)).argmin()
        lfe_south_phase_indices.append(south_index)

        north_index = (np.abs(north_time - lfe_start_doy2004)).argmin()
        lfe_north_phase_indices.append(north_index)


    print(len(lfe_south_phase_indices))
    LFE_df["south phase"] = np.array(south_phase)[lfe_south_phase_indices]
    LFE_df["north phase"] = np.array(north_phase)[lfe_north_phase_indices]

    print(f"Saving new csv file to {data_directory+file_name}")
    LFE_df.to_csv(data_directory + file_name)

# Defined paths - adjust according to where EPHEMERIS data from SPICE is hosted
config = configparser.ConfigParser()
config.read('config_LFE_stats.ini')   
data_directory = config['filepaths']['LFE_data_directory'] # Directory where SN_ms_tot_V2.0.csv, SN_d_tot_V2.0.csv, and LFEs_joined.csv are located

original_unet = "lfe_detections_unet_2874.csv"
lfe_joined_list = "LFEs_joined.csv"
one_min_resol_mission = "20040101000000_20170915115700_ephemeris.csv" # whole mission (2004/01/01 00:00:00 - 2017/09/15 11:56:00) coverage
one_min_resol_og = "2004001_2017258_ephemeris.csv" # original LFE list coverage
one_min_resol_joined = "LFEs_joined_ephemeris.csv" # joined LFE list coverage
ppo_file = "mag_phases_2004_2017_final.sav" # Phases Calibration Data

# Define joined polygon UNET
trajectories_df = pd.read_csv(data_directory + one_min_resol_mission)
join_unet = pd.read_csv(data_directory + lfe_joined_list, index_col = 0) 
LFE_join = pd.read_csv(data_directory + one_min_resol_joined)

join_unet['Range'] = LFE_join['R_KSM']
join_unet['subLST'] = LFE_join['subLST']
join_unet['subLat'] = LFE_join['subLat']
join_unet['x_ksm'] = LFE_join['x_KSM']
join_unet['y_ksm'] = LFE_join['y_KSM']
join_unet['z_ksm'] = LFE_join['z_KSM']
# join_unet.to_csv("LFEs_joined_times_range_lst_lat.csv") 

# Generate joined dataframe with global phases
if os.path.exists(data_directory + "Joined_LFEs_w_phases.csv") == True:
    pass
else:
    SavePPO(data_directory + ppo_file, join_unet, data_directory, "Joined_LFEs_w_phases.csv")

lfe_phases = pd.read_csv(data_directory + "Joined_LFEs_w_phases.csv", index_col = 0)

# Save new index LIST w/out south phase nans - missing calibration comparisons in original .sav file
nans = np.where(np.isnan(lfe_phases['south phase']))[0]
all = np.arange(0, np.shape(lfe_phases)[0], 1)

no_nans = []
for element in all:
    if element not in nans:
        no_nans.append(element)
# JOINED LFE df w/ non-nan global phases
lfe_phase_no_nan = lfe_phases.iloc[no_nans].reset_index()

# Calculate & PLOT Global Phases for each Joined LFE Event
def global_phase_figures(lfe_phase_no_nan, dusk_dawn_plots = False):
    if dusk_dawn_plots == True:
        # Create two subsets of the data - 1) LFEs seen from Dawn (02 - 10 hr LST), and 2) LFEs seen from Dusk (14 - 22 hr LST)
        LFE_df = lfe_phase_no_nan
        dawn_lfe, = np.where((LFE_df['subLST'] >= 2) & (LFE_df['subLST'] <= 10 ))
        dusk_lfe, = np.where((LFE_df['subLST'] >= 14) & (LFE_df['subLST'] <= 22))

        dawn_phase = lfe_phase_no_nan.iloc[dawn_lfe].reset_index()
        dusk_phase = lfe_phase_no_nan.iloc[dusk_lfe].reset_index()

        # Generate GLOBAL Degree measurements 
        north = np.array(dusk_phase["north phase"]) % 360
        south = np.array(dusk_phase["south phase"]) % 360

        dusk_phase['north_deg'] = north
        dusk_phase['south_deg'] = south

        # Plot Dawn & Dusk Contours
        # DUSK - CONTOURS
        dusk_phase['vals'] = np.ones(np.shape(dusk_phase)[0])
        dusk_phase['north'] = np.ones(np.shape(dusk_phase)[0])
        dusk_phase['south'] = np.ones(np.shape(dusk_phase)[0])

        # Assign grid values -- For both Colorful & Black/White Filled Contours
        x1 = np.arange(0, 360, 30) 
        y1 =np.arange(0, 360, 30) 

        for i in range(np.shape(dusk_phase)[0]):
            dusk_phase['north'].iloc[i] = x1[np.where(dusk_phase['north_deg'].iloc[i] - x1 >= 0)[0][-1]]
            dusk_phase['south'].iloc[i] = x1[np.where(dusk_phase['south_deg'].iloc[i] - x1 >= 0)[0][-1]]

        Z = pd.pivot_table(dusk_phase, index = 'north', columns = 'south', values = 'vals', aggfunc = sum, fill_value = 0)

        # Periodic phase wrap at edges of contour plot implemented
        x1_plot = np.arange(-360, 720, 30)
        y1_plot = np.arange(-360, 720, 30)
        X, Y = np.meshgrid(x1_plot, y1_plot)

        # 2d histogram - both contours are [0,15, ..., 345]
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel("North Phase ($^\circ$)", fontsize = 20)
        ax.set_xlabel("South Phase ($^\circ$)", fontsize = 20)
        ax.set_title(f"Dusk (14 - 22 hr LST) LFE Phase Distribution \n  N = {np.shape(dusk_phase)[0]} - Peak at 150-180˚ S and 330-360˚ N", fontsize = 20)

        Z_rev = np.matrix(Z)[::1]
        Z_rev_plot = np.tile(Z_rev, [3,3])

        plot1 = ax.contour(X + 15, Y + 15, Z_rev_plot, cmap = 'spring_r')
        plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[0, 360, 0, 360], origin = 'lower') 

        ax.set_xticks(np.arange(0, 360, 30))
        ax.set_xticklabels(np.arange(0, 360, 30), rotation = 90)
        ax.set_yticks(np.arange(0, 360, 30))

        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)

        cbar = fig.colorbar(plot2, ax = ax, shrink = 0.74)
        cbar.add_lines(plot1)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('# of LFE occurrences', rotation=270)
        cbar.ax.tick_params(labelsize = 12)

        fig.tight_layout()
        #plt.savefig("Contours_Dusk.jpeg", dpi = 300)
        plt.show()

        # DAWN - CONTOURS
        dawn_phase['vals'] = np.ones(np.shape(dawn_phase)[0])
        dawn_phase['north'] = np.ones(np.shape(dawn_phase)[0])
        dawn_phase['south'] = np.ones(np.shape(dawn_phase)[0])
        
        x1 = np.arange(0, 360, 30) 
        y1 =np.arange(0, 360, 30) 

        for i in range(np.shape(dawn_phase)[0]):
            dawn_phase['north'].iloc[i] = x1[np.where(dawn_phase['north_deg'].iloc[i] - x1 >= 0)[0][-1]]
            dawn_phase['south'].iloc[i] = x1[np.where(dawn_phase['south_deg'].iloc[i] - x1 >= 0)[0][-1]]

        Z1 = pd.pivot_table(dawn_phase, index = 'north', columns = 'south', values = 'vals', aggfunc = sum, fill_value = 0)
    
        # 2d histogram - both contours are [0,15, ..., 345]
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel("North Phase ($^\circ$)", fontsize = 20)
        ax.set_xlabel("South Phase ($^\circ$)", fontsize = 20)
        ax.set_title(f"Dawn (02 - 10 hr LST) LFE Phase Distribution \n N = {np.shape(dawn_phase)[0]} - Peak at 120-150˚ S and 300-330˚ N", fontsize = 20)

        Z_rev = np.matrix(Z1)[::1]
        Z_rev_plot = np.tile(Z_rev, [3,3])

        plot1 = ax.contour(X+15, Y+15, Z_rev_plot, levels = 15, cmap = 'spring_r')
        plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[0, 360, 0, 360], origin = 'lower') 

        ax.set_xticks(np.arange(0, 360, 30))
        ax.set_xticklabels(np.arange(0, 360, 30), rotation = 90)
        ax.set_yticks(np.arange(0, 360, 30))

        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)

        cbar = fig.colorbar(plot2, ax = ax, shrink = 0.74)
        cbar.add_lines(plot1)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('# of LFE occurrences', rotation=270)
        cbar.ax.tick_params(labelsize = 12)

        fig.tight_layout()
        #plt.savefig("Countours_Dawn.jpeg", dpi = 300)
        plt.show()

    else: # Global phases across Dusk & Dawn
        north = np.array(lfe_phase_no_nan["north phase"]) % 360
        south = np.array(lfe_phase_no_nan["south phase"]) % 360

        lfe_phase_no_nan['north_deg'] = north
        lfe_phase_no_nan['south_deg'] = south

        joint_phases_wout_nan = lfe_phase_no_nan # Variable name change

        joint_phases_wout_nan['vals'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        joint_phases_wout_nan['north'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        joint_phases_wout_nan['south'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        
        x1 = np.arange(0, 360, 30) # Black/White Contours
        y1 =np.arange(0, 360, 30) 

        for i in range(np.shape(joint_phases_wout_nan)[0]):
            joint_phases_wout_nan['north'].iloc[i] = x1[np.where(joint_phases_wout_nan['north_deg'].iloc[i] - x1 >= 0)[0][-1]]
            joint_phases_wout_nan['south'].iloc[i] = x1[np.where(joint_phases_wout_nan['south_deg'].iloc[i] - x1 >= 0)[0][-1]]

        Z = pd.pivot_table(joint_phases_wout_nan, index = 'north', columns = 'south', values = 'vals', aggfunc = sum)
        
        # Periodic phase wrap at edges of contour plot implemented
        x1_plot = np.arange(-360, 720, 30)
        y1_plot = np.arange(-360, 720, 30)
        X, Y = np.meshgrid(x1_plot, y1_plot)

        # 2d histogram - both contours are [0,15, ..., 345]
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel("North Phase ($^\circ$)", fontsize = 20)
        ax.set_xlabel("South Phase ($^\circ$)", fontsize = 20)
        ax.set_title(f"Total LFE Phase Distribution \n N = {np.shape(joint_phases_wout_nan)[0]}", fontsize = 20)

        # Contour wrap at edges taken into account
        Z_rev = np.matrix(Z)[::1]
        Z_rev_plot = np.tile(Z_rev, [3,3])

        plot1 = ax.contour(X+15, Y+15, Z_rev_plot, levels = 10, cmap = 'spring_r')
        plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[0, 360, 0, 360], origin = 'lower') 

        ax.set_xticks(np.arange(0, 360, 30))
        ax.set_xticklabels(np.arange(0, 360, 30), rotation = 90)
        ax.set_yticks(np.arange(0, 360, 30))

        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)

        cbar = fig.colorbar(plot2, ax = ax, shrink = 0.74)
        cbar.add_lines(plot1)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('# of LFE occurrences', rotation=270)
        cbar.ax.tick_params(labelsize = 12)

        fig.tight_layout()
        plt.show() # ----- EX. /Sample_Visualizations/Contours_Global_Joined_LFEs.jpeg



def local_phase_figures(lfe_phase_no_nan, trajectories_df):
    # Calculate Global Phases (degrees) for ALL JOINT LFEs
    north = np.array(lfe_phase_no_nan["north phase"]) % 360
    south = np.array(lfe_phase_no_nan["south phase"]) % 360

    lfe_phase_no_nan['north_deg'] = north
    lfe_phase_no_nan['south_deg'] = south

    # We are using Matt's LST calculations for local time 
    x = lfe_phase_no_nan["x_ksm"]
    y = lfe_phase_no_nan["y_ksm"]
    z = lfe_phase_no_nan["z_ksm"]

    # Local Time as defined by SPICE
    spacecraft_lt = np.array(trajectories_df['subLST'])

    azimuth = []
    for lt in spacecraft_lt:
        azimuth.append(((lt-12) * 15 + 720) % 360) # what is the +720 for? 

    local_phase_north = []
    local_phase_south = []
    for north_phase, south_phase, az in zip(north, south, azimuth):
        local_phase_north.append(((north_phase - az) + 720) % 360)
        local_phase_south.append(((south_phase - az) + 720) % 360)

    local_phase_north = np.array(local_phase_north)
    local_phase_south = np.array(local_phase_south)

    # Append local phases to dataframe
    lfe_phase_no_nan['local north phase'] = local_phase_north
    lfe_phase_no_nan['local south phase'] = local_phase_south

    # Create 2-d Histogram of South Local Phases vs North Local Phases for all JOINED LFEs
    lfe_phase_no_nan['vals'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    lfe_phase_no_nan['north'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    lfe_phase_no_nan['south'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    
    x1 = np.arange(0, 360, 30) # Black/White Filled Contours
    y1 =np.arange(0, 360, 30) #360

    for i in range(np.shape(lfe_phase_no_nan)[0]):
        lfe_phase_no_nan['north'].iloc[i] = x1[np.where(lfe_phase_no_nan['local north phase'].iloc[i] - x1 >= 0)[0][-1]]
        lfe_phase_no_nan['south'].iloc[i] = x1[np.where(lfe_phase_no_nan['local south phase'].iloc[i] - x1 >= 0)[0][-1]]

    Z2 = pd.pivot_table(lfe_phase_no_nan, index = 'north', columns = 'south', values = 'vals', aggfunc = sum, fill_value = 0)

    # Periodic phase wrap at edges of contour plot implemented
    x1_plot = np.arange(-360, 720, 30)
    y1_plot = np.arange(-360, 720, 30)
    X, Y = np.meshgrid(x1_plot, y1_plot)
    
    # 2d histogram - both contours are [0,15, ..., 345]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylabel("North Local Phase ($^\circ$)", fontsize = 20)
    ax.set_xlabel("South Local Phase ($^\circ$)", fontsize = 20)
    ax.set_title(f"SPICE-based LFE Phase Distribution \n N = {np.shape(lfe_phase_no_nan)[0]}", fontsize = 20)

    # Contour wrap at edges taken into account
    Z_rev = np.matrix(Z2)[::1]
    Z_rev_plot = np.tile(Z_rev, [3,3])

    plot1 = ax.contour(X+15, Y+15, Z_rev_plot, levels = 10, cmap = 'spring_r')
    plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[0, 360, 0, 360], origin = 'lower')

    ax.set_xticks(np.arange(0, 360, 30))
    ax.set_xticklabels(np.arange(0, 360, 30), rotation = 90)
    ax.set_yticks(np.arange(0, 360, 30))

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)

    cbar = fig.colorbar(plot2, ax = ax, shrink = 0.74)
    cbar.add_lines(plot1)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# of LFE occurrences', rotation=270)
    cbar.ax.tick_params(labelsize = 12)

    fig.tight_layout()
    #plt.savefig("Countours_Local_Phases.jpeg", dpi = 300)
    plt.show()

# Actual Code 
global_phase_figures(lfe_phase_no_nan, dusk_dawn_plots=False) # ----- EX. /Sample_Visualizations/Contours_Global_Joined_LFEs.jpeg
local_phase_figures(lfe_phase_no_nan, trajectories_df=trajectories_df) # ----- EX. /Sample_Visualizations/Contours_Local_Joined_LFEs.jpeg
