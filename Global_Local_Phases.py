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
        north1 = np.array(dusk_phase["north phase"]) % 360
        south1 = np.array(dusk_phase["south phase"]) % 360
        north2 = np.array(dawn_phase["north phase"]) % 360
        south2 = np.array(dawn_phase["south phase"]) % 360

        dusk_phase['north_deg'] = north1
        dusk_phase['south_deg'] = south1
        dawn_phase['north_deg'] = north2
        dawn_phase['south_deg'] = south2

        # Plot Dawn & Dusk Contours
        # DUSK - CONTOURS
        dusk_phase['vals1'] = np.ones(np.shape(dusk_phase)[0])
        dusk_phase['north1'] = np.ones(np.shape(dusk_phase)[0])
        dusk_phase['south1'] = np.ones(np.shape(dusk_phase)[0])
        dusk_phase['vals2'] = np.ones(np.shape(dusk_phase)[0])
        dusk_phase['north2'] = np.ones(np.shape(dusk_phase)[0])
        dusk_phase['south2'] = np.ones(np.shape(dusk_phase)[0])

        # Assign grid values
        x1 = np.arange(-15, 405, 30) # 360 - filled
        y1 =np.arange(-15, 405, 30) # 360

        x2 = np.arange(-15, 375, 30) # topographic lines 
        y2 = np.arange(-15, 375, 30)

        for i in range(np.shape(dusk_phase)[0]):
            dusk_phase['north1'].iloc[i] = x1[np.where(dusk_phase['north_deg'].iloc[i] - x1 >= 0)[0][-1]]
            dusk_phase['south1'].iloc[i] = x1[np.where(dusk_phase['south_deg'].iloc[i] - x1 >= 0)[0][-1]]
            dusk_phase['north2'].iloc[i] = x2[np.where(dusk_phase['north_deg'].iloc[i] - x2 >= 0)[0][-1]]
            dusk_phase['south2'].iloc[i] = x2[np.where(dusk_phase['south_deg'].iloc[i] - x2 >= 0)[0][-1]]

        Z1 = pd.pivot_table(dusk_phase, index = 'north1', columns = 'south1', values = 'vals1', aggfunc = sum, fill_value = 0)
        Z2 = pd.pivot_table(dusk_phase, index = 'north2', columns = 'south2', values = 'vals2', aggfunc = sum, fill_value = 0)

        X1, Y1 = np.meshgrid(x1,y1)
        X2, Y2 = np.meshgrid(x2,y2)

        # Special addition to create offset contour plot
        custom = np.zeros(13)
        Z1['360.0'] = custom

        # New row with different column values
        new_row = {Z1.columns[0]: 0, Z1.columns[1]: 0 , Z1.columns[2]: 0, Z1.columns[3]: 0, Z1.columns[4]: 0, Z1.columns[5]: 0 , 
                Z1.columns[6]: 0, Z1.columns[7]: 0, Z1.columns[8]: 0, Z1.columns[9]: 0 , Z1.columns[10]: 0, Z1.columns[11]: 0, Z1.columns[12]: 0, Z1.columns[13]: 0}

        # Convert the new row to a DataFrame
        new_row_df = pd.DataFrame([new_row])

        # Append the new row to the DataFrame
        df = pd.concat([Z1, new_row_df])                
        index_list = df.index.tolist()
        index_list[-1] = 360

        df.index = index_list
        Z1 = df

        # 2d histogram - both contours are [0,15, ..., 345]
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel("North Phase ($^\circ$)", fontsize = 20)
        ax.set_xlabel("South Phase ($^\circ$)", fontsize = 20)
        ax.set_title(f"Dusk (14 - 22 hr LST) LFE Phase Distribution \n  N = {np.shape(dusk_phase)[0]} - Peak at 135-165˚ S and 315-345˚ N", fontsize = 20)

        plot1 = ax.contour(X1, Y1, Z1, levels = 15, cmap = 'spring_r')
        Z_rev = np.matrix(Z2)[::-1]
        plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[-15, 375, -15, 375]) 

        ax.set_xticks(np.arange(-15, 375, 30))
        ax.set_xticklabels(np.arange(-15, 375, 30), rotation = 90)
        ax.set_yticks(np.arange(-15, 375, 30))

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
        dawn_phase['vals1'] = np.ones(np.shape(dawn_phase)[0])
        dawn_phase['north1'] = np.ones(np.shape(dawn_phase)[0])
        dawn_phase['south1'] = np.ones(np.shape(dawn_phase)[0])
        dawn_phase['vals2'] = np.ones(np.shape(dawn_phase)[0])
        dawn_phase['north2'] = np.ones(np.shape(dawn_phase)[0])
        dawn_phase['south2'] = np.ones(np.shape(dawn_phase)[0])

        x1 = np.arange(-15, 405, 30) # 360 - filled
        y1 =np.arange(-15, 405, 30) # 360

        x2 = np.arange(-15, 375, 30) # topographic lines 
        y2 = np.arange(-15, 375, 30)

        for i in range(np.shape(dawn_phase)[0]):
            dawn_phase['north1'].iloc[i] = x1[np.where(dawn_phase['north_deg'].iloc[i] - x1 >= 0)[0][-1]]
            dawn_phase['south1'].iloc[i] = x1[np.where(dawn_phase['south_deg'].iloc[i] - x1 >= 0)[0][-1]]
            dawn_phase['north2'].iloc[i] = x2[np.where(dawn_phase['north_deg'].iloc[i] - x2 >= 0)[0][-1]]
            dawn_phase['south2'].iloc[i] = x2[np.where(dawn_phase['south_deg'].iloc[i] - x2 >= 0)[0][-1]]
        
        Z3 = pd.pivot_table(dawn_phase, index = 'north1', columns = 'south1', values = 'vals1', aggfunc = sum, fill_value = 0)
        Z4 = pd.pivot_table(dawn_phase, index = 'north2', columns = 'south2', values = 'vals2', aggfunc = sum, fill_value = 0)

        X1, Y1 = np.meshgrid(x1,y1)
        X2, Y2 = np.meshgrid(x2,y2)

        custom = np.zeros(13)
        Z3['360.0'] = custom

        # New row with different column values
        new_row = {Z3.columns[0]: 0, Z3.columns[1]: 0 , Z3.columns[2]: 0, Z3.columns[3]: 0, Z3.columns[4]: 0, Z3.columns[5]: 0 , 
                Z3.columns[6]: 0, Z3.columns[7]: 0, Z3.columns[8]: 0, Z3.columns[9]: 0 , Z3.columns[10]: 0, Z3.columns[11]: 0, Z3.columns[12]: 0, Z3.columns[13]: 0}

        # Convert the new row to a DataFrame
        new_row_df = pd.DataFrame([new_row])

        # Append the new row to the DataFrame
        df = pd.concat([Z3, new_row_df])                
        index_list = df.index.tolist()
        index_list[-1] = 360

        df.index = index_list
        Z3 = df

        # 2d histogram - both contours are [0,15, ..., 345]
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel("North Phase ($^\circ$)", fontsize = 20)
        ax.set_xlabel("South Phase ($^\circ$)", fontsize = 20)
        ax.set_title(f"Dawn (02 - 10 hr LST) LFE Phase Distribution \n N = {np.shape(dawn_phase)[0]} - Peak at 105-135˚ S and 285-315˚ N", fontsize = 20)

        plot1 = ax.contour(X1, Y1, Z3, levels = 15, cmap = 'spring_r')
        Z_rev = np.matrix(Z4)[::-1]
        plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[-15, 375, -15, 375]) 

        ax.set_xticks(np.arange(-15, 375, 30))
        ax.set_xticklabels(np.arange(-15, 375, 30), rotation = 90)
        ax.set_yticks(np.arange(-15, 375, 30))

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

        joint_phases_wout_nan['vals1'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        joint_phases_wout_nan['north1'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        joint_phases_wout_nan['south1'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        joint_phases_wout_nan['vals2'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        joint_phases_wout_nan['north2'] = np.ones(np.shape(joint_phases_wout_nan)[0])
        joint_phases_wout_nan['south2'] = np.ones(np.shape(joint_phases_wout_nan)[0])

        x1 = np.arange(-15, 405, 30) # 360
        y1 =np.arange(-15, 405, 30) #360
        
        x2 = np.arange(-15, 375, 30)
        y2 = np.arange(-15, 375, 30)

        for i in range(np.shape(joint_phases_wout_nan)[0]):
            joint_phases_wout_nan['north1'].iloc[i] = x1[np.where(joint_phases_wout_nan['north_deg'].iloc[i] - x1 >= 0)[0][-1]]
            joint_phases_wout_nan['south1'].iloc[i] = x1[np.where(joint_phases_wout_nan['south_deg'].iloc[i] - x1 >= 0)[0][-1]]
            joint_phases_wout_nan['north2'].iloc[i] = x2[np.where(joint_phases_wout_nan['north_deg'].iloc[i] - x2 >= 0)[0][-1]]
            joint_phases_wout_nan['south2'].iloc[i] = x2[np.where(joint_phases_wout_nan['south_deg'].iloc[i] - x2 >= 0)[0][-1]]

        Z1 = pd.pivot_table(joint_phases_wout_nan, index = 'north1', columns = 'south1', values = 'vals1', aggfunc = sum)
        Z2 = pd.pivot_table(joint_phases_wout_nan, index = 'north2', columns = 'south2', values = 'vals2', aggfunc = sum)

        X1, Y1 = np.meshgrid(x1,y1)
        X2, Y2 = np.meshgrid(x2,y2)
        Z1['360.0'] = np.zeros(13)

        # New row with different column values
        new_row = {Z1.columns[0]: 0, Z1.columns[1]: 0 , Z1.columns[2]: 0, Z1.columns[3]: 0, Z1.columns[4]: 0, Z1.columns[5]: 0 , 
                Z1.columns[6]: 0, Z1.columns[7]: 0, Z1.columns[8]: 0, Z1.columns[9]: 0 , Z1.columns[10]: 0, Z1.columns[11]: 0, Z1.columns[12]: 0, Z1.columns[13]: 0}

        # Convert the new row to a DataFrame
        new_row_df = pd.DataFrame([new_row])

        # Append the new row to the DataFrame
        df = pd.concat([Z1, new_row_df])                
        index_list = df.index.tolist()
        index_list[-1] = 360

        df.index = index_list
        Z1 = df

        # 2d histogram - both contours are [0,15, ..., 345]
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_ylabel("North Phase ($^\circ$)", fontsize = 20)
        ax.set_xlabel("South Phase ($^\circ$)", fontsize = 20)
        ax.set_title(f"Total LFE Phase Distribution \n N = {np.shape(joint_phases_wout_nan)[0]}", fontsize = 20)

        plot1 = ax.contour(X1, Y1, Z1, levels = 15, cmap = 'spring_r')
        Z_rev = np.matrix(Z2)[::-1]
        plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[-15, 375, -15, 375]) 

        ax.set_xticks(np.arange(-15, 375, 30))
        ax.set_xticklabels(np.arange(-15, 375, 30), rotation = 90)
        ax.set_yticks(np.arange(-15, 375, 30))

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
    lfe_phase_no_nan['vals1'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    lfe_phase_no_nan['north1'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    lfe_phase_no_nan['south1'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    lfe_phase_no_nan['vals2'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    lfe_phase_no_nan['north2'] = np.ones(np.shape(lfe_phase_no_nan)[0])
    lfe_phase_no_nan['south2'] = np.ones(np.shape(lfe_phase_no_nan)[0])

    x1 = np.arange(-15, 405, 30) # 360 - filled
    y1 =np.arange(-15, 405, 30) # 360

    x2 = np.arange(-15, 375, 30) # topographic lines 
    y2 = np.arange(-15, 375, 30)

    for i in range(np.shape(lfe_phase_no_nan)[0]):
        lfe_phase_no_nan['north1'].iloc[i] = x1[np.where(lfe_phase_no_nan['local north phase'].iloc[i] - x1 >= 0)[0][-1]]
        lfe_phase_no_nan['south1'].iloc[i] = x1[np.where(lfe_phase_no_nan['local south phase'].iloc[i] - x1 >= 0)[0][-1]]
        lfe_phase_no_nan['north2'].iloc[i] = x2[np.where(lfe_phase_no_nan['local north phase'].iloc[i] - x2 >= 0)[0][-1]]
        lfe_phase_no_nan['south2'].iloc[i] = x2[np.where(lfe_phase_no_nan['local south phase'].iloc[i] - x2 >= 0)[0][-1]]

    Z5 = pd.pivot_table(lfe_phase_no_nan, index = 'north1', columns = 'south1', values = 'vals1', aggfunc = sum, fill_value = 0)
    Z6 = pd.pivot_table(lfe_phase_no_nan, index = 'north2', columns = 'south2', values = 'vals2', aggfunc = sum, fill_value = 0)

    X1, Y1 = np.meshgrid(x1,y1)
    X2, Y2 = np.meshgrid(x2,y2)

    custom = np.zeros(13)
    Z5['360.0'] = custom

    # New row with different column values
    new_row = {Z5.columns[0]: 0, Z5.columns[1]: 0 , Z5.columns[2]: 0, Z5.columns[3]: 0, Z5.columns[4]: 0, Z5.columns[5]: 0 , 
            Z5.columns[6]: 0, Z5.columns[7]: 0, Z5.columns[8]: 0, Z5.columns[9]: 0 , Z5.columns[10]: 0, Z5.columns[11]: 0, Z5.columns[12]: 0, Z5.columns[13]: 0}

    # Convert the new row to a DataFrame
    new_row_df = pd.DataFrame([new_row])

    # Append the new row to the DataFrame
    df = pd.concat([Z5, new_row_df])                
    index_list = df.index.tolist()
    index_list[-1] = 360

    df.index = index_list
    Z5 = df

    # 2d histogram - both contours are [0,15, ..., 345]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylabel("North Local Phase ($^\circ$)", fontsize = 20)
    ax.set_xlabel("South Local Phase ($^\circ$)", fontsize = 20)
    ax.set_title(f"SPICE-based LFE Phase Distribution \n N = {np.shape(lfe_phase_no_nan)[0]}", fontsize = 20)

    plot1 = ax.contour(X1, Y1, Z5, levels = 15, cmap = 'spring_r')
    Z_rev = np.matrix(Z6)[::-1]
    plot2 = ax.imshow(Z_rev, cmap='Greys_r', extent=[-15, 375, -15, 375]) 

    ax.set_xticks(np.arange(-15, 375, 30))
    ax.set_xticklabels(np.arange(-15, 375, 30), rotation = 90)
    ax.set_yticks(np.arange(-15, 375, 30))

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
