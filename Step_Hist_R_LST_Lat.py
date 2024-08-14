import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap 
from mpl_toolkits import axes_grid1
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from bisect import bisect_left
import configparser
from tqdm import tqdm
from scipy.io import readsav
from matplotlib.patches import Circle, PathPatch

# Defined paths - adjust according to where EPHEMERIS data from SPICE is hosted
config = configparser.ConfigParser()
config.read('config_LFE_stats.ini')   
LFE_data_directory= config['filepaths']['LFE_data_directory'] # Directory where SN_ms_tot_V2.0.csv, SN_d_tot_V2.0.csv, and LFEs_joined.csv are located

original_unet = "lfe_detections_unet_2874.csv"
lfe_joined_list = "LFEs_joined.csv"

one_min_resol_mission = "20040101000000_20170915115700_ephemeris.csv" # whole mission (2004/01/01 00:00:00 - 2017/09/15 11:56:00) coverage
one_min_resol_og = "2004001_2017258_ephemeris.csv" # original LFE list coverage
one_min_resol_joined = "LFEs_joined_ephemeris.csv" # joined LFE list coverage
ppo_file = "mag_phases_2004_2017_final.sav" # Phases Calibration Data

def step_plot_R_LST_Lat(data_directory = LFE_data_directory):
    # Defined by 1 minute resolution of Cassini - Ephemeris Generation SPICE - entire mission 01-01-2004 to 09-15-2017
    trajectories_df = pd.read_csv(data_directory + one_min_resol_mission)
    LFE_og = pd.read_csv(data_directory + one_min_resol_og)
    LFE_join = pd.read_csv(data_directory + one_min_resol_joined)
    og_unet = pd.read_csv(data_directory + original_unet, index_col = 0)
    join_unet = pd.read_csv(data_directory + lfe_joined_list, index_col = 0)

    # Append Range, LST, and Lat to UNET Output Files from SPICE Files
    og_unet['Range'] = LFE_og['R_KSM']
    og_unet['subLST'] = LFE_og['subLST']
    og_unet['subLat'] = LFE_og['subLat']
    og_unet['x_ksm'] = LFE_og['x_KSM']
    og_unet['y_ksm'] = LFE_og['y_KSM']
    og_unet['z_ksm'] = LFE_og['z_KSM']

    join_unet['Range'] = LFE_join['R_KSM']
    join_unet['subLST'] = LFE_join['subLST']
    join_unet['subLat'] = LFE_join['subLat']
    join_unet['x_ksm'] = LFE_join['x_KSM']
    join_unet['y_ksm'] = LFE_join['y_KSM']
    join_unet['z_ksm'] = LFE_join['z_KSM']

    # Use Range for x_label
    max_range = 1451
    max_lst = 24
    max_lat = 80

    range_bin_size = 10
    lst_bin_size = 1
    lat_bin_size = 5

    range_bins = int(max_range / range_bin_size)
    lst_bins = int(max_lst / lst_bin_size)
    lat_bins = int(max_lat / lat_bin_size)

    # Spacecraft ephemeris provided by SPICE files
    spacecraft_times = trajectories_df["datetime"]
    spacecraft_ranges = trajectories_df["R_KSM"]
    spacecraft_LST = trajectories_df["subLST"]
    spacecraft_Lat = trajectories_df["subLat"]

    # Spacecraft positions at LFE occurences
    lfe_starts = join_unet["start"]
    lfe_ranges = join_unet["Range"]
    lfe_LST = join_unet["subLST"]
    lfe_Lat = join_unet["subLat"]

    mesh_inner_edges = {
        "r": (np.arange(0, range_bins + 1) * max_range) / range_bins, # ranging from 1 to max_range - bin size
        "lst": (np.arange(0, lst_bins + 1) * max_lst) / lst_bins,
        "lat": (np.arange(-lat_bins, lat_bins + 1) * max_lat) / lat_bins # need from -80 to 80, not just 0 to 80
    }
    mesh_outer_edges = {
        "r": (np.arange(1, range_bins + 1) * max_range) / range_bins, # ranging from 2 + bin size TO max_range
        "lst": (np.arange(1, lst_bins + 1) * max_lst) / lst_bins,
        "lat": (np.arange(-lat_bins + 1, lat_bins + 1) * max_lat) / lat_bins
    }

    # Bin Initialization
    timeSpentInBin_range = np.zeros(range_bins+1)
    lfe_detections_in_bin_range = np.zeros(range_bins+1)
    norm_detections_in_bin_range = np.zeros(range_bins+1)

    timeSpentInBin_lst = np.zeros(lst_bins+1)
    lfe_detections_in_bin_lst = np.zeros(lst_bins+1)
    norm_detections_in_bin_lst = np.zeros(lst_bins+1)

    timeSpentInBin_lat = np.zeros(2 * lat_bins+1)
    lfe_detections_in_bin_lat = np.zeros(2 * lat_bins+1)
    norm_detections_in_bin_lat = np.zeros(2 * lat_bins+1)

    # RANGE
    for mesh_range in tqdm(range(len(mesh_outer_edges["r"])), desc = "r"):
        # Determines at which time indices (in spacecraft's COMPLETE ephemeris data) it will be within the current bin
        time_indices_in_region = np.where((spacecraft_ranges <= mesh_outer_edges["r"][mesh_range]) & (spacecraft_ranges > mesh_inner_edges["r"][mesh_range]))
            # How much total time during the entire mission Cassini spent at that specific range (i.e. 10->20 Rs)
        
        # Determines at which time indices (in joined LFE list) it will be in the current bin
        lfe_indices_in_region = np.where((lfe_ranges <= mesh_outer_edges["r"][mesh_range]) & (lfe_ranges > mesh_inner_edges["r"][mesh_range]))

        # Get the time spent in the current bin TRANSFORMED into minutes
        timeInRegion = len(time_indices_in_region[0])
        if timeInRegion == 0: timeInRegion = 0
        timeSpentInBin_range[mesh_range] = timeInRegion

        lfe_detections_in_region = len(lfe_indices_in_region[0])
        if lfe_detections_in_region == 0: lfe_detections_in_region = 0
        lfe_detections_in_bin_range[mesh_range] = lfe_detections_in_region

    # LST
    for mesh_lst in tqdm(range(len(mesh_outer_edges["lst"])), desc = "lst"):
        # Determines at which time indices (in spacecraft's COMPLETE ephemeris data) it will be within the current bin
        time_indices_in_region = np.where((spacecraft_LST < mesh_outer_edges["lst"][mesh_lst]) & (spacecraft_LST >= mesh_inner_edges["lst"][mesh_lst]))
            # How much total time during the entire mission Cassini spent at that specific range (i.e. 10->20 Rs)
        
        # Determines at which time indices (in joined LFE list) it will be in the current bin
        lfe_indices_in_region = np.where((lfe_LST <= mesh_outer_edges["lst"][mesh_lst]) & (lfe_LST > mesh_inner_edges["lst"][mesh_lst]))

        # Get the time spent in the current bin TRANSFORMED into minutes
        timeInRegion = len(time_indices_in_region[0])
        if timeInRegion == 0: timeInRegion = 0
        timeSpentInBin_lst[mesh_lst] = timeInRegion

        lfe_detections_in_region = len(lfe_indices_in_region[0])
        if lfe_detections_in_region == 0: lfe_detections_in_region = 0
        lfe_detections_in_bin_lst[mesh_lst] = lfe_detections_in_region

    # Lat
    for mesh_lat in tqdm(range(len(mesh_outer_edges["lat"])), desc = "lat"):
        # Determines at which time indices (in spacecraft's COMPLETE ephemeris data) it will be within the current bin
        time_indices_in_region = np.where((spacecraft_Lat <= mesh_outer_edges["lat"][mesh_lat]) & (spacecraft_Lat > mesh_inner_edges["lat"][mesh_lat]))
            # How much total time during the entire mission Cassini spent at that specific range (i.e. 10->20 Rs)
        
        # Determines at which time indices (in joined LFE list) it will be in the current bin
        lfe_indices_in_region = np.where((lfe_Lat <= mesh_outer_edges["lat"][mesh_lat]) & (lfe_Lat > mesh_inner_edges["lat"][mesh_lat]))

        # Get the time spent in the current bin TRANSFORMED into minutes
        timeInRegion = len(time_indices_in_region[0])
        if timeInRegion == 0: timeInRegion = 0
        timeSpentInBin_lat[mesh_lat] = timeInRegion

        lfe_detections_in_region = len(lfe_indices_in_region[0])
        if lfe_detections_in_region == 0: lfe_detections_in_region = 0
        lfe_detections_in_bin_lat[mesh_lat] = lfe_detections_in_region


    # Figures which illustrate Location Spread of LFEs as a function of Cassini Range, LST, and Latitute
    fig, ax = plt.subplots(3, figsize = (10,12))

    # AX[0] = RANGE
    ax[0].set_title("Range Distribution of LFEs ")
    x0 = np.arange(0, max_range+1, range_bin_size)
    p1, = ax[0].step(x0, timeSpentInBin_range / sum(timeSpentInBin_range), "b", where = "post")
    # p1, = ax[0].plot(x0, timeSpentInBin_range / sum(timeSpentInBin_range), "b-", label = "Normalized Spacecraft Residence Time")
    ax[0].set_xlabel("Range Bins (Saturn Radii)")
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(0)
    ax[0].set_ylabel("Normalized Spacecraft Residence Time", fontsize = 10)
    twin1 = ax[0].twinx()

    p2, = twin1.step(x0, (lfe_detections_in_bin_range / 4553) / (timeSpentInBin_range / sum(timeSpentInBin_range)), "r",  where = "post")
    #p2, = twin1.plot(x0, (lfe_detections_in_bin_range / 4553) / (timeSpentInBin_range / sum(timeSpentInBin_range)), "r-", label = "Normalized LFE Occurrence Rate")
    twin1.set_ylabel("Normalized LFE Occurrence Rate", fontsize = 10)
    twin1.set_xlim(0, 100)
    twin1.set_ylim(0, 2)
    ax[0].yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    ax[0].set_xticks(np.arange(0, 100, 10))
    ax[0].set_xticks(np.arange(0, 100, 2), minor = True)
    ax[0].tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax[0].tick_params(axis='x', **tkw)

    ax[0].legend(handles=[p1, p2], loc = [0.6,0.80])

    # AX[1] = LST
    ax[1].set_title("Local Time (LST) Distribution of LFEs ")
    x1 = np.arange(0, max_lst+1, lst_bin_size)
    p1, = ax[1].step(x1, timeSpentInBin_lst / sum(timeSpentInBin_lst), "b", where = 'post')
    #p1, = ax[1].plot(x1, timeSpentInBin_lst / sum(timeSpentInBin_lst), "b-", label = "Normalized Spacecraft Residence Time")
    ax[1].set_xlabel("LST Bins (Hours)")
    ax[1].set_xlim(0, max_lst - lst_bin_size)
    ax[1].set_ylim(0)
    ax[1].set_ylabel("Normalized Spacecraft Residence Time", fontsize = 10)
    twin1 = ax[1].twinx()

    p2, = twin1.step(x1,(lfe_detections_in_bin_lst / 4553) / (timeSpentInBin_lst / sum(timeSpentInBin_lst)), "r", where = "post")
    #p2, = twin1.plot(x1, (lfe_detections_in_bin_lst / 4553) / (timeSpentInBin_lst / sum(timeSpentInBin_lst)), "r-", label = "Normalized LFE Occurrence Rate")
    twin1.set_ylabel("Normalized LFE Occurrence Rate", fontsize = 10)
    twin1.set_xlim(0, max_lst - lst_bin_size)
    ax[1].yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    ax[1].tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax[1].tick_params(axis='x', which = 'minor', length = 3.5)
    ax[1].tick_params(axis='x', **tkw)
    ax[1].set_xticks(np.arange(0, 25, 4))
    ax[1].set_xticks(np.arange(0, 25, 1), minor = True)

    ax[1].legend(handles=[p1, p2], loc = [0.6,0.80])

    # AX[2] = Lat
    ax[2].set_title("Latitude (Lat) Distribution of LFEs ")
    x2 = np.arange(-max_lat, max_lat+1, lat_bin_size)
    p1, = ax[2].step(x2, timeSpentInBin_lat / sum(timeSpentInBin_lat), 'b', where = 'post') # interval [x[i], x[i+1]] are assigned value y[i]
    #p1, = ax[2].plot(x2, timeSpentInBin_lat / sum(timeSpentInBin_lat), "b-", label = "Normalized Spacecraft Residence Time")
    ax[2].set_xlabel("Lat Bins (Degrees)")
    ax[2].set_xlim(-max_lat, max_lat )
    ax[2].set_ylim(0)
    ax[2].set_ylabel("Normalized Spacecraft Residence Time", fontsize = 10)
    twin1 = ax[2].twinx()

    p2, = twin1.step(x2, (lfe_detections_in_bin_lat / 4553) / (timeSpentInBin_lat / sum(timeSpentInBin_lat)), 'red', where = 'post')
    #p2, = twin1.plot(x2, (lfe_detections_in_bin_lat / 4553) / (timeSpentInBin_lat / sum(timeSpentInBin_lat)), "r-", label = "Normalized LFE Occurrence Rate")
    twin1.set_ylabel("Normalized LFE Occurrence Rate", fontsize = 10)
    twin1.set_xlim(-max_lat, max_lat)
    ax[2].yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    ax[2].tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    #ax[1].tick_params(axis='x', which = 'minor', length = 3.5)
    ax[2].tick_params(axis='x', **tkw)
    ax[2].set_xticks(np.arange(-80, 81, 10))
    ax[2].set_xticks(np.arange(-80, 81, 2), minor = True)

    ax[2].legend(handles=[p1, p2], loc = [0.6,0.80])

    plt.tight_layout()
    plt.show()
    #plt.savefig("R_LST_Lat.jpeg", dpi = 300)

# Actual Code to Plot Step Plot of LFE Distribution 
step_plot_R_LST_Lat(data_directory = LFE_data_directory)