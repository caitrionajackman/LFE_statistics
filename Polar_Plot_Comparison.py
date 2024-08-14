import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
import configparser
import os

# Initalize Directories
config = configparser.ConfigParser()
config.read('config_LFE_stats.ini')   
data_directory = config['filepaths']['LFE_data_directory'] # Directory where SN_ms_tot_V2.0.csv, SN_d_tot_V2.0.csv, and LFEs_joined.csv are located
input_data_fp= config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']

def polar_plot_lfe_comparison(input_data_fp):
    join_unet = pd.read_csv(input_data_fp + "/LFEs_joined_times_range_lst_lat.csv", index_col = 0) 
    traj_df = pd.read_csv(input_data_fp + "/20040101000000_20170915115700_ephemeris.csv")

    # pivot_table creation to count the amount of occurrences 
    join_unet['vals1'] = np.ones(np.shape(join_unet)[0])
    join_unet['lst1'] = np.ones(np.shape(join_unet)[0])
    join_unet['ranges1'] = np.ones(np.shape(join_unet)[0])

    # Set x, y conditions
    lst_times = np.arange(0, 25, 1)
    ranges = np.arange(0, 110, 10)

    # Restrict Range to [0,100]
    join_unet_restrict = join_unet.iloc[np.where(join_unet['Range (RS)'] <= 100)[0]].reset_index()

    # Bin appropriately to conditions
    for i in range(np.shape(join_unet_restrict)[0]):
        join_unet_restrict['lst1'].iloc[i] = lst_times[np.where(join_unet_restrict['LocalTime (hr)'].iloc[i] - lst_times >= 0)[0][-1]]
        join_unet_restrict['ranges1'].iloc[i] = ranges[np.where(join_unet_restrict['Range (RS)'].iloc[i] - ranges >= 0)[0][-1]]

    Z1 = pd.pivot_table(join_unet_restrict, index = 'ranges1', columns = 'lst1', values = 'vals1', aggfunc = sum, fill_value = 0)
    LST, R = np.meshgrid(lst_times, ranges)

    # Use Range for x_label
    max_range = 100
    max_lst = 24

    range_bin_size = 10
    lst_bin_size = 1

    range_bins = int(max_range / range_bin_size)
    lst_bins = int(max_lst / lst_bin_size)

    # Spacecraft ephemeris provided by SPICE files
    spacecraft_times = traj_df["datetime"]
    spacecraft_ranges = traj_df["R_KSM"]
    spacecraft_LST = traj_df["subLST"]

    # Spacecraft positions at LFE occurences
    lfe_starts = join_unet["start"]
    lfe_ranges = join_unet["Range (RS)"]
    lfe_LST = join_unet["LocalTime (hr)"]

    mesh_inner_edges = {
        "r": (np.arange(0, range_bins + 1) * max_range) / range_bins, # ranging from 1 to max_range - bin size
        "lst": (np.arange(0, lst_bins + 1) * max_lst) / lst_bins
    }
    mesh_outer_edges = {
        "r": (np.arange(1, range_bins + 1) * max_range) / range_bins, # ranging from 2 + bin size TO max_range
        "lst": (np.arange(1, lst_bins + 1) * max_lst) / lst_bins
    }

    # Bin Initialization
    timeSpentInBin = np.zeros((range_bins + 1, lst_bins + 1))
    lfe_detections_in_bin = np.zeros((range_bins + 1, lst_bins + 1))
    lfe_duration_in_bin = np.zeros((range_bins + 1, lst_bins + 1))
    norm_detections_in_bin= np.zeros((range_bins + 1, lst_bins + 1))
    norm_duration_in_bin= np.zeros((range_bins + 1, lst_bins + 1))
    norm_det_dur_in_bin = np.zeros((range_bins + 1, lst_bins + 1))

    # R & LST Pairing
    for mesh_r in range(len(mesh_outer_edges["r"])):
        for mesh_lst in range(len(mesh_outer_edges["lst"])):
            # Determines at which time indices (in spacecraft's COMPLETE ephemeris data) it will be within the current bin
            time_indices_in_region = np.where((spacecraft_LST < mesh_outer_edges["lst"][mesh_lst]) & (spacecraft_LST >= mesh_inner_edges["lst"][mesh_lst]) &\
                                            (spacecraft_ranges < mesh_outer_edges["r"][mesh_r]) & (spacecraft_ranges >= mesh_inner_edges["r"][mesh_r]))
                # How much total time during the entire mission Cassini spent at that specific range (i.e. 10->20 Rs)
            
            # Determines at which time indices (in joined LFE list) it will be in the current bin
            lfe_indices_in_region = np.where((lfe_LST <= mesh_outer_edges["lst"][mesh_lst]) & (lfe_LST > mesh_inner_edges["lst"][mesh_lst]) &\
                                            (lfe_ranges <= mesh_outer_edges["r"][mesh_r]) & (lfe_ranges > mesh_inner_edges["r"][mesh_r]))

            # Get the time spent in the current bin TRANSFORMED into minutes
            timeInRegion = len(time_indices_in_region[0])
            if timeInRegion == 0: timeInRegion = np.nan
            timeSpentInBin[mesh_r][mesh_lst] = timeInRegion

            lfe_detections_in_region = len(lfe_indices_in_region[0])
            if lfe_detections_in_region == 0: lfe_detections_in_region = 0
            lfe_detections_in_bin[mesh_r][mesh_lst] = lfe_detections_in_region

            duration_sums = np.sum(join_unet.iloc[lfe_indices_in_region[0]]['duration'])
            if len(lfe_indices_in_region[0]) == 0: duration_sums = 0
            lfe_duration_in_bin[mesh_r][mesh_lst] = duration_sums / (60*60) # In Hours

            norm_detections_in_bin[mesh_r][mesh_lst] = (lfe_detections_in_region / (timeInRegion / 60)) * 100 # lfe / hour count
            
            norm_duration_in_bin[mesh_r][mesh_lst] = ((duration_sums / (60*60)) / (timeInRegion / 60)) * 100 # Total LFE Duration / Observing TIme in Bin (Hrs)

            norm_det_dur_in_bin[mesh_r][mesh_lst] = norm_duration_in_bin[mesh_r][mesh_lst] / norm_detections_in_bin[mesh_r][mesh_lst] # Average lfe duration in bin

    # Shading flat requires removing last point
    timeSpentInBin = timeSpentInBin[:-1,:-1]
    lfe_detections_in_bin = lfe_detections_in_bin[:-1,:-1]
    norm_detections_in_bin = norm_detections_in_bin[:-1,:-1]
    lfe_duration_in_bin = lfe_duration_in_bin[:-1, :-1]
    norm_duration_in_bin=norm_duration_in_bin[:-1, :-1]
    norm_det_dur_in_bin = norm_det_dur_in_bin[:-1, :-1]

    total_cassini_time = np.shape(traj_df)[0]
    timeSpentInBin_per = (timeSpentInBin / total_cassini_time)*100

    # Figure set up
    fig = plt.figure(figsize = (15, 10))
    ax_time = fig.add_subplot(2, 3, 1, projection = "polar") # 1st subplot in layout with 1 row and 3 columns of subplots
    ax_lfes = fig.add_subplot(2, 3, 2, projection = "polar")
    ax_lfdur = fig.add_subplot(2, 3, 3, projection = "polar")
    ax_norm = fig.add_subplot(2, 3, 4, projection = "polar")
    ax_dur = fig.add_subplot(2, 3, 5, projection = "polar")
    ax_dur_det = fig.add_subplot(2, 3, 6, projection = "polar")

    csize=10

    # Label the Ticks
    xtickpos = []
    xticklab = []
    for i in range(0, 24):
        xtickpos.append((i) * ((2*np.pi)/24.0))
        xticklab.append('%02d' % (i))

    # Convert LST into angle (considering 00 at 0deg)
    lst_times = (np.arange(0, 25)) * ((2*np.pi) / 24)
    ranges = np.arange(0, 110, 10)

    # Uniform Meshgrid 
    LST, R = np.meshgrid(lst_times, ranges)
    cmap = copy.copy(matplotlib.cm.get_cmap("inferno"))
    cmap.set_under(color='grey')  

    # % observing time plot
    ax_time.set_title("Residence Time (%) \n 0 ($R_S$) < Range < 100 ($R_S$)", fontsize = csize * (17/15))
    pcm_time = ax_time.pcolor(LST, R, timeSpentInBin_per, edgecolor = 'white', linewidth = 0.2, cmap = cmap)
    cbar_time = fig.colorbar(pcm_time, ax = ax_time, fraction = 0.046, pad = 0.1, orientation = 'horizontal')
    cbar_time.ax.tick_params(labelsize = csize)
    cbar_time.set_label(size = csize, label = "% Observing Time")
    #pcm_time.set_clim(np.min(timeSpentInBin_per, where=~np.isnan(timeSpentInBin_per), initial = 1), np.max(timeSpentInBin_per, where=~np.isnan(timeSpentInBin_per), initial = 1))
    ax_time.set_facecolor("grey")

    ax_time.spines['polar'].set_visible(False)
    ax_time.set_xticks(xtickpos)
    ax_time.set_xticklabels(xticklab)
    ax_time.set_yticks([10,20,30,40,50,60,70,80,90])
    ax_time.set_yticklabels([10, 20, 30, 40 , 50, 60, 70, 80, 90], color = 'white', fontsize = csize * (17/15))
    ax_time.set_theta_zero_location('N') # Make sure 00 is at NORTH Position
    ax_time.set_theta_direction("counterclockwise")
    ax_time.tick_params(labelsize=csize)
    ax_time.text(-150, 165,'(a)', va = 'bottom', ha = 'right', fontsize = csize * (17/15))
    lab_pos = ax_time.get_rlabel_position()
    ax_time.text(np.radians(lab_pos - 15), ax_time.get_rmax() * 0.75, '$R_S$', ha = 'center', va = 'center', fontsize = csize, color = 'white')


    # LFE count plot
    ax_lfes.set_title("LFE Detections \n 0 ($R_S$) < Range < 100 ($R_S$)", fontsize = csize * (17/15))
    pcm_lfe = ax_lfes.pcolor(LST, R, lfe_detections_in_bin, edgecolor = 'white', linewidth = 0.2, cmap = cmap, vmin = 1)
    cbar_lfe = fig.colorbar(pcm_lfe, ax = ax_lfes, fraction = 0.046, pad = 0.1, orientation = 'horizontal')
    cbar_lfe.ax.tick_params(labelsize = csize)
    cbar_lfe.set_label(size = csize, label = "LFE Detections")
    cbar_lfe.set_ticks([1, 20, 40, 60, 80, 100, 120, 140, 160])
    ax_lfes.set_facecolor("grey")

    ax_lfes.spines['polar'].set_visible(False)
    ax_lfes.set_yticks([10,20,30,40,50,60,70,80,90])
    ax_lfes.set_yticklabels([10, 20, 30, 40 , 50, 60, 70, 80, 90], color = 'white', fontsize = csize * (17/15))
    ax_lfes.set_xticks(xtickpos)
    ax_lfes.set_xticklabels(xticklab, color = 'black')
    ax_lfes.set_theta_zero_location('N') # Make sure 00 is at NORTH Position
    ax_lfes.set_theta_direction("counterclockwise")
    ax_lfes.tick_params(labelsize=csize)
    ax_lfes.text(-150, 165,'(b)', va = 'bottom', ha = 'right', fontsize = csize * (17/15))
    lab_pos = ax_lfes.get_rlabel_position()
    ax_lfes.text(np.radians(lab_pos - 15), ax_lfes.get_rmax() * 0.75, '$R_S$', ha = 'center', va = 'center', fontsize = csize, color = 'white')

    # LFE Duration
    ax_lfdur.set_title("Total LFE Duration \n 0 ($R_S$) < Range < 100 ($R_S$)", fontsize = csize * (17/15))
    pcm_lfdur = ax_lfdur.pcolor(LST, R, lfe_duration_in_bin, edgecolor = 'white', linewidth = 0.2, cmap = cmap, vmin = 1)
    cbar_lfdur = fig.colorbar(pcm_lfdur, ax = ax_lfdur, fraction = 0.046, pad = 0.1, orientation = 'horizontal')
    cbar_lfdur.ax.tick_params(labelsize = csize)
    cbar_lfdur.set_label(size = csize, label = "Summed LFE Duration (hr)")
    ax_lfdur.set_facecolor("grey")

    ax_lfdur.spines['polar'].set_visible(False)
    ax_lfdur.set_yticks([10,20,30,40,50,60,70,80,90])
    ax_lfdur.set_yticklabels([10, 20, 30, 40 , 50, 60, 70, 80, 90], color = 'white', fontsize = csize * (17/15))
    ax_lfdur.set_xticks(xtickpos)
    ax_lfdur.set_xticklabels(xticklab, color = 'black')
    ax_lfdur.set_theta_zero_location('N') # Make sure 00 is at NORTH Position
    ax_lfdur.set_theta_direction("counterclockwise")
    ax_lfdur.tick_params(labelsize=csize)
    ax_lfdur.text(-150, 165,'(c)', va = 'bottom', ha = 'right', fontsize = csize * (17/15))
    lab_pos = ax_lfdur.get_rlabel_position()
    ax_lfdur.text(np.radians(lab_pos - 15), ax_lfdur.get_rmax() * 0.75, '$R_S$', ha = 'center', va = 'center', fontsize = csize, color = 'white')


    # Detection Normalization plot
    ax_norm.set_title("LFE Detections normalized by Observing Time \n 0 ($R_S$) < Range < 100 ($R_S$)", fontsize = csize * (17/15))
    pcm_norm = ax_norm.pcolor(LST, R, norm_detections_in_bin, edgecolor = 'white', linewidth = 0.2, cmap = cmap)
    cbar_norm = fig.colorbar(pcm_norm, ax = ax_norm, fraction = 0.046, pad = 0.1, orientation = 'horizontal', )
    cbar_norm.ax.tick_params(labelsize = csize)
    cbar_norm.set_label(size = csize* (20/15), label = r' $\frac{LFE\; Detections}{Observing\; Time\; per\; Bin\;}$ $\times$ 100  [$hr^{-1}$]' )
    ax_norm.set_facecolor("grey")

    ax_norm.spines['polar'].set_visible(False)
    ax_norm.set_yticks([10,20,30,40,50,60,70,80,90])
    ax_norm.set_yticklabels([10, 20, 30, 40 , 50, 60, 70, 80, 90], color = 'white', fontsize = csize * (17/15))
    ax_norm.set_xticks(xtickpos)
    ax_norm.set_xticklabels(xticklab)
    ax_norm.set_theta_zero_location('N') # Make sure 00 is at NORTH Position
    ax_norm.set_theta_direction("counterclockwise")
    ax_norm.tick_params(labelsize=csize)
    ax_norm.text(-150, 165,'(d)', va = 'bottom', ha = 'right', fontsize = csize * (17/15))
    lab_pos = ax_norm.get_rlabel_position()
    ax_norm.text(np.radians(lab_pos - 15), ax_norm.get_rmax() * 0.75, '$R_S$', ha = 'center', va = 'center', fontsize = csize, color = 'white')


    # Duration Normalization plot
    ax_dur.set_title("Total LFE Duration normalized by Observing Time \n 0 ($R_S$) < Range < 100 ($R_S$)", fontsize = csize * (17/15))
    pcm_dur = ax_dur.pcolor(LST, R, norm_duration_in_bin, edgecolor = 'white', linewidth = 0.2, cmap = cmap)
    cbar_dur = fig.colorbar(pcm_dur, ax = ax_dur, fraction = 0.046, pad = 0.1, orientation = 'horizontal', )
    cbar_dur.ax.tick_params(labelsize = csize)
    cbar_dur.set_label(size = csize* (20/15), label = r' $\frac{Summed \; LFE\; Duration}{Observing\; Time\; per\; Bin\;}$ $\times$ 100' )
    ax_dur.set_facecolor("grey")

    ax_dur.spines['polar'].set_visible(False)
    ax_dur.set_yticks([10,20,30,40,50,60,70,80,90])
    ax_dur.set_yticklabels([10, 20, 30, 40 , 50, 60, 70, 80, 90], color = 'white', fontsize = csize * (17/15))
    ax_dur.set_xticks(xtickpos)
    ax_dur.set_xticklabels(xticklab)
    ax_dur.set_theta_zero_location('N') # Make sure 00 is at NORTH Position
    ax_dur.set_theta_direction("counterclockwise")
    ax_dur.tick_params(labelsize=csize)
    ax_dur.text(-150, 165,'(e)', va = 'bottom', ha = 'right', fontsize = csize * (17/15))
    lab_pos = ax_dur.get_rlabel_position()
    ax_dur.text(np.radians(lab_pos - 15), ax_norm.get_rmax() * 0.75, '$R_S$', ha = 'center', va = 'center', fontsize = csize, color = 'white')

    # Duration / Detection Normalization plot
    ax_dur_det.set_title("Total LFE Duration normalized by LFE Detections per Bin \n 0 ($R_S$) < Range < 100 ($R_S$)", fontsize = csize * (17/15))
    pcm_dur_det = ax_dur_det.pcolor(LST, R, norm_det_dur_in_bin, edgecolor = 'white', linewidth = 0.2, cmap = cmap)
    cbar_dur_det = fig.colorbar(pcm_dur_det, ax = ax_dur_det, fraction = 0.046, pad = 0.1, orientation = 'horizontal', )
    cbar_dur_det.ax.tick_params(labelsize = csize)
    cbar_dur_det.set_label(size = csize* (20/15), label = r' $\frac{Summed \; LFE\; Duration}{LFE\; Detections\; per\; Bin\;}$' )
    ax_dur_det.set_facecolor("grey")

    ax_dur_det.spines['polar'].set_visible(False)
    ax_dur_det.set_yticks([10,20,30,40,50,60,70,80,90])
    ax_dur_det.set_yticklabels([10, 20, 30, 40 , 50, 60, 70, 80, 90], color = 'white', fontsize = csize * (17/15))
    ax_dur_det.set_xticks(xtickpos)
    ax_dur_det.set_xticklabels(xticklab)
    ax_dur_det.set_theta_zero_location('N') # Make sure 00 is at NORTH Position
    ax_dur_det.set_theta_direction("counterclockwise")
    ax_dur_det.tick_params(labelsize=csize)
    ax_dur_det.text(-150, 165,'(f)', va = 'bottom', ha = 'right', fontsize = csize * (17/15))
    lab_pos = ax_dur_det.get_rlabel_position()
    ax_dur_det.text(np.radians(lab_pos - 15), ax_dur_det.get_rmax() * 0.75, '$R_S$', ha = 'center', va = 'center', fontsize = csize, color = 'white')

    fig.tight_layout()
    plt.savefig("/Users/hannamag/Desktop/LFE_statistics_original/Sample_Visualizations/Polar_Plot.jpeg", dpi = 300)
    plt.show()


# Actual Code 
polar_plot_lfe_comparison(input_data_fp= "/Users/hannamag/Desktop/LFE_statistics/cassini_saturn_skr")