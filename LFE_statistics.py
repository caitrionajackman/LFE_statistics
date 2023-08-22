# -*- coding: utf-8 -*-
"""
Created on Friday June 23rd 2023

@author: Caitriona Jackman
"""
import matplotlib.ticker as mticker 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from matplotlib.colors import ListedColormap 
from mpl_toolkits import axes_grid1
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from bisect import bisect_left
import configparser
from tqdm import tqdm
from scipy.io import readsav

def main():
    plt.rcParams.update({'font.size': 12})

    unet=True
    data_directory = "./../data/"
    # data_directory = "C:/Users/Local Admin/Documents/Collaborations/Jackman_LFEs/"
    lfe_unet_data = "lfe_detections_unet.csv" # Processed using findDetectionPositions.py
    lfe_training_data = "lfe_detections_training.csv" # "
    trajectories_file = "cassini_output/trajectorytotal.csv" # Output from Beth's "Cassini_Plotting" repo
    ppo_file = "mag_phases_2004_2017_final.sav"
    LFE_phase_df = "lfe_with_phase.csv" # File saved by SavePPO()

    lfe_duration_split = 11 # measured in hours #this may change to become the median of the duration distribution (or some other physically meaningful number)

    plot = {
        "duration_histograms": False,
        "inspect_longest_lfes": False,
        #TODO: Add function here to plot spectrogram (call radio data), and overplot polygons (call Unet json file)
        #TODO: Action on the shortest LFEs (compare to Reed 30 minute lower bound criterion)
        "residence_time_multiplots": True,
        "lfe_distributions": False,
        "ppo_save": False,  #this takes 4-5 minutes and produces LFE_phase_df
        "ppo_plot": False,
        "local_ppo_plot": False,

        "split_ppo_by_local_time": False, # Split the above PPO plots by local time
        "normalise_histograms": False # Plots probability density instead: density = counts / (sum(counts), * np.diff(bins)). The area under the histogram integrates to 1.
    }

    #Read in LFE list (output of Elizabeth's U-Net run on full Cassini dataset)
    print('Reading in the LFE list')
    #Unet True means the code uses the output of O'Dwyer Unet (4950 examples) 28/07/2023 - before post-processing
    #Unet False means the code uses the training data (984 examples)
    if unet is True:
        LFE_df = pd.read_csv(data_directory + lfe_unet_data, parse_dates=['start','end'])

    else:
        LFE_df = pd.read_csv(data_directory + lfe_training_dataa, parse_dates=['start','end'])


    LFE_duration=LFE_df['end']-LFE_df['start']  #want this in minutes and to be smart about day/year boundaries

    LFE_secs=[]
    for i in range(np.array(LFE_duration).size):
        LFE_secs.append(LFE_duration[i].total_seconds())

    if plot["duration_histograms"]:
        PlotDurationHistogram(LFE_secs, unet=unet)

    if plot["inspect_longest_lfes"]:
        #Next want to explore some manual inspection of the longest LFEs to see if they're "real"
        InspectLongestLFEs(LFE_df, LFE_secs, LFE_duration)

    if plot["residence_time_multiplots"] or plot["lfe_distributions"]:
        print("Loading trajectories...")
        trajectories = pd.read_csv(data_directory + trajectories_file, parse_dates=["datetime_ut"])

    if plot["residence_time_multiplots"]:
        ResidencePlots(trajectories, LFE_df, z_bounds=[-30, 30], unet=unet, saturation_factor=1, local_time_labels=True)
        #TODO: check how saturation factor works (lower priority until after the post-processing is done)
        #TODO: Make quick sorting function to manually examine LFEs in particular LT sectors. Sort the LFEs by hrs of LT        
        #TODO ditto for latitude    
    
    if plot["lfe_distributions"]:       
        PlotLfeDistributions(trajectories, LFE_df, unet=unet, scale="linear", long_lfe_cutoff=lfe_duration_split, normalise_histograms=plot["normalise_histograms"])

    if plot["ppo_save"]:
        SavePPO(data_directory + ppo_file, LFE_df, data_directory, "lfe_with_phase.csv")

    if plot["ppo_plot"]:
        PlotPPO(data_directory + LFE_phase_df, np.arange(0, 360+15, 15), LFE_df, long_lfe_cutoff=lfe_duration_split, local=False, split_by_local_time=plot["split_ppo_by_local_time"], normalise_histograms=plot["normalise_histograms"])

    if plot["local_ppo_plot"]:
        PlotPPO(data_directory + LFE_phase_df, np.arange(0, 360+15, 15), LFE_df, long_lfe_cutoff=lfe_duration_split, local=True, split_by_local_time=plot["split_ppo_by_local_time"], normalise_histograms=plot["normalise_histograms"])

def PlotPPO(file_path, bins, LFE_df, long_lfe_cutoff, unet=True, local=False, split_by_local_time=False, normalise_histograms=False):

    data = pd.read_csv(file_path)

    #TODO: Check with Gabs - is this how to start treatment of PPOs at start?
    north_phase = np.array(data["north phase"]) % 360
    south_phase = np.array(data["south phase"]) % 360

    x = LFE_df["x_ksm"]
    y = LFE_df["y_ksm"]
    z = LFE_df["z_ksm"]

    spacecraft_r, spacecraft_theta, spacecraft_z = CartesiansToCylindrical(x, y, z)

    # Calculate local time
    spacecraft_lt = []
    for longitude_rads in spacecraft_theta:
        longitude_degs = longitude_rads*180/np.pi
        spacecraft_lt.append(((longitude_degs+180)*24/360) % 24)

    azimuth = []
    for lt in spacecraft_lt:
        azimuth.append(((lt-12) * 15 + 720) % 360)


    #differentiate between local phases and "global" phases - and both require similar data
    if local is True:

        local_phase_north = []
        local_phase_south = []
        for north_phase, south_phase, az in zip(north_phase, south_phase, azimuth):
            local_phase_north.append(((north_phase - az) + 720) % 360)
            local_phase_south.append(((south_phase - az) + 720) % 360)

        local_phase_north = np.array(local_phase_north)
        local_phase_south = np.array(local_phase_south)

    #long_lfe_cutoff is set to lfe_duration_split, which for now (28/07) is set to 11 hours.
    short_LFEs = np.where(LFE_df["duration"] <= long_lfe_cutoff*60*60)
    long_LFEs = np.where(LFE_df["duration"] > long_lfe_cutoff*60*60)

    if not split_by_local_time:

        if normalise_histograms is True:
            alpha = 0.7
        else:
            alpha = 1

        if local is False:
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            ax_north, ax_south = axes

            ax_north.hist([north_phase[i] for i in short_LFEs], bins=bins, color="indianred", density=normalise_histograms, alpha=alpha)
            ax_north.hist([north_phase[i] for i in long_LFEs], bins=bins, color="mediumturquoise", density=normalise_histograms, alpha=alpha)

            ax_south.hist([south_phase[i] for i in short_LFEs], bins=bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)
            ax_south.hist([south_phase[i] for i in long_LFEs], bins=bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours", density=normalise_histograms, alpha = alpha)


            ax_north.set_title("North Phase")
            
            ax_south.set_title("South Phase")
            ax_south.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)

            titleTag = "PPO"

        else:
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            ax_local_north, ax_local_south = axes

            ax_local_north.hist([local_phase_north[i] for i in short_LFEs], bins=bins, color="indianred", density=normalise_histograms, alpha=alpha)
            ax_local_north.hist([local_phase_north[i] for i in long_LFEs], bins=bins, color="mediumturquoise", density=normalise_histograms, alpha=alpha)

            ax_local_south.hist([local_phase_south[i] for i in short_LFEs], bins=bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)
            ax_local_south.hist([local_phase_south[i] for i in long_LFEs], bins=bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)

            ax_local_north.set_title("North Local Phase")
            ax_local_south.set_title("South Local Phase")

            ax_local_south.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)

            titleTag = "Local"
        
        for ax in fig.get_axes():
            if normalise_histograms is True:
                ax.set_ylabel("LFE Probability Density")
            else:
                ax.set_ylabel("# of LFEs")

            if ax == fig.get_axes()[-1]:
                ax.set_xlabel("Phase ($^\circ$)")
            ax.margins(x=0)
            ax.set_xticks(bins[0::2])

    else:
        if local:
            north_phase = local_phase_north
            south_phase = local_phase_south
            titleTag = "Local"

        else:
            titleTag = "Global"
    
        # Get the indices where LFE is in each lt sector
        spacecraft_lt = np.array(spacecraft_lt)
        dawn_LFEs = np.where((spacecraft_lt >= 3) & (spacecraft_lt < 9))
        noon_LFEs = np.where((spacecraft_lt >= 9) & (spacecraft_lt < 15))
        dusk_LFEs = np.where((spacecraft_lt >= 15) & (spacecraft_lt < 21))
        midnight_LFEs = np.where((spacecraft_lt >= 21) | (spacecraft_lt < 3))

        lfe_lt_groups = [dawn_LFEs, noon_LFEs, dusk_LFEs, midnight_LFEs]
        lfe_lt_group_names = ["Dawn (3 <= LT < 9)", "Noon (9 <= LT < 15)", "Dusk (15 <= LT < 21)", "Midnight (21 <= LT < 3)"]

        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
        north_axes = ax_north_dawn, ax_north_noon, ax_north_dusk, ax_north_midnight = axes[0]
        south_axes = ax_south_dawn, ax_south_noon, ax_south_dusk, ax_south_midnight = axes[1]

        north_labels = ["(a)", "(b)", "(c)", "(d)"]
        for index, (ax, label) in enumerate(zip(north_axes, north_labels)):
            ax.hist([north_phase[i] for i in lfe_lt_groups[index]])
            ax.margins(x=0)

            if index == 0:
                ax.set_ylabel("North")

            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans, va='bottom')

        south_labels = ["(e)", "(f)", "(g)", "(h)"]
        for index, (ax, label) in enumerate(zip(south_axes, south_labels)):
            ax.hist([south_phase[i] for i in lfe_lt_groups[index]])
            ax.margins(x=0)

            ax.set_xticks(bins[0::4])
            ax.set_xlabel(lfe_lt_group_names[index])

            if index == 0:
                ax.set_ylabel("South")

            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans, va='bottom')

        
        fig.text(0.5, 0.02, 'PPO Phase ($^\circ$)', ha='center', fontsize=16)
        fig.text(0.08, 0.5, '# of LFEs', va='center', rotation='vertical', fontsize=16)

    if unet:
        dataTag="UNET Output"
    else:
        datTag="Training Data"
        
    fig.suptitle(f"Northern and Southern {titleTag} Phases ({dataTag})", fontsize=18)
    plt.subplots_adjust(wspace=0.1, bottom=0.2)

    plt.show()


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
        lfe_start_doy2004 = (lfe_start_time - doy2004_0).total_seconds() / 60 / 60 / 24 # days since 2004-01-01 00:00:00

        # Find minimum time difference
        south_index = (np.abs(south_time - lfe_start_doy2004)).argmin()
        lfe_south_phase_indices.append(south_index)

        north_index = (np.abs(north_time - lfe_start_doy2004)).argmin()
        lfe_north_phase_indices.append(north_index)


    LFE_df["south phase"] = np.array(south_phase)[lfe_south_phase_indices]
    LFE_df["north phase"] = np.array(north_phase)[lfe_north_phase_indices]

    print(f"Saving new csv file to {data_directory+file_name}")
    LFE_df.to_csv(data_directory + file_name)
                       

def PlotDurationHistogram(LFE_secs, unet=True):
    fig, ax = plt.subplots(1, tight_layout=True, sharey = True, figsize=(8,8))
    ax.hist(np.array(LFE_secs)/(60.*24.),bins=np.linspace(0,250,126), label=f"N = {len(LFE_secs)}")

    if unet:
        ax.set_title('Histogram of duration of LFEs across Cassini mission (UNET Output)')
    else:
        ax.set_title('Histogram of duration of LFEs across Cassini mission (Training Data)')

    ax.set_xlabel('LFE duration (hours)')
    ax.set_ylabel('# of LFEs')
    ax.set_xscale('log')

    median = np.median(np.array(LFE_secs)/(60.*24.))
    mean = np.mean(np.array(LFE_secs)/(60.*24.))

    ax.axvline(x=median, color="indianred", linewidth=2, label=f"Median: {median:.2f} hours")
    ax.axvline(x=mean, color="indianred", linewidth=2, linestyle="dashed", label=f"Mean: {mean:.2f} hours")

    plt.legend()

    plt.show()



def InspectLongestLFEs(LFE_df, LFE_secs, LFE_duration):
    LFE_df['LFE_duration']=LFE_duration
    LFE_df['LFE_secs']=LFE_secs

    LFE_df_sorted=LFE_df.sort_values(by=['LFE_secs'],ascending=False)
    print(LFE_df_sorted)

    #Want to be able to look at these spectrograms to see if any need to be removed as outliers/unphysical


def ResidencePlots(trajectories_df, LFE_df, z_bounds, max_r=80, r_bin_size=10, theta_bins=24, unet=True, saturation_factor=1, local_time_labels=True):

    r_bins = int(max_r / r_bin_size)

    spacecraft_times = trajectories_df["datetime_ut"]
    x = trajectories_df["xpos_ksm"]
    y = trajectories_df["ypos_ksm"]
    z = trajectories_df["zpos_ksm"]

    spacecraft_r, spacecraft_theta, spacecraft_z = CartesiansToCylindrical(x, y, z)

    lfe_x, lfe_y, lfe_z = (LFE_df["x_ksm"], LFE_df["y_ksm"], LFE_df["z_ksm"])
    lfe_r, lfe_theta, lfe_z = CartesiansToCylindrical(lfe_x, lfe_y, lfe_z)

    mesh_outer_edges={
        "r": (np.arange(1, r_bins+1)*max_r)/r_bins, # ranging from 2 + bin size to max_r
        "theta": (np.arange(1, theta_bins+1)*np.pi)/(theta_bins/2) - np.pi # ranging from -pi + binsize to pi
    }
    mesh_inner_edges={
        "r": (np.arange(0, r_bins+1)*max_r)/r_bins, # ranging from 1 to max_r - bin size
        "theta": (np.arange(0, theta_bins+1)*np.pi)/(theta_bins/2) - np.pi # ranging from -pi to pi - bin size
    }

    
    print("Comparing spacecraft location to bins")
    timeSpentInBin = np.zeros((r_bins+1, theta_bins+1))
    lfe_detections_in_bin = np.zeros((r_bins+1, theta_bins+1))
    norm_detections_in_bin = np.zeros((r_bins+1, theta_bins+1))

    for mesh_r in tqdm(range(len(mesh_outer_edges["r"])), desc="r"):
        for mesh_theta in tqdm(range(len(mesh_outer_edges["theta"])), desc="theta", leave=False):
            # Determines at what time indices is the spacecraft within the current bin
            time_indices_in_region = np.where((spacecraft_r <= mesh_outer_edges["r"][mesh_r]) & (spacecraft_r > mesh_inner_edges["r"][mesh_r]) &\
                         (spacecraft_theta <= mesh_outer_edges["theta"][mesh_theta]) & (spacecraft_theta > mesh_inner_edges["theta"][mesh_theta]) &\
                         (spacecraft_z <= np.max(z_bounds)) & (spacecraft_z > np.min(z_bounds)))[0]

            lfe_indices_in_region = np.where((lfe_r <= mesh_outer_edges["r"][mesh_r]) & (lfe_r > mesh_inner_edges["r"][mesh_r]) &\
                         (lfe_theta <= mesh_outer_edges["theta"][mesh_theta]) & (lfe_theta > mesh_inner_edges["theta"][mesh_theta]) &\
                         (lfe_z <= np.max(z_bounds)) & (lfe_z > np.min(z_bounds)))[0]


            # Get the time spent in the current bin in minutes
            timeInRegion = len(time_indices_in_region)
            if timeInRegion == 0: timeInRegion = np.nan
            timeSpentInBin[mesh_r][mesh_theta] = timeInRegion

            lfe_detections_in_region = len(lfe_indices_in_region)
            if lfe_detections_in_region == 0: lfe_detections_in_region = np.nan
            lfe_detections_in_bin[mesh_r][mesh_theta] = lfe_detections_in_region

            norm_detections_in_bin[mesh_r][mesh_theta] = lfe_detections_in_region / (timeInRegion/60) # conver to per hour
            

    # Shading flat requires removing last point
    timeSpentInBin = timeSpentInBin[:-1,:-1]
    lfe_detections_in_bin = lfe_detections_in_bin[:-1,:-1]
    norm_detections_in_bin = norm_detections_in_bin[:-1,:-1]

    fig = plt.figure(figsize=(16.5, 10))

    ax_cartesian = fig.add_subplot(1, 3, 1, zorder=10)
    ax_cartesian.patch.set_alpha(0)
    ax_polar = fig.add_axes(ax_cartesian.get_position(), polar=True, zorder=-10)
    
    ax_cartesian.set_title(f"Residence Time\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")

    ax_cartesian_lfe = fig.add_subplot(1, 3, 2, zorder=10)
    ax_cartesian_lfe.patch.set_alpha(0)
    ax_polar_lfe = fig.add_axes(ax_cartesian_lfe.get_position(), polar=True, zorder=-10)
    
    if unet:
        ax_cartesian_lfe.set_title(f"LFE Detections (UNET Output)\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")
    else:
        ax_cartesian_lfe.set_title(f"LFE Detections (Training Data)\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")

    ax_cartesian_norm = fig.add_subplot(1, 3, 3, zorder=10)
    ax_cartesian_norm.patch.set_alpha(0)
    ax_polar_norm = fig.add_axes(ax_cartesian_norm.get_position(), polar=True, zorder=-10)
    
    ax_cartesian_norm.set_title(f"Detections normalised by Residence Time\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")

    
    timeMesh = ax_polar.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], timeSpentInBin/60, cmap="viridis", shading="flat")

    lfeMesh = ax_polar_lfe.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], lfe_detections_in_bin, vmax=np.nanmax(lfe_detections_in_bin)/saturation_factor, cmap="magma", shading="flat", zorder=-1)
    ax_polar_lfe.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], timeSpentInBin, cmap=ListedColormap(["lightgrey"]), shading="flat", zorder=-2)

    normMesh = ax_polar_norm.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], norm_detections_in_bin, vmax=np.nanmax(norm_detections_in_bin)/saturation_factor, cmap="plasma", shading="flat", zorder=-1)
    ax_polar_norm.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], timeSpentInBin, cmap=ListedColormap(["lightgrey"]), shading="flat", zorder=-2)

    
    polar_axes = [ax_polar, ax_polar_lfe, ax_polar_norm]
    cartesian_axes = [ax_cartesian, ax_cartesian_lfe, ax_cartesian_norm]
    colormeshes = [timeMesh, lfeMesh, normMesh]
    labels = ["hours", "detections", "detections / hour"]

    for polar_axis, cartesian_axis, colormesh, label in zip(polar_axes, cartesian_axes, colormeshes, labels):
        polar_grid_ticks_r = np.linspace(0, max_r, r_bins+1)
        polar_grid_ticks_theta = np.linspace(0, 2*np.pi, theta_bins+1)

        polar_axis.set_xticklabels('')
        polar_axis.set_yticklabels('')

        cartesian_axis.set_xticks(np.concatenate((np.negative(np.flip(polar_grid_ticks_r)[0:-1]), polar_grid_ticks_r))[0::2])
        cartesian_axis.set_yticks(np.concatenate((np.negative(np.flip(polar_grid_ticks_r)[0:-1]), polar_grid_ticks_r))[0::2])

        cartesian_axis.set_xlabel("X$_{KSM}$ (R$_S$)")
        cartesian_axis.set_ylabel("Y$_{KSM}$ (R$_S$)")

        polar_axis.grid(True, linestyle="dotted")
        polar_axis.spines['polar'].set_visible(False)

        cartesian_axis.set_aspect('equal', adjustable='box')
        polar_axis.set_aspect('equal', adjustable='box')

        pos = cartesian_axis.get_position()
        colorbarAxis = fig.add_axes([pos.x0, pos.y0-pos.height*0.6, pos.width, pos.height], frameon=False, xticks=[], yticks=[])
        fig.colorbar(colormesh, ax=colorbarAxis, orientation="horizontal", location="bottom", label=label)

        
        # Add text for local time
        if local_time_labels is True:
            lt_label_distance = 75
            lt_label_positions = [el * np.pi/180 for el in np.arange(0, 360, 45)]
            lt_label_positions = lt_label_positions[4:] + lt_label_positions[:4]
            lt_labels = [str(el) for el in np.arange(0, 24, 3)]

            for pos, label in zip(lt_label_positions, lt_labels):
                polar_axis.text(pos, lt_label_distance, label, va="center", ha="center", color="grey")



    # divider1 = axes_grid1.make_axes_locatable(ax_cartesian)
    # divider2 = axes_grid1.make_axes_locatable(ax_polar)
    # cax1 = divider1.append_axes("right", size="3%", pad="2%")
    # cax2 = divider2.append_axes("right", size="3%", pad="2%")
    # cax2.axis("off")

    # fig.colorbar(pc, label="hours")

    plt.show()

def PlotLfeDistributions(trajectories_df, LFE_df, split_by_duration=True, r_hist_bins=np.linspace(0, 160, 160), lat_hist_bins=np.linspace(-20, 20, 40), lt_hist_bins=np.linspace(0, 24, 48), unet=True, scale="linear", long_lfe_cutoff=11, normalise_histograms=False):
    
    if normalise_histograms is True:
        alpha = 0.7
    else:
        alpha = 1

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    (r_axis, lat_axis, lt_axis) = axes
    # Define secondary axes for spacecraft time
    
    secondary_axes = [axis.twinx() for axis in axes]

    r_secondary_axis, lat_secondary_axis, lt_secondary_axis = secondary_axes    

    for axis in secondary_axes:
        axis.set_ylabel("Normalised Time Spent")
        axis.margins(0)
        axis.set_ylim([0, 1])

    x = trajectories_df["xpos_ksm"]
    y = trajectories_df["ypos_ksm"]
    z = trajectories_df["zpos_ksm"]

    spacecraft_r, spacecraft_theta, spacecraft_z = CartesiansToCylindrical(x, y, z)
    spacecraft_lat = []
    for r, z in zip(spacecraft_r, spacecraft_z):
        spacecraft_lat.append(np.tan(z/r))

    spacecraft_lt = []
    for longitude_rads in spacecraft_theta:
        longitude_degs = longitude_rads*180/np.pi
        spacecraft_lt.append(((longitude_degs+180)*24/360) % 24)

    # Define spacecraft location histograms
    r_frequency, r_hist = np.histogram(spacecraft_r, bins=r_hist_bins)
    r_hist_bin_centers = 0.5*(r_hist[1:]+ r_hist[:-1])
    
    lat_frequency, lat_hist = np.histogram(spacecraft_lat, bins=lat_hist_bins)
    lat_hist_bin_centers = 0.5*(lat_hist[1:]+ lat_hist[:-1])

    lt_frequency, lt_hist = np.histogram(spacecraft_lt, bins=lt_hist_bins)
    lt_hist_bin_centers = 0.5*(lt_hist[1:]+ lt_hist[:-1])

    r_secondary_axis.plot(r_hist_bin_centers, r_frequency/np.max(r_frequency), color="black", label="Spacecraft Time")
    lat_secondary_axis.plot(lat_hist_bin_centers, lat_frequency/np.max(lat_frequency), color="black", label="Spacecraft Time")
    lt_secondary_axis.plot(lt_hist_bin_centers, lt_frequency/np.max(lt_frequency), color="black", label="Spacecraft Time")


    lfe_x, lfe_y, lfe_z = (LFE_df["x_ksm"], LFE_df["y_ksm"], LFE_df["z_ksm"])
    lfe_r, lfe_theta, lfe_z = CartesiansToCylindrical(lfe_x, lfe_y, lfe_z)

    lfe_lat = []
    for r, z in zip(lfe_r, lfe_z):
        lfe_lat.append(np.tan(z/r))

    lfe_lt = []
    for longitude_rads in lfe_theta:
        longitude_degs = longitude_rads*180/np.pi
        lfe_lt.append(((longitude_degs+180)*24/360) % 24)


    if not split_by_duration:

        r_axis.hist(lfe_r, bins=r_hist_bins, color="indianred")

        lat_axis.hist(lfe_lat, bins=lat_hist_bins, color="indianred")

        lt_axis.hist(lfe_lt, bins=lt_hist_bins, color="indianred")

    else:
        # returns the indices
        short_LFEs = np.where(LFE_df["duration"] <= long_lfe_cutoff*60*60)
        long_LFEs = np.where(LFE_df["duration"] > long_lfe_cutoff*60*60) 

        r_axis.hist([lfe_r[i] for i in short_LFEs], bins=r_hist_bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)
        r_axis.hist([lfe_r[i] for i in long_LFEs], bins=r_hist_bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)

        lat_axis.hist([np.array(lfe_lat)[i] for i in short_LFEs], bins=lat_hist_bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)
        lat_axis.hist([np.array(lfe_lat)[i] for i in long_LFEs], bins=lat_hist_bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)

        lt_axis.hist([np.array(lfe_lt)[i] for i in short_LFEs], bins=lt_hist_bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)
        lt_axis.hist([np.array(lfe_lt)[i] for i in long_LFEs], bins=lt_hist_bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours", density=normalise_histograms, alpha=alpha)
        
        lt_axis.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)



    for ax in axes:
        if normalise_histograms is True:
            ax.set_ylabel("LFE Probability Density")
        else:
            ax.set_ylabel("LFE Count")
        ax.margins(0)
        ax.set_yscale(scale)

    r_axis.set_xlabel("Radial Distance (R$_S$)")

    lat_axis.set_xlabel("Latitude ($^\circ$)") # note latitude is measured with respect to the KSM frame i.e. tan(z_ksm/r)

    lt_axis.set_xlabel("Local Time")
    lt_axis.set_xticks(np.arange(0, 24+3, 3), minor=False)
    lt_axis.set_xticks(np.arange(0, 24+1, 1), minor=True)

    if unet:
        fig.suptitle("LFE Distributions (UNET Output)")
    else:
        fig.suptitle("LFE Distributions (Training Data)")

    fig.tight_layout()
    plt.show()



def CartesiansToCylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta, z)


if __name__ == "__main__":
    main()
