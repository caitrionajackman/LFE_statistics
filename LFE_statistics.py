# -*- coding: utf-8 -*-
"""
Created on Friday June 23rd 2023

@author: Caitriona Jackman
"""
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

def main():
    plt.rcParams.update({'font.size': 12})

    unet=True
    data_directory = "./../data/"
    lfe_unet_data = "/lfe_detections_unet.csv" # Processed using findDetectionPositions.py
    lfe_training_data = "/lfe_detections_training.csv" # "
    trajectories_file = "cassini_output/trajectorytotal.csv" # Output from Beth's "Cassini_Plotting" repo

    lfe_duration_split = 11 # measured in hours

    plot = {
        "duration_histograms": False,
        "inspect_longest_lfes": False,
        "residence_time_multiplots": True,
        "lfe_distributions": False
    }

    #Read in LFE list (output of Elizabeth's U-Net run on full Cassini dataset)
    print('Reading in the LFE list')
    
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
        ResidencePlots(trajectories, LFE_df, z_bounds=[-30, 30], unet=unet, saturation_factor=1.5)
    
    if plot["lfe_distributions"]:       
        PlotLfeDistributions(trajectories, LFE_df, unet=unet, scale="linear", long_lfe_cutoff=lfe_duration_split)



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


def ResidencePlots(trajectories_df, LFE_df, z_bounds, max_r=80, r_bin_size=10, theta_bins=24, unet=True, saturation_factor=1):

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

    fig = plt.figure(figsize=(16, 10))


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



    # divider1 = axes_grid1.make_axes_locatable(ax_cartesian)
    # divider2 = axes_grid1.make_axes_locatable(ax_polar)
    # cax1 = divider1.append_axes("right", size="3%", pad="2%")
    # cax2 = divider2.append_axes("right", size="3%", pad="2%")
    # cax2.axis("off")

    # fig.colorbar(pc, label="hours")
    plt.show()

def PlotLfeDistributions(trajectories_df, LFE_df, split_by_duration=True, r_hist_bins=np.linspace(0, 160, 160), lat_hist_bins=np.linspace(-20, 20, 40), lt_hist_bins=np.linspace(0, 24, 48), unet=True, scale="linear", long_lfe_cutoff=11):
    
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

        r_axis.hist([lfe_r[i] for i in short_LFEs], bins=r_hist_bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours")
        r_axis.hist([lfe_r[i] for i in long_LFEs], bins=r_hist_bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours")

        lat_axis.hist([np.array(lfe_lat)[i] for i in short_LFEs], bins=lat_hist_bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours")
        lat_axis.hist([np.array(lfe_lat)[i] for i in long_LFEs], bins=lat_hist_bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours")

        lt_axis.hist([np.array(lfe_lt)[i] for i in short_LFEs], bins=lt_hist_bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours")
        lt_axis.hist([np.array(lfe_lt)[i] for i in long_LFEs], bins=lt_hist_bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours")
        
        lt_axis.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)



    for ax in axes:
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
