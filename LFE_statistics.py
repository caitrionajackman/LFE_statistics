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
    #SORT CONFIG FILE LATER
    #config = configparser.ConfigParser()
    #config.read('configurations.ini')
    #input_data_fp = config['filepaths']['input_data']
    #output_data_fp= config['filepaths']['output_data']

    # input_data_fp='C:/Users/Local Admin/Documents/Data'
    input_data_fp = "./../data/"

    #Read in LFE list (output of Elizabeth's U-Net run on full Cassini dataset)
    print('First step is to read in the LFE list')
    LFE_df = pd.read_csv(input_data_fp + '/lfe_detections.csv',parse_dates=['start','end'])
    #print(LFE_df) 
    #start	end	label
    #24/07/2004  20:41:27    25/07/2004  02:13:52    LFE

    LFE_duration=LFE_df['end']-LFE_df['start']  #want this in minutes and to be smart about day/year boundaries
    #print(LFE_duration[0:2])


    LFE_secs=[]
    for i in range(np.array(LFE_duration).size):
        LFE_secs.append(LFE_duration[i].total_seconds())


    # PlotDurationHistogram(LFE_secs)

    #Next want to explore some manual inspection of the longest LFEs to see if they're "real"
    # InspectLongestLFEs(LFE_df, LFE_secs, LFE_duration)

    print("Next step is to plot residence time")
    #Make a function to plot residence time (See Charlie example code for Mercury)
    #This needs to read in Cassini trajectory data first - from Elizabeth other code
    #Then take in the LFE list and plot them over each other
    print("Loading trajectories...")
    trajectories = pd.read_csv(input_data_fp + "cassini_output/trajectorytotal.csv", parse_dates=["datetime_ut"])
    ResidencePlots(trajectories, LFE_df, z_bounds=[-30, 30])
    


def PlotDurationHistogram(LFE_secs):
    fig, ax = plt.subplots(1, tight_layout=True, sharey = True)
    ax.hist(np.array(LFE_secs)/(60.*24.),bins=np.linspace(0,250,126))
    ax.set_title('Histogram of duration of LFEs across Cassini mission')
    ax.set_xlabel('LFE duration (hours)')
    ax.set_ylabel('# of LFEs')
    ax.set_xscale('log')

    #TO ADD legend to include: N of LFEs, mean or median of distribution
    print(np.median(np.array(LFE_secs)/(60.*24.)))
    print(np.mean(np.array(LFE_secs)/(60.*24.)))
    #perhaps overplot these as vertical lines

    plt.show()


def InspectLongestLFEs(LFE_df, LFE_secs, LFE_duration):
    LFE_df['LFE_duration']=LFE_duration
    LFE_df['LFE_secs']=LFE_secs

    LFE_df_sorted=LFE_df.sort_values(by=['LFE_secs'],ascending=False)
    print(LFE_df_sorted)

    #Want to be able to look at these spectrograms to see if any need to be removed as outliers/unphysical


def ResidencePlots(trajectories_df, LFE_df, z_bounds, max_r=80, r_bin_size=10, theta_bins=24):

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

    fig = plt.figure()

    # setting the axis limits in [left, bottom, width, height]
    # cartesian_rectangle = [0.2, 0.2, 0.6, 0.6]
    # polar_rectangle = cartesian_rectangle # [0.3, 0.3, 0.4, 0.4]

    ax_cartesian = fig.add_subplot(1, 3, 1, zorder=10)
    ax_cartesian.patch.set_alpha(0)
    ax_polar = fig.add_axes(ax_cartesian.get_position(), polar=True, zorder=-10)
    
    ax_cartesian.set_title(f"Residence Time\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")

    ax_cartesian_lfe = fig.add_subplot(1, 3, 2, zorder=10)
    ax_cartesian_lfe.patch.set_alpha(0)
    ax_polar_lfe = fig.add_axes(ax_cartesian_lfe.get_position(), polar=True, zorder=-10)
    
    ax_cartesian_lfe.set_title(f"LFE Detections\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")

    ax_cartesian_norm = fig.add_subplot(1, 3, 3, zorder=10)
    ax_cartesian_norm.patch.set_alpha(0)
    ax_polar_norm = fig.add_axes(ax_cartesian_norm.get_position(), polar=True, zorder=-10)
    
    ax_cartesian_norm.set_title(f"Detections normalised by Residence Time\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")

    
    timeMesh = ax_polar.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], timeSpentInBin/60, cmap="viridis", shading="flat")

    lfeMesh = ax_polar_lfe.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], lfe_detections_in_bin, cmap="magma", shading="flat", zorder=-1)
    ax_polar_lfe.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], timeSpentInBin, cmap=ListedColormap(["lightgrey"]), shading="flat", zorder=-2)

    normMesh = ax_polar_norm.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], norm_detections_in_bin, cmap="RdPu_r", shading="flat", zorder=-1)
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

        cartesian_axis.set_xticks(np.concatenate((np.negative(np.flip(polar_grid_ticks_r)[0:-1]), polar_grid_ticks_r)))
        cartesian_axis.set_yticks(np.concatenate((np.negative(np.flip(polar_grid_ticks_r)[0:-1]), polar_grid_ticks_r)))

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


def CartesiansToCylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta, z)


if __name__ == "__main__":
    main()
