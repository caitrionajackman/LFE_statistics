# -*- coding: utf-8 -*-
"""
Created on Friday June 23rd 2023

@author: Caitriona Jackman
"""
import matplotlib.ticker as mticker 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    LFE_df = pd.read_csv(input_data_fp + '/lfe_timestamps.csv',parse_dates=['start','end'])
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

    start_times, end_times = (LFE_df["start"], LFE_df["end"])

    mesh_outer_edges={
        "r": (np.arange(1, r_bins+1)*max_r)/r_bins, # ranging from 2 + bin size to max_r
        "theta": (np.arange(1, theta_bins+1)*np.pi)/(theta_bins/2) - np.pi # ranging from -pi + binsize to pi
    }
    mesh_inner_edges={
        "r": (np.arange(0, r_bins+1)*max_r)/r_bins, # ranging from 1 to max_r - bin size
        "theta": (np.arange(0, theta_bins+1)*np.pi)/(theta_bins/2) - np.pi # ranging from -pi to pi - bin size
    }

    print("Determining LFE locations")

    """
    closest_times = []
    for lfe_time in tqdm(start_times, total=len(start_times)):
        differences = np.subtract(spacecraft_times, lfe_time)
        closet_time = np.min(abs(differences))
        closest_times.append(closet_time)
        
    print(closest_times)

    return
    """

    print("Comparing spacecraft location to bins")
    timeSpentInBin = np.zeros((r_bins+1, theta_bins+1))
    for mesh_r in tqdm(range(len(mesh_outer_edges["r"])), desc="r"):
        for mesh_theta in tqdm(range(len(mesh_outer_edges["theta"])), desc="theta", leave=False):
            # Determines at what time indices is the spacecraft within the current bin
            time_indices_in_region = np.where((spacecraft_r <= mesh_outer_edges["r"][mesh_r]) & (spacecraft_r > mesh_inner_edges["r"][mesh_r]) &\
                         (spacecraft_theta <= mesh_outer_edges["theta"][mesh_theta]) & (spacecraft_theta > mesh_inner_edges["theta"][mesh_theta]) &\
                         (spacecraft_z <= np.max(z_bounds)) & (spacecraft_z > np.min(z_bounds)))[0]


            # Get the time spent in the current bin in minutes
            timeInRegion = len(time_indices_in_region)
            if timeInRegion == 0: timeInRegion = np.nan
            timeSpentInBin[mesh_r][mesh_theta] = timeInRegion

    # Shading flat requires removing last point
    timeSpentInBin = timeSpentInBin[:-1,:-1]

    fig = plt.figure()

    # setting the axis limits in [left, bottom, width, height]
    cartesian_rectangle = [0.2, 0.2, 0.6, 0.6]
    polar_rectangle = cartesian_rectangle # [0.3, 0.3, 0.4, 0.4]

    ax_cartesian = fig.add_subplot(cartesian_rectangle, zorder=10)
    ax_cartesian.patch.set_alpha(0)
    ax_polar = fig.add_axes(polar_rectangle, polar=True, zorder=-10)

    pc = ax_polar.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], timeSpentInBin/60, shading="flat")

    # leftRect = patches.Rectangle((cartesian_rectangle[0]-0.2, cartesian_rectangle[1]-0.2), 0.2, 0.7, linewidth=1, fill=True, facecolor='white', zorder=0, transform=fig.transFigure, figure=fig)
    # bottomRect = patches.Rectangle((cartesian_rectangle[0]-0.2, cartesian_rectangle[1]-0.2), 0.7, 0.2, linewidth=1, fill=True, facecolor='white', zorder=0, transform=fig.transFigure, figure=fig)
    # topRect = patches.Rectangle((cartesian_rectangle[0]-0.2, cartesian_rectangle[1]+cartesian_rectangle[3]), 0.7, 0.2, linewidth=1, fill=True, facecolor='white', zorder=0, transform=fig.transFigure, figure=fig)
    # rightRect = patches.Rectangle((cartesian_rectangle[0]+cartesian_rectangle[2], cartesian_rectangle[1]-0.2), 0.2, 0.7, linewidth=1, fill=True, facecolor='white', zorder=0, transform=fig.transFigure, figure=fig)
    # fig.patches.extend([leftRect, bottomRect, topRect, rightRect])

    polar_axes = [ax_polar]
    cartesian_axes = [ax_cartesian]

    for polar_axis, cartesian_axis in zip(polar_axes, cartesian_axes):
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

        cartesian_axis.set_title(f"Residence Time\n{z_bounds[0]} " + "(R$_S$) < Z$_{KSM}$ <" + f" {z_bounds[1]} (R$_S$)")


    # divider1 = axes_grid1.make_axes_locatable(ax_cartesian)
    # divider2 = axes_grid1.make_axes_locatable(ax_polar)
    # cax1 = divider1.append_axes("right", size="3%", pad="2%")
    # cax2 = divider2.append_axes("right", size="3%", pad="2%")
    # cax2.axis("off")

    # fig.colorbar(pc, label="hours", cax=cax1)
    plt.show()


def CartesiansToCylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta, z)


if __name__ == "__main__":
    main()
