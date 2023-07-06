# -*- coding: utf-8 -*-
"""
Created on Friday June 23rd 2023

@author: Caitriona Jackman
"""
import matplotlib.ticker as mticker 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    ResidencePlots(trajectories, LFE_df, z_bounds=[-100, 100])
    


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


def ResidencePlots(trajectories_df, LFE_df, z_bounds, max_r=5, r_bins=20, theta_bins=40):
    times = trajectories_df["datetime_ut"]
    x = trajectories_df["xpos_ksm"]
    y = trajectories_df["ypos_ksm"]
    z = trajectories_df["zpos_ksm"]

    spacecraft_r, spacecraft_theta, spacecraft_z = CartesiansToCylindrical(x, y, z)
    
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
    for mesh_r in tqdm(range(len(mesh_outer_edges["r"])), desc="r"):
        for mesh_theta in tqdm(range(len(mesh_outer_edges["theta"])), desc="theta", leave=False):
            # Determines at what time indices is the spacecraft within the current bin
            time_indices_in_region = np.where((spacecraft_r <= mesh_outer_edges["r"][mesh_r]) & (spacecraft_r > mesh_inner_edges["r"][mesh_r]) &\
                         (spacecraft_theta <= mesh_outer_edges["theta"][mesh_theta]) & (spacecraft_theta > mesh_inner_edges["theta"][mesh_theta]) &\
                         (spacecraft_z <= np.max(z_bounds)) & (spacecraft_z > np.min(z_bounds)))[0]

            # Get the time spent in the current bin in minutes
            timeInRegion = len(time_indices_in_region)
            timeSpentInBin[mesh_r][mesh_theta] = timeInRegion

    # Shading flat requires removing last point
    timeSpentInBin = timeSpentInBin[:-1,:-1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    pc = ax.pcolormesh(mesh_inner_edges["theta"], mesh_inner_edges["r"], timeSpentInBin/60, shading="flat")

    fig.colorbar(pc, label="hours")

    ax.set_xlabel("Theta ($^\circ$)")
    ax.set_title("Residence Time")

    rlabels = ax.get_ymajorticklabels()
    for label in rlabels:
        label.set_color('lightgrey')

    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position+10),ax.get_rmax()/2.,'R (R$_S$)', rotation=label_position,ha='center',va='center', color="lightgrey")
    
    plt.show()



def CartesiansToCylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta, z)


if __name__ == "__main__":
    main()
