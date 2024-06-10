# -*- coding: utf-8 -*-
"""
Created on Friday June 23rd 2023 2024

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
from scipy.io import readsav

def main():
    plt.rcParams.update({'font.size': 12})

    unet=True   #use the joined list for analysis
    data_directory = "C:/Users/Local Admin/Documents/Collaborations/Jackman_LFEs/"
    #lfe_unet_data = "lfe_detections_unet.csv" # Processed using findDetectionPositions.py
    lfe_unet_data = "lfe_detections_unet_2874.csv"#Processed using findDetectionPositions.py (Elizabeth's updated UNet output file)
    
    lfe_training_data = "lfe_detections_training.csv" # "
    #lfe_training_data = "test_times.csv" # " #placeholder file to check specific times for PPO phases (23/01/2024)
   
    lfe_joined_list = "LFEs_joined.csv"
    
    trajectories_file = "cassini_output/trajectorytotal.csv" # Output from Beth's "Cassini_Plotting" repo
    ppo_file = "mag_phases_2004_2017_final.sav"
    LFE_phase_df = "lfe_with_phase.csv" # File saved by SavePPO()

    lfe_duration_split = 11 # measured in hours #this may change to become the median of the duration distribution (or some other physically meaningful number)

    plot = {
        "delta_t": True,
        "Join_LFEs": False,   #only need to run this one time to generate LFEs_joined.csv
        "duration_histograms": True,
        "inspect_longest_lfes": False,
        #TODO: Add function here to plot spectrogram (call radio data), and overplot polygons (call Unet json file)
        "residence_time_multiplots": False,
        "lfe_distributions": False,
        "ppo_save": False,  #this takes 4-5 minutes and produces LFE_phase_df
        "ppo_plot": False,
        "check_PPOs": False,
        "local_ppo_plot": False
    }


    #the deltaT examination should happen first, then joining, then all other operations on the joined list


    if plot["delta_t"]:
        print("in deltat function now")
        if unet is False:
            print("you are trying to join the training data")
            LFE_df = pd.read_csv(data_directory + lfe_training_data, parse_dates=['start','end'])
        if unet is True:
            print("running deltaT visualisation on raw unet 2874 data")
            LFE_df = pd.read_csv(data_directory + lfe_unet_data, parse_dates=['start','end'])
            
        LFE_duration=LFE_df['end']-LFE_df['start']  #want this in minutes and to be smart about day/year boundaries
        LFE_secs=[]
        #print(LFE_secs)
        Delta_t_LFEs(LFE_df, LFE_secs, LFE_duration, unet=unet)
        #print("see if return works from deltaT function")
        #time_diff_df=time_diff_df
        #time_diff_minutes=time_diff_minutes
        #print(np.min(np.array(time_diff_minutes)))
        
        
    if plot["Join_LFEs"]:
        if unet is False:
            print("you are trying to join the training data")
            LFE_df = pd.read_csv(data_directory + lfe_training_data, parse_dates=['start','end'])
        if unet is True:
            print("running on raw unet 2874 data")
            LFE_df = pd.read_csv(data_directory + lfe_unet_data, parse_dates=['start','end'])
            
        LFE_duration=LFE_df['end']-LFE_df['start']  #want this in minutes and to be smart about day/year boundaries
        LFE_secs=[]
        for i in range(np.array(LFE_duration).size):
            LFE_secs.append(LFE_duration[i].total_seconds())
        LFE_joiner(data_directory,LFE_df, LFE_secs, unet=unet)    




    #Now for all further operations, work off joined list.
    print('Reading in the LFE list')
    #Unet True means the code uses the output of O'Dwyer Unet (4950 examples) 28/07/2023 - before post-processing. FOLLOWING THE JOINING
    #Unet False means the code uses the training data (984 examples)
    if unet is True: #use the joined list here (and have separate read in of the unet in the joiner code)
    #LFE_df = pd.read_csv(data_directory + lfe_unet_data, parse_dates=['start','end'])
        LFE_df = pd.read_csv(data_directory + lfe_joined_list, parse_dates=['start','end'])

    else:        
        LFE_df = pd.read_csv(data_directory + lfe_training_data, parse_dates=['start','end'])

    LFE_duration=LFE_df['end']-LFE_df['start']  #want this in minutes and to be smart about day/year boundaries
    LFE_secs=[]
    
    for i in range(np.array(LFE_duration).size):
        LFE_secs.append(LFE_duration[i].total_seconds())

   #print('ARTIFICIALLY REMOVING THE LFES WITH NO TRAJECTORY DATA')
   #LFE_df.drop(index=np.linspace(0,500,501),inplace=True)
   #LFE_df.drop(index=4440,inplace=True)
   #LFE_df.reset_index(drop=True,inplace=True)


    if plot["duration_histograms"]:
        print("running the duration histogram on the joined events")
        #LFE_df = pd.read_csv(data_directory + lfe_joined_list, parse_dates=['start','end'])
        #LFE_duration=LFE_df['end']-LFE_df['start']  #want this in minutes and to be smart about day/year boundaries
        #LFE_secs=[]
        PlotDurationHistogram(LFE_secs)

        
    if plot["inspect_longest_lfes"]:
        #Next want to explore some manual inspection of the longest LFEs to see if they're "real"
        InspectLongestLFEs(LFE_df, LFE_secs, LFE_duration)

    if plot["residence_time_multiplots"] or plot["lfe_distributions"]:
        print("Loading trajectories...")
        trajectories = pd.read_csv(data_directory + trajectories_file, parse_dates=["datetime_ut"])

    if plot["residence_time_multiplots"]:
        ResidencePlots(trajectories, LFE_df, z_bounds=[-30, 30], unet=unet, saturation_factor=1)
        ResidencePolar(trajectories, LFE_df)
        return trajectories,LFE_df
        #TODO: check how saturation factor works (lower priority until after the post-processing is done)
        #TODO: Make quick sorting function to manually examine LFEs in particular LT sectors. Sort the LFEs by hrs of LT        
        #TODO ditto for latitude    
    
    if plot["lfe_distributions"]:       
        #PlotLfeDistributions(trajectories, LFE_df, unet=unet, scale="linear", long_lfe_cutoff=lfe_duration_split)
        PlotLfeDistributions1(trajectories, LFE_df, unet=unet, scale="linear", long_lfe_cutoff=lfe_duration_split)
        
    if plot["ppo_save"]:
       # data=[[pd.Timestamp('2015-09-30T15'),pd.Timestamp('2015-09-30T20')],[pd.Timestamp('2015-10-01T15'),pd.Timestamp('2015-10-01T18')]]
        #LFE_df=pd.DataFrame(data,columns=['start','end'])
        #LFE_df=pd.DataFrame(data)
       # SavePPO(data_directory + ppo_file, LFE_df, data_directory, "lfe_with_phaseTEST.csv")
       SavePPO(data_directory + ppo_file, LFE_df, data_directory, "lfe_with_phase.csv")
       
    if plot["check_PPOs"]:
        PPOphasecheck(data_directory + ppo_file, data_directory)

    bin_width_PPO=15
    if plot["ppo_plot"]:
        PlotPPO(data_directory + LFE_phase_df, bin_width_PPO, LFE_df, long_lfe_cutoff=lfe_duration_split, local=False)
        #PlotPPO(data_directory + LFE_phase_df, 15, LFE_df, long_lfe_cutoff=lfe_duration_split, local=False)
    if plot["local_ppo_plot"]:
        PlotPPO(data_directory + LFE_phase_df, bin_width_PPO, LFE_df, long_lfe_cutoff=lfe_duration_split, local=True)

def LFE_joiner(data_directory,LFE_df,LFE_secs,unet=True):
    #print("want to examine dt between successive LFEs and join those which are short")
    time_diff_df=pd.DataFrame({'st':LFE_df['start'][1:].values, 'en':LFE_df['end'][:-1].values})
    time_diff_minutes=time_diff_df.st-time_diff_df.en
    time_diff_minutes = [time_diff_minute.total_seconds()/60. for time_diff_minute in time_diff_minutes]
    #print("checking time difference")
    
    starts_joined=[]
    ends_joined=[]
    x_ksm_joined=[]
    y_ksm_joined=[]
    z_ksm_joined=[]
    label_joined=[]
    tdm=np.array(time_diff_minutes)
    starts_joined.append(LFE_df['start'][0])    #manually fill the first LFE (short gap after)
    ends_joined.append(LFE_df['end'][0])    #manually fill the first LFE (short gap after)
    x_ksm_joined.append(LFE_df['x_ksm'][0]) #manually fill the first LFE (short gap after)
    y_ksm_joined.append(LFE_df['y_ksm'][0]) #manually fill the first LFE (short gap after)
    z_ksm_joined.append(LFE_df['z_ksm'][0]) #manually fill the first LFE (short gap after)
    label_joined.append(LFE_df['label'][0]) #manually fill the first LFE (short gap after)
    
    iter_skip=True
    for i in range(tdm.size):
        if iter_skip:
            #print("iteration skipped")
            iter_skip=False
            continue
        #print("iteration number:")
        #print(i)
        if tdm[i] >10:  #10 minute cutoff for between events  
            #just keep starts and end as in original list
            starts_joined.append(LFE_df['start'][i])
            ends_joined.append(LFE_df['end'][i])#
            x_ksm_joined.append(LFE_df['x_ksm'][i])
            y_ksm_joined.append(LFE_df['y_ksm'][i])
            z_ksm_joined.append(LFE_df['z_ksm'][i])
            label_joined.append(LFE_df['label'][i])
            #print("long gap and no joining")    
            #print(i)
        else:
            #keep start as original list and revised end as next entry
            print("short gap and joining")
            print(LFE_df['start'][i])
            starts_joined.append(LFE_df['start'][i])    
            ends_joined.append(LFE_df['end'][i+1])
            x_ksm_joined.append(LFE_df['x_ksm'][i])
            y_ksm_joined.append(LFE_df['y_ksm'][i])
            z_ksm_joined.append(LFE_df['z_ksm'][i])
            label_joined.append(LFE_df['label'][i])
            iter_skip=True   #want it to skip an iteration
            
        
    #now make a new dataframe with the joined LFEs
    LFE_df_joined=pd.DataFrame({'start':starts_joined,'end':ends_joined,'x_ksm':x_ksm_joined,'y_ksm':y_ksm_joined,'z_ksm':z_ksm_joined,'label':label_joined})  
    LFE_duration_joined=LFE_df_joined['end']-LFE_df_joined['start']  #want this in minutes and to be smart about day/year boundaries
  
    LFE_secs_joined=[]
    for i in range(np.array(LFE_duration_joined).size):
        LFE_secs_joined.append(LFE_duration_joined[i].total_seconds())

    #Add the duration to the new datafram
    LFE_df_joined['duration']=LFE_secs_joined
    #breakpoint()
    file_name='LFEs_joined.csv'
    print(f"Saving new csv file to {data_directory+file_name}")
    LFE_df_joined.to_csv(data_directory + file_name)
    
    print("printing the n elements in new LFE joined")
    print(LFE_df_joined)

    return LFE_df_joined,LFE_secs_joined
    
     

def PlotPPO(file_path, bin_width, LFE_df, long_lfe_cutoff, unet=True, local=False):

    bins=np.arange(0, 360+bin_width, bin_width)
    data = pd.read_csv(file_path)

    #TODO: Check with Gabs - is this how to start treatment of PPOs at start?
    north = np.array(data["north phase"]) % 360
    south = np.array(data["south phase"]) % 360

    #differentiate between local phases and "global" phases - and both require similar data
    if local is True:
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


        local_phase_north = []
        local_phase_south = []
        for north_phase, south_phase, az in zip(north, south, azimuth):
            local_phase_north.append(((north_phase - az) + 720) % 360)
            local_phase_south.append(((south_phase - az) + 720) % 360)

        local_phase_north = np.array(local_phase_north)
        local_phase_south = np.array(local_phase_south)

    #long_lfe_cutoff is set to lfe_duration_split, which for now (January 2024) is set to 11 hours.
    short_LFEs, = np.where(LFE_df["duration"] <= long_lfe_cutoff*60*60)
    #print("Short LFEs size")
    #print(short_LFEs.size)
    
    long_LFEs, = np.where(LFE_df["duration"] > long_lfe_cutoff*60*60)
    #print("Long LFEs size")
    #print(long_LFEs.size)


    if local is False:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        ax_north, ax_south = axes

        
        n_north_short,bin_edges=np.histogram([north[i] for i in short_LFEs], bins=bins)
        ax_north.plot(bins[:-1]+bin_width/2.,(n_north_short/short_LFEs.size)*100.,color="indianred",linewidth=6)
        
        n_north_long,bin_edges=np.histogram([north[i] for i in long_LFEs], bins=bins)
        ax_north.plot(bins[:-1]+bin_width/2.,(n_north_long/long_LFEs.size)*100.,color="mediumturquoise",linewidth=6)
        
       
        n_south_short,bin_edges=np.histogram([south[i] for i in short_LFEs], bins=bins)
        ax_south.plot(bins[:-1]+bin_width/2.,(n_south_short/short_LFEs.size)*100.,color="indianred",linewidth=6)

        n_south_long,bin_edges=np.histogram([south[i] for i in long_LFEs], bins=bins)
        ax_south.plot(bins[:-1]+bin_width/2.,(n_south_long/long_LFEs.size)*100.,color="mediumturquoise",linewidth=6)

        ax_north.set_title("North Phase")     
        ax_south.set_title("South Phase")
        ax_north.set_ylim(bottom=0)
        ax_south.set_ylim(bottom=0)
        ax_south.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)

        titleTag = "PPO"

    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        ax_local_north, ax_local_south = axes

       # ax_local_north.hist([local_phase_north[i] for i in short_LFEs], bins=bins, color="indianred",density=True)
       # ax_local_north.hist([local_phase_north[i] for i in long_LFEs], bins=bins, color="mediumturquoise",density=True,histtype='step',linewidth=6)

        #ax_local_south.hist([local_phase_south[i] for i in short_LFEs], bins=bins, color="indianred", label=f"duration < {long_lfe_cutoff} hours")
        #ax_local_south.hist([local_phase_south[i] for i in long_LFEs], bins=bins, color="mediumturquoise", label=f"duration > {long_lfe_cutoff} hours",histtype='step',linewidth=6)

        n_north_short,bin_edges=np.histogram([north[i] for i in short_LFEs], bins=bins)
        ax_local_north.plot(bins[:-1]+bin_width/2.,(n_north_short/short_LFEs.size)*100.,color="indianred",linewidth=6)
    
        n_north_long,bin_edges=np.histogram([north[i] for i in long_LFEs], bins=bins)
        ax_local_north.plot(bins[:-1]+bin_width/2.,(n_north_long/long_LFEs.size)*100.,color="mediumturquoise",linewidth=6)
    
    
        n_south_short,bin_edges=np.histogram([south[i] for i in short_LFEs], bins=bins)
        ax_local_south.plot(bins[:-1]+bin_width/2.,(n_south_short/short_LFEs.size)*100.,color="indianred",linewidth=6)
    
        n_south_long,bin_edges=np.histogram([south[i] for i in long_LFEs], bins=bins)
        ax_local_south.plot(bins[:-1]+bin_width/2.,(n_south_long/long_LFEs.size)*100.,color="mediumturquoise",linewidth=6)
    
        ax_local_north.set_title("North Local Phase")
        ax_local_south.set_title("South Local Phase")

        ax_local_south.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)

        titleTag = "Local"

    for ax in fig.get_axes():
        ax.set_ylabel("# of LFEs")
        ax.set_xlabel("Phase ($^\circ$)")
        ax.margins(x=0)
        ax.set_xticks(bins[0::2])

    if unet:
        dataTag="UNET Output"
    else:
        datTag="Training Data"
        
    fig.suptitle(f"Northern and Southern {titleTag} Phases ({dataTag})")

    plt.tight_layout()
    plt.show()



    #Now split by hemisphere
    #calculation of LFE range, latitude, local time
    lfe_x, lfe_y, lfe_z = (LFE_df["x_ksm"], LFE_df["y_ksm"], LFE_df["z_ksm"])
    LFE_r=np.sqrt(lfe_x**2 + lfe_y**2 + lfe_z**2)   #range in RS
    theta = np.arctan2(lfe_y, lfe_x)
    LFE_lat=np.tan(lfe_z/LFE_r)
    LFE_lat_deg=LFE_lat*(180./np.pi)
    #theta = np.arctan2(y, x)
    #not sure what longitude_rads is? replace with theta
    longitude_degs = theta*180/np.pi
    LFE_lt=((longitude_degs+180)*24/360) % 24

    #Check positive and negative latitudes        
    NorthHem, = np.where(LFE_lat_deg > 0)
    SouthHem, = np.where(LFE_lat_deg < 0)
    print('Southern hemisphere LFEs:')
    print(SouthHem.size)
    print('Northern hemisphere LFEs:')
    print(NorthHem.size)


    if local is False:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        ax_north, ax_south = axes
       
        n_north_SouthHem,bin_edges=np.histogram([north[i] for i in SouthHem], bins=bins)
        ax_north.plot(bins[:-1]+bin_width/2.,(n_north_SouthHem/SouthHem.size)*100.,color="indianred",linewidth=6)
       
        n_north_NorthHem,bin_edges=np.histogram([north[i] for i in NorthHem], bins=bins)
        ax_north.plot(bins[:-1]+bin_width/2.,(n_north_NorthHem/NorthHem.size)*100.,color="mediumturquoise",linewidth=6)
         
        n_south_SouthHem,bin_edges=np.histogram([south[i] for i in SouthHem], bins=bins)
        ax_south.plot(bins[:-1]+bin_width/2.,(n_south_SouthHem/SouthHem.size)*100.,color="indianred",linewidth=6)

        n_south_NorthHem,bin_edges=np.histogram([south[i] for i in NorthHem], bins=bins)
        ax_south.plot(bins[:-1]+bin_width/2.,(n_south_NorthHem/NorthHem.size)*100.,color="mediumturquoise",linewidth=6)

        ax_north.set_title("North Phase. Red=SouthHem, Blue=NorthHem")     
        ax_south.set_title("South Phase. Red=SouthHem, Blue=NorthHem")
        ax_north.set_ylim(bottom=0)
        ax_south.set_ylim(bottom=0)
        ax_south.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)

        titleTag = "PPO"

    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        ax_local_north, ax_local_south = axes

        n_north_SouthHem,bin_edges=np.histogram([north[i] for i in SouthHem], bins=bins)
        ax_local_north.plot(bins[:-1]+bin_width/2.,(n_north_SouthHem/SouthHem.size)*100.,color="indianred",linewidth=6)
   
        n_north_NorthHem,bin_edges=np.histogram([north[i] for i in NorthHem], bins=bins)
        ax_local_north.plot(bins[:-1]+bin_width/2.,(n_north_NorthHem/NorthHem.size)*100.,color="mediumturquoise",linewidth=6)
    
        n_south_SouthHem,bin_edges=np.histogram([south[i] for i in SouthHem], bins=bins)
        ax_local_south.plot(bins[:-1]+bin_width/2.,(n_south_SouthHem/SouthHem.size)*100.,color="indianred",linewidth=6)
        
        n_south_NorthHem,bin_edges=np.histogram([south[i] for i in NorthHem], bins=bins)
        ax_local_south.plot(bins[:-1]+bin_width/2.,(n_south_NorthHem/NorthHem.size)*100.,color="mediumturquoise",linewidth=6)
   
        ax_local_north.set_title("North Local Phase. Red=SouthHem, Blue=NorthHem")
        ax_local_south.set_title("South Local Phase. Red=SouthHem, Blue=NorthHem")

        ax_local_south.legend(bbox_to_anchor=(0.5, -0.5), loc="center", ncol=2)

        titleTag = "Local"

    for ax in fig.get_axes():
        ax.set_ylabel("# of LFEs")
        ax.set_xlabel("Phase ($^\circ$)")
        ax.margins(x=0)
        ax.set_xticks(bins[0::2])

    if unet:
        dataTag="UNET Output"
    else:
        datTag="Training Data"
       
    fig.suptitle(f"Northern and Southern {titleTag} Phases ({dataTag})")

    plt.tight_layout()
    plt.show()




def PPOphasecheck(file_path, data_directory):
    #calculates PPO phases for individual dates/times
    print("Finding LFE Phase for a single example point")

    #print(f"Loading {file_path}")
    ppo_df = readsav(file_path)

    south_time = ppo_df["south_model_time"] # minutes since 2004-01-01 00:00:00
    south_phase = ppo_df["south_mag_phase"]

    north_time = ppo_df["north_model_time"]
    north_phase = ppo_df["north_mag_phase"]

    doy2004_0 = pd.Timestamp(2004, 1, 1)
    lfe_start_time=pd.Timestamp(2015,9,30,15)
        
    lfe_start_doy2004 = (lfe_start_time - doy2004_0).total_seconds() / 60 / 60 / 24 # days since 2004-01-01 00:00:00
        # Find minimum time difference
    south_index = (np.abs(south_time - lfe_start_doy2004)).argmin()
    north_index = (np.abs(north_time - lfe_start_doy2004)).argmin()
    
    south_phase_LFE_time=(south_phase[south_index])%360
    north_phase_LFE_time=(north_phase[north_index])%360
    
    print(south_phase_LFE_time)
    print(north_phase_LFE_time)
    
    #now need the x,y,z position for that given time                   
    # x = LFE_df["x_ksm"]
    # y = LFE_df["y_ksm"]
    # z = LFE_df["z_ksm"]

    # spacecraft_r, spacecraft_theta, spacecraft_z = CartesiansToCylindrical(x, y, z)

    # # Calculate local time
    # spacecraft_lt = []
    # for longitude_rads in spacecraft_theta:
    #     longitude_degs = longitude_rads*180/np.pi
    #     spacecraft_lt.append(((longitude_degs+180)*24/360) % 24)

    # azimuth = []
    # for lt in spacecraft_lt:
    #     azimuth.append(((lt-12) * 15 + 720) % 360)


    # local_phase_north = []
    # local_phase_south = []
    # for north_phase, south_phase, az in zip(north, south, azimuth):
    #     local_phase_north.append(((north_phase - az) + 720) % 360)
    #     local_phase_south.append(((south_phase - az) + 720) % 360)

    # local_phase_north = np.array(local_phase_north)
    # local_phase_south = np.array(local_phase_south)



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
                       

def PlotDurationHistogram(LFE_secs):
    fig, ax = plt.subplots(1, tight_layout=True, sharey = True, figsize=(8,8))
#    ax.hist(np.array(LFE_secs)/(60.*24.),bins=np.linspace(0,250,126), label=f"N = {len(LFE_secs)}")
    ax.hist(np.array(LFE_secs)/(60.*60.),bins=np.linspace(0,250,126), label=f"N = {len(LFE_secs)}")
    #print(LFE_secs)

    #if unet:
    ax.set_title('Histogram of duration of LFEs across Cassini mission (Joined List)')
    #else:
    #    ax.set_title('Histogram of duration of LFEs across Cassini mission (UNET false)')

    ax.set_xlabel('LFE duration (hours)')
    ax.set_ylabel('# of LFEs')
    ax.set_xscale('log')
    ax.set_yscale('log')

    median = np.median(np.array(LFE_secs)/(60.*60.))    #values from sec to hours
    mean = np.mean(np.array(LFE_secs)/(60.*60.))     #values from sec to hours
    
    #print('WITHIN FUNCTION...')
    print('min of LFE duration is: ')
    print(min(LFE_secs))
    print('max of LFE duration is: ')
    print(max(LFE_secs))
    print('median of LFE duration is: ')
    print(np.median(np.array(LFE_secs)))
    print('mean of LFE duration is: ')
    print(np.mean(np.array(LFE_secs)))

    ax.axvline(x=median, color="indianred", linewidth=2, label=f"Median: {median:.2f} hours")
    ax.axvline(x=mean, color="indianred", linewidth=2, linestyle="dashed", label=f"Mean: {mean:.2f} hours")
    ax.axvline(x=12.0, color="indianred", linewidth=2, linestyle="dotted", label=f"Long cutoff: 11 hours")

    plt.legend()

    plt.show()

 

def Delta_t_LFEs(LFE_df, LFE_secs, LFE_duration, unet=True):
    
    time_diff_df=pd.DataFrame({'st':LFE_df['start'][1:].values, 'en':LFE_df['end'][:-1].values})
    time_diff_minutes=time_diff_df.st-time_diff_df.en
    time_diff_minutes = [time_diff_minute.total_seconds()/60. for time_diff_minute in time_diff_minutes]
       
    fig, ax = plt.subplots(1, tight_layout=True, sharey = True, figsize=(8,8))
    #ax.hist(time_diff_minutes,bins=100,range=(0,300))
    n,bins,patches=ax.hist(time_diff_minutes,bins=30,range=(0,300))
    #, label=f"N = {len(time_diff_minutes)}")
    #, label=f"N = {len(time_diff_minutes)}" #bins are 10 minutes wide
   # ax.hist(np.array(LFE_secs)/(60.*60.),bins=np.linspace(0,250,126), label=f"N = {len(LFE_secs)}")
   
    print("bins","patches","n")
    print(bins[0],n[0])#how to find the indices of where these deltat<10 minutes are?
    #fart=df.where(df.time_diff_minutes<10)  #np.where doesn't work either???
    #print(fart)
    #print(bins[1],n[1])
    #print(bins[2],n[2])
    if unet:
        ax.set_title('Histogram of LFE deltaT across Cassini mission (UNet Output)')
    else:
        ax.set_title('Histogram of LFE deltaT across Cassini mission (Training Data)')

    ax.set_xlabel('LFE time difference (minutes)')
    ax.set_ylabel('# of LFEs')
    ax.set_xlim([0,300])

    median = np.median(np.array(time_diff_minutes))    #values from sec to hours
    mean = np.mean(np.array(time_diff_minutes))     #values from sec to hours
    minimum = np.min(np.array(time_diff_minutes))    #values from sec to hours
    maximum = np.max(np.array(time_diff_minutes))     #values from sec to hours
    
    #print('FUNCTION CALCULATIONS OF MEAN AND MEDIAN')
    print('median of deltaT is: ')
    print(median)
    print('mean of deltaT is: ')
    print(mean)
    print('min of deltaT is: ')
    print(minimum)
    print('max of deltaT is: ')
    print(maximum)
    

    ax.axvline(x=10000, color="blue", linewidth=2, label=f"N = {len(time_diff_minutes)}")
    ax.axvline(x=median, color="indianred", linewidth=2, label=f"Median: {median:.2f} hours")
    ax.axvline(x=mean, color="indianred", linewidth=2, linestyle="dashed", label=f"Mean: {mean:.2f} hours")
    #ax.axvline(x=12.0, color="indianred", linewidth=2, linestyle="dotted", label=f"Long cutoff: 11 hours")

    plt.legend()

    plt.show()
  
  
    #If we have lots of delta_T that are about 30 minutes or less, consider a simple manual joining 
    #Thus make a new LFE_df_joined with (most) of the same parameters as LFE_df (start, end, duration, xyz). No label or probability as these will change
    #Then return this LFE_df_joined back to main so it can be called for the PPOs, duration histogram etc. etc.
    #For this need to be able to plot a spectrogram and show how the joining has been conducted
    #Search Elizabeth code on GitHub and my 2023PRE paper code for spectrogram plotting
    #Python_code/Cassini_plotting_main/Plot_spectrogram works! [haven't tried the polygons over it yet]
    
    return (time_diff_df,time_diff_minutes)


def InspectLongestLFEs(LFE_df, LFE_secs, LFE_duration):
    LFE_df['LFE_duration']=LFE_duration
    LFE_df['LFE_secs']=LFE_secs

    LFE_df_sorted=LFE_df.sort_values(by=['LFE_secs'],ascending=False)
    print(LFE_df_sorted)

    #Want to be able to look at these spectrograms to see if any need to be removed as outliers/unphysical



def ResidencePolar(trajectories_df,LFE_df):
    #Using Alexandra's polar plotting code rather than X-Y polar plots
    
    print(LFE_df)
    #call Cassini trajectory
    #call LFE times
    #match LFE times to get the R, lat, LT of each from Cassini trajectory
    #for i in range(500,503):
    LFE_rad=[]
    LFE_lat=[]
    LFE_LT=[]
    for i in range(len(LFE_df)):
        #print(LFE_df.start.iloc[i])
        #print(trajectories_df.loc[trajectories_df.datetime_ut + pd.Timedelta(seconds=30) == LFE_df.start.iloc[i]]) 
        LFE_rad.append(trajectories_df.loc[trajectories_df.datetime_ut + pd.Timedelta(seconds=30) == LFE_df.start.iloc[i], 'range'].values[0])
        LFE_lat.append(trajectories_df.loc[trajectories_df.datetime_ut + pd.Timedelta(seconds=30) == LFE_df.start.iloc[i], 'lat'].values[0])
        LFE_LT.append(trajectories_df.loc[trajectories_df.datetime_ut + pd.Timedelta(seconds=30) == LFE_df.start.iloc[i], 'localtime'].values[0])
    
    #define bins of the histogram
    #bin: (i) time spent by spacecraft in bins, (ii) #LFE detections in each bin, (iii) normalise detections by residence time
    #plot
    print("i love IDL")
    
    
    
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



    # divider1 = axes_grid1.make_axes_locatable(ax_cartesian)
    # divider2 = axes_grid1.make_axes_locatable(ax_polar)
    # cax1 = divider1.append_axes("right", size="3%", pad="2%")
    # cax2 = divider2.append_axes("right", size="3%", pad="2%")
    # cax2.axis("off")

    # fig.colorbar(pc, label="hours")

    plt.show()

def PlotLfeDistributions1(trajectories_df, LFE_df, split_by_duration=True, r_hist_bins=np.linspace(0, 160, 160), lat_hist_bins=np.linspace(-20, 20, 40), lt_hist_bins=np.linspace(0, 24, 48), unet=True, scale="linear", long_lfe_cutoff=11):
#this is my code with simplified finding of R, lat, LT etc.

    lfe_x, lfe_y, lfe_z = (LFE_df["x_ksm"], LFE_df["y_ksm"], LFE_df["z_ksm"])
    LFE_r=np.sqrt(lfe_x**2 + lfe_y**2 + lfe_z**2)   #range in RS
    theta = np.arctan2(lfe_y, lfe_x)
    LFE_lat=np.tan(lfe_z/LFE_r)
    LFE_lat_deg=LFE_lat*(180./np.pi)
    #theta = np.arctan2(y, x)
    #not sure what longitude_rads is? replace with theta
    longitude_degs = theta*180/np.pi
    LFE_lt=((longitude_degs+180)*24/360) % 24
    
        #need to check that these local times, latitudes are calculated correctly!

    #Local time histogram
    plt.hist(LFE_lt,bins=23)
    plt.xlabel='LFE LT (hours)'
    plt.ylabel='# of LFEs'
  
    if unet:
        plt.title('Histogram of LFE LT across Cassini mission (UNET Output)')
    else:
        plt.title('Histogram of LFE LT across Cassini mission (Training Data)')

    plt.show()

     #Local time histogram
    plt.hist(LFE_lat_deg,bins=80)
    plt.xlabel='LFE Latitude (degrees)'
    plt.ylabel='# of LFEs'

    if unet:
        plt.title('Histogram of LFE Latitude across Cassini mission (UNET Output)')
    else:
        plt.title('Histogram of LFE Latitude across Cassini mission (Training Data)')

    plt.show()


    #Local time histogram
    #Set hard limit in maximum of range to be 100 RS (to avoid pre-SOI interval)
    plt.hist(LFE_r,range=[0,100],bins=50)
    plt.xlabel='LFE Range (RS)'
    plt.ylabel='# of LFEs'

    if unet:
        plt.title('Histogram of LFE Range across Cassini mission (UNET Output)')
    else:
        plt.title('Histogram of LFE Range across Cassini mission (Training Data)')

    plt.show()



def PlotLfeDistributions(trajectories_df, LFE_df, split_by_duration=True, r_hist_bins=np.linspace(0, 160, 160), lat_hist_bins=np.linspace(-20, 20, 40), lt_hist_bins=np.linspace(0, 24, 48), unet=True, scale="linear", long_lfe_cutoff=11):
 #this has incorrect latitude calculation - needs to be coverted from radians to degrees   
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

    print('checking the calculated latitude')
    print(max(lfe_lat))
    #print('min of LFE duration is: ')
    #print(min(LFE_secs))



def CartesiansToCylindrical(x, y, z):
    #r = np.sqrt(x**2 + y**2)    #r was originally defined as this
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    return (r, theta, z)


if __name__ == "__main__":
    main()
