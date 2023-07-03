# -*- coding: utf-8 -*-
"""
Created on Friday June 23rd 2023

@author: Caitriona Jackman
"""
import matplotlib.ticker as mticker 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from bisect import bisect_left
import configparser

#SORT CONFIG FILE LATER
#config = configparser.ConfigParser()
#config.read('configurations.ini')
#input_data_fp = config['filepaths']['input_data']
#output_data_fp= config['filepaths']['output_data']

input_data_fp='C:/Users/Local Admin/Documents/Data'

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

############################
#Next want to explore some manual inspection of the longest LFEs to see if they're "real"

LFE_df['LFE_duration']=LFE_duration
LFE_df['LFE_secs']=LFE_secs

LFE_df_sorted=LFE_df.sort_values(by=['LFE_secs'],ascending=False)
print(LFE_df_sorted)

#Want to be able to look at these spectrograms to see if any need to be removed as outliers/unphysical



############################
print("Next step is to plot residence time")
#Make a function to plot residence time (See Charlie example code for Mercury)
#This needs to read in Cassini trajectory data first - from Elizabeth other code
#Then take in the LFE list and plot them over each other