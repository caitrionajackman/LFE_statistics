# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:51:35 2022
@author: eliza
Adaptations on 25/01/2023 @caitrionajackman
Added def plot_int_power
Added 3-panel plot capability to show spectrogram, integrated power and |B|
Continuing to adapt from 13/02 2024

Further adaptations on 23/01/2024 by @caitrionajackman
Want to plot a single spectrogram
Move main to the top
"""

import numpy as np
from scipy.io import readsav
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits import axes_grid1
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import time as t
import pandas as pd
from os import path
#import matplotlib as mpl#don't uncomment this (it's already defined above?)
import plotting_func as plt_func
import configparser
#Below are both needed if we are drawing polygons over the top of the spectrogram
from tfcat import TFCat
import shapely.geometry as sg

def main():
    plt.rcParams.update({'font.size': 12})

    config = configparser.ConfigParser()
    config.read('config_LFE_stats.ini')
    input_data_fp = config['filepaths']['input_data']
    output_data_fp= config['filepaths']['output_data']
    LFE_data_directory= config['filepaths']['LFE_data_directory']

    lfe_unet_data = "lfe_detections_unet_2874.csv"#Processed using findDetectionPositions.py (Elizabeth's updated UNet output file)
    lfe_training_data = "lfe_detections_training.csv" # "
    lfe_joined_list = "LFEs_joined.csv"

    #dates you would like to plot visualisations for year-month-day
    #data_start=pd.Timestamp('2004-07-24') #Badman figure 3 example
    #data_end=pd.Timestamp('2004-07-30')
    data_start=pd.Timestamp('2006-02-03')
    data_end=pd.Timestamp('2006-02-10')
    
    #data_start=pd.Timestamp('2009-11-02')
    #data_end=pd.Timestamp('2009-11-10')

    year = datetime.strftime(data_start, '%Y')

    print(data_start)
    
    if year == '2017':
        file = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file =input_data_fp + '/SKR_{}_CJ.sav'.format(year)

    saved_polys=None

    plot = {
        "spectrogram_only": False,
        "spectrogram_intpower": False,
        "two_spectrograms": True,
        "spectrogram_polygons": False,  #for this call Unet json file
        "B_time_series": False,
    }

    unet=True
    #Read in LFE list (output of Elizabeth's U-Net run on full Cassini dataset)
    print('Reading in the LFE list')
    #Unet True means the code uses the output of O'Dwyer Unet (4950 examples) 28/07/2023 - before post-processing
    #Unet False means the code uses the training data (984 examples)
    #have this unet=True (or traning = true) in codes and if True overplot LFE starts or polygons
    if unet is True:
        LFE_df = pd.read_csv(LFE_data_directory + lfe_unet_data, parse_dates=['start','end'])
    else:
        #LFE_df = pd.read_csv(LFE_data_directory + lfe_training_data, parse_dates=['start','end'])
        LFE_df = pd.read_csv(LFE_data_directory + lfe_joined_list, parse_dates=['start','end'])

    LFE_duration=LFE_df['end']-LFE_df['start']  #want this in minutes and to be smart about day/year boundaries
  
    LFE_secs=[]
    for i in range(np.array(LFE_duration).size):
        LFE_secs.append(LFE_duration[i].total_seconds())

    # if plot["spectrogram_polygons"]:
    #     Sgram(LFE_df, unet=unet)

    if plot["B_time_series"]:
        B_only(data_start,data_end,output_data_fp)
       
    
    if plot["spectrogram_intpower"]:    
       #read in data
        fp = output_data_fp + '/trajectory{}.csv'.format(year)
        mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
        intpwr_df=pd.read_csv(output_data_fp + '/intpwr_withtraj{}.csv'.format(year), parse_dates=['datetime_ut'])
       #set parameters for plot
        num_panels=2
        fontsize=18
        fig_height = 16
        fig_width = 24
       #define figure
        plt.ioff()
        fig,ax = plt.subplots(num_panels,1,figsize=(fig_width, fig_height))
       #set width and height space between subplots
        fig.subplots_adjust(wspace=0.1,hspace=0.1)
       #plot flux
        saved_polys=None
        ax[0]=plot_flux(ax[0], data_start, data_end, file,colour_in=None, fontsize=fontsize)

     #   LFE_df = pd.read_csv(LFE_data_directory + lfe_joined_list, parse_dates=['start','end'])
        #Loop through to draw horizontal bars for joined LFEs
        #for i in range(len(LFE_df)):
        #    ax[0].hlines(y=1000,xmin=pd.Timestamp(LFE_df['start'][i]),xmax=pd.Timestamp(LFE_df['end'][i]),linewidth=4,color='dimgray')

        #Loop through to draw offset horizontal bars for Original UNet LFEs
     #   LFE_df = pd.read_csv(LFE_data_directory + lfe_unet_data, parse_dates=['start','end'])
        #for i in range(len(LFE_df)):
        #    ax[0].hlines(y=1200,xmin=pd.Timestamp(LFE_df['start'][i]),xmax=pd.Timestamp(LFE_df['end'][i]),linewidth=4,color='dimgray')

    #Offset horizontal bars for original Unet output                  
      #  ax[0].hlines(y=1100,xmin=pd.Timestamp(2004,7,24,1,21),xmax=pd.Timestamp(2004,7,24,5,51),linewidth=4,color='skyblue')
      #  ax[0].hlines(y=1100,xmin=pd.Timestamp(2004,7,24,7,51),xmax=pd.Timestamp(2004,7,24,16,51),linewidth=4,color='skyblue')
        
      #  ax[0].hlines(y=1200,xmin=pd.Timestamp(2004,7,24,21,12),xmax=pd.Timestamp(2004,7,25,3,21),linewidth=4,color='skyblue')
      #  ax[0].hlines(y=1150,xmin=pd.Timestamp(2004,7,25,3,30),xmax=pd.Timestamp(2004,7,25,13,51),linewidth=4,color='skyblue')

      #  ax[0].hlines(y=1100,xmin=pd.Timestamp(2004,7,25,17,39),xmax=pd.Timestamp(2004,7,28,2,12),linewidth=4,color='skyblue')
      #  ax[0].hlines(y=1100,xmin=pd.Timestamp(2004,7,28,2,51),xmax=pd.Timestamp(2004,7,28,9,3),linewidth=4,color='skyblue')

      #  ax[0].hlines(y=1200,xmin=pd.Timestamp(2004,7,28,10,6),xmax=pd.Timestamp(2004,7,28,20),linewidth=4,color='skyblue')
      #  ax[0].hlines(y=1150,xmin=pd.Timestamp(2004,7,28,19,51),xmax=pd.Timestamp(2004,7,29,17,45),linewidth=4,color='skyblue')
        
        #horizontal bars for joined LFEs
      #  ax[0].hlines(y=1000,xmin=pd.Timestamp(2004,7,24,1,21),xmax=pd.Timestamp(2004,7,24,5,51),linewidth=4,color='dimgray')
      #  ax[0].hlines(y=1000,xmin=pd.Timestamp(2004,7,24,7,51),xmax=pd.Timestamp(2004,7,24,16,51),linewidth=4,color='dimgray')
      #  ax[0].hlines(y=1000,xmin=pd.Timestamp(2004,7,24,21,12),xmax=pd.Timestamp(2004,7,25,13,51),linewidth=4,color='dimgray')
      #  ax[0].hlines(y=1000,xmin=pd.Timestamp(2004,7,25,17,39),xmax=pd.Timestamp(2004,7,28,2,12),linewidth=4,color='dimgray')
      #  ax[0].hlines(y=1000,xmin=pd.Timestamp(2004,7,28,2,51),xmax=pd.Timestamp(2004,7,28,9,3),linewidth=4,color='dimgray')
      #  ax[0].hlines(y=1000,xmin=pd.Timestamp(2004,7,28,10,6),xmax=pd.Timestamp(2004,7,29,17,45),linewidth=4,color='dimgray')
       
        #Plot 3 integrated power bands
        ax[1]=plot_int_power(ax[1], intpwr_df, data_start, data_end, 'orange', '100-600', csize=fontsize)
        ax[1]=plot_int_power(ax[1], intpwr_df, data_start, data_end, 'skyblue', '40-100', csize=fontsize)
        ax[1]=plot_int_power(ax[1], intpwr_df, data_start, data_end, 'dimgray', '5-40', csize=fontsize)
        ax[1].legend(fontsize=fontsize-2)
       #format y axis
        ymax=intpwr_df['100-600'].max()
       #ax[1].set_ylim(0, ymax)
        ax[1].set_yscale('log')
        #Vertical lines for original UNet output
       # LFE_df = pd.read_csv(LFE_data_directory + lfe_unet_data, parse_dates=['start','end'])

        for i in range(len(LFE_df)):
            ax[1].axvline(x=pd.Timestamp(LFE_df['start'][i]),color='dimgray')
            ax[1].axvline(x=pd.Timestamp(LFE_df['end'][i]),color='skyblue')
            
        # ax[1].axvline(x=pd.Timestamp(2004,7,24,1,21),color='dimgray')
        # ax[1].axvline(x=pd.Timestamp(2004,7,24,5,51),color='skyblue')
      
       
        #Add vertical linesat specified dates (eventually the LFE starts and ends)    
        #xcoords=LFE_df['start'] #in proper date format 2004-01-01 14:30:00
      #  xcoords=[pd.Timestamp(2006,1,1,15),pd.Timestamp(2006,1,2,16)]
       # breakpoint()
       # for xc in xcoords:
        #    plt.axvline(x=xc,linewidth=2)
       # ax[1].axvline(x=pd.Timestamp(2006,1,1,15))
              
 
        plt.tight_layout()
        plt.show()
        plt.close()

    #plt.savefig('test.png', bbox_inches='tight')
 #close figure
 
       #vertical lines for crossing
        # crossing_list = pd.read_csv(input_data_fp +'/Cassini_crossing_list.txt')
        # t_timestamp = pd.Series([pd.Timestamp(str(year)) - pd.Timedelta(1, 'D') + \
        #                         pd.Timedelta(t * 1440, 'm') for year, t in \
        #                         zip(crossing_list['Yearcross'], crossing_list['DOYFRACcross'])])
        # types = crossing_list['Typecross']
           
       # for i,type_ in zip(t_timestamp, types):
       #     num_ = dates.date2num(i)
       # #    ax[2].vlines(num_, 0, ymax, color='k')
       #     if type_ =='MP':
       #         ax[2].vlines(num_, 0, ymax, color='blue', label='MP')
       #     elif type_ =='BS':
       #         ax[2].vlines(num_, 0, ymax, color='red', label='BS')
         
      

        
    # if plot["spectrogram_only"]:
    #     #Sgram(LFE_df, unet=unet)
    #     #plot_flux(ax,time_view_start, time_view_end, file, colour_in=None, frequency_lines=None):
    #     #plot_flux(data_start,data_end,file)
    #     time, freq, flux = extract_data(file, time_view_start=data_start,\
    #                                     time_view_end=data_end,val='s')
    #     #Parameters for colorbar
    #     #This is the function that does flux normalisation based on s/c location
    #     #vmin, vmax=plt_func.flux_norm(time[0], time[-1])   #change from log10 to actual values.
    #     clrmap ='viridis'
    #     vmin = np.quantile(flux[flux > 0.], 0.05)
    #     vmax = np.quantile(flux[flux > 0.], 0.95)
    #     scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
        
    #     #Make figure
    #     fontsize = 20
    #     fig = plt.figure()
    #     im=ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=clrmap,  shading='auto')
    #     ax.set_yscale('log')
                
    #     #format axis 
    #     ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
    #     ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
        
        
        
    
#Uncomment this for plotting polygons from .json file
#polygon_fp=root+"output_data/ML_lfes.json"
#saved_polys = get_polygons(polygon_fp,data_start, data_end)
#Uncomment this not to plot polygons
  #  saved_polys=None

    #make first figure
    # fig,ax = plt.subplots(1,1,figsize=(16,12))
    # year = datetime.strftime(data_start, '%Y')
    # fp = output_data_fp + '/trajectory{}.csv'.format(year)
    # mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
    # intpwr_df=pd.read_csv(output_data_fp + '/intpwr_withtraj{}.csv'.format(year))
    # intpwr_df['datetime_ut'] = mag_df['datetime_ut']
    # ax=plot_b_total(ax, mag_df, data_start, data_end)
    # plt.show()

    
# #Make figure with given number of panels
# num_panels=3
# plt.ioff()
# fig,ax = plt.subplots(num_panels,1,figsize=(16,12))
# fig.subplots_adjust(wspace=0.5,hspace=0.5)
# #add in extra argument 'frequency_lines'= [100, 200...] to plot horizontal lines at given frequency
# ax[0]=plot_flux(ax[0], data_start, data_end, file,colour_in=saved_polys)
# ax[1]=plot_int_power(ax[1], intpwr_df, data_start, data_end)
# ax[2]=plot_b_total(ax[2], mag_df, data_start, data_end)


    if plot["two_spectrograms"]:    
       #read in data
        fp = output_data_fp + '/trajectory{}.csv'.format(year)
        mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
        intpwr_df=pd.read_csv(output_data_fp + '/intpwr_withtraj{}.csv'.format(year), parse_dates=['datetime_ut'])
       #set parameters for plot
        num_panels=2
        fontsize=18
        fig_height = 16
        fig_width = 24
       #define figure
        plt.ioff()
        fig,ax = plt.subplots(num_panels,1,figsize=(fig_width, fig_height))
       #set width and height space between subplots
        fig.subplots_adjust(wspace=0.1,hspace=0.1)
       #plot flux
        saved_polys=None
        ax[0]=plot_flux(ax[0], data_start, data_end, file,colour_in=None, fontsize=fontsize)


        polygon_fp=LFE_data_directory + "2004001_2017258_catalogue.json"
        print("do we read polygon file path correctly?")
        print(polygon_fp)
        saved_polys = get_polygons(polygon_fp,data_start, data_end)
       # breakpoint()

        ax[1]=plot_flux(ax[1], data_start, data_end, file, colour_in=saved_polys,fontsize=fontsize)
          
        #plot_pol(ax,time_view_start, time_view_end, file,colour_in=None,frequency_lines=None):          
        #plot_pol(ax[1],data_starttime_view_starttime_view_start, time_view_end, file,colour_in=None,frequency_lines=None):          
        #ax[1]=plot_pol(ax[1],data_start,data_end,file,colour_in=None,frequency_lines=None)
            
        
#Uncomment this for plotting polygons from .json file

     #   from tfcat.codec import load

     #   with open(polygon_fp, 'r') as f:
     #       tfcat_data = load(f)
     #       breakpoint()
       # catalogue1=cls(tfcat_data, file_uri=polygon_fp)
        
 
#Uncomment this not to plot polygons
  #  saved_polys=None

#I don't see how these are connected... surely need to retrieve the polygons from the json file and then overplot them on the spectrogram?

    #For plotting polygons onto spectrogram.
        #if colour_in is not None:
        #    for shape in colour_in:
        #        shape_=shape.copy()
        #        shape_[:,0]=[mdates.date2num(datetime.fromtimestamp(i)) for i in shape_[:,0]]
        #        ax.add_patch(Polygon(shape_, color=(0.163625, 0.471133, 0.558148), linestyle='dashed',linewidth=4, alpha=1, fill=False))

    
        plt.tight_layout()
        plt.show()
        plt.close()

    
    
#'''____This was adapted from original code for the space labelling tool!!!___'''
def get_polygons(polygon_fp,start, end):
    print("are we getting here and finding polygons?")        
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        print(" a path exists ")
        catalogue = TFCat.from_file(polygon_fp)
        #breakpoint()
        print(" a catalogue exists ")
        #for i in range(len(catalogue)):
        for i in range(len(catalogue._data["features"])):
            time_points=np.array(catalogue._data["features"][i]['geometry']['coordinates'][0])[:,0]
            if any(time_points <= unix_end) and any(time_points >= unix_start):
                polygon_array.append(np.array(catalogue._data["features"][i]['geometry']['coordinates'][0]))
    
    return polygon_array
                
    
#    polygon_array=[]
 #   if path.exists(polygon_fp):
  #      print(" a path exists ")
   #     catalogue = TFCat.from_file(polygon_fp)
    #    #breakpoint()
     #   print(" a catalogue exists ")
      #  for i in range(len(catalogue)):
       #     #time_points=np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:,0]
        #    time_points=np.array(catalogue.data.features[i]['geometry']['coordinates'][0])[:,0]
         #   if any(time_points <= unix_end) and any(time_points >= unix_start):
          #      #polygon_array.append(np.array(catalogue._data.features[i]['geometry']['coordinates'][0]))
           #     polygon_array.append(np.array(catalogue.data.features[i]['geometry']['coordinates'][0]))
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval 

    
        
        


     
def B_only(data_start,data_end,output_data_fp):
    #make first figure
    fig,ax = plt.subplots(1,1,figsize=(16,12))
    year = datetime.strftime(data_start, '%Y')
    print(year)
    fp = output_data_fp + '/trajectory{}.csv'.format(year)
    mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
    print(mag_df)
    intpwr_df=pd.read_csv(output_data_fp + '/intpwr_withtraj{}.csv'.format(year))
    intpwr_df['datetime_ut'] = mag_df['datetime_ut']
    ax=plot_b_total(ax, mag_df, data_start, data_end)
    plt.show()



'''____This was adapted from original code for the space labelling tool!!!___'''
def extract_data(file_data, time_view_start, time_view_end, val):
    # read the save file and copy variables
    time_index = 't'
    freq_index = 'f'
    val_index = val
    file = readsav(file_data)

    t_doy = file[time_index].copy()
    doy_one = pd.Timestamp(str(1997)) - pd.Timedelta(1, 'D')
    t_timestamp = np.array([doy_one + pd.Timedelta(t * 1440, 'm') for t in t_doy],
        dtype=pd.Timestamp)
    t_isostring = np.array([datetime.strftime(i,'%Y-%m-%dT%H:%M:%S') for i in t_timestamp])
    time =t_isostring
    #print(time)
    #time = np.vectorize(fix_iso_format)(t_isostring)
    time = np.array(time, dtype=np.datetime64)
    time_view = time[(time >= time_view_start) & (time < time_view_end)]

    # copy the flux and frequency variable into temporary variable in
    # order to interpolate them in log scale
    s = file[val_index][:, (time >= time_view_start) & (time <= time_view_end)].copy()
    frequency_tmp = file[freq_index].copy()

    # frequency_tmp is in log scale from f[0]=3.9548001 to f[24] = 349.6542
    # and then in linear scale above so it's needed to transfrom the frequency
    # table in a full log table and einterpolate the flux table (s --> flux
    frequency = 10**(np.arange(np.log10(frequency_tmp[0]), np.log10(frequency_tmp[-1]), (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/399, dtype=float))
    flux = np.zeros((frequency.size, len(time_view)), dtype=float)

    for i in range(len(time_view)):
        flux[:, i] = np.interp(frequency, frequency_tmp, s[:, i])

    return time_view, frequency, flux

def plot_mask(ax,time_view_start, time_view_end, val, file_data,polygon_fp):
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval
    polygon_array=get_polygons(polygon_fp, time_view_start, time_view_end)
    #signal data and time frequency values within the time range specified.
    time_dt64, frequency, flux=extract_data(file_data, time_view_start, time_view_end, val)
    time_unix=[i.astype('uint64').astype('uint32') for i in time_dt64]
    #Meshgrid of time/frequency vals.
    times, freqs=np.meshgrid(time_unix, frequency)
    #Total length of 2D signal array.
    data_len = len(flux.flatten())
    #indices of each item in flattened 2D signal array.
    index = np.arange(data_len, dtype=int)
    #Co-ordinates of each item in 2D signal array.
    coords = [(t, f) for t,f in zip(times.flatten(), freqs.flatten())]
    data_points = sg.MultiPoint([sg.Point(x, y, z) for (x, y), z in zip(coords, index)])
    #Make mask array.
    mask = np.zeros((data_len,))
    #Find overlap between polygons and signal array.
    #Set points of overlap to 1.
    for i in polygon_array:
        fund_polygon = sg.Polygon(i)
        fund_points = fund_polygon.intersection(data_points)
        if len(fund_points.bounds)>0:
            mask[[int(geom.z) for geom in fund_points.geoms]] = 1
    mask = (mask == 0)
    #Set non-polygon values to zero in the signal array.
    #flux_ones = np.where(flux>0, 1, np.nan)
    v = np.ma.masked_array(flux, mask=mask).filled(np.nan)
    
    #colorbar limits
    vmin = np.quantile(flux[flux > 0.], 0.05)
    vmax = np.quantile(flux[flux > 0.], 0.95)
    scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap('binary_r').copy()
    
    #Plot Figure
    fontsize=20
    fig = plt.figure()
    im=ax.pcolormesh(time_dt64,frequency, v,norm=scaleZ,cmap=cmap,shading='auto')
    
    #format axis 
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize-6)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    #ax.set_title(f'{time_view_start} to {time_view_end}', fontsize=fontsize + 2)
    dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_formatter(dateFmt)
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    if val == 's':
        cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2)
    elif val =='v':
        cb.set_label('Normalized'+'\n Degree of'+'\n Circular Polarization', fontsize=fontsize-2)
        
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    plt.close(fig)
    
    return ax


def plot_flux(ax,time_view_start, time_view_end, file, colour_in=None, frequency_lines=None,fontsize=18):
    
    #Load data from .sav file
    time, freq, flux = extract_data(file, time_view_start=time_view_start,\
                                    time_view_end=time_view_end,val='s')
    #Parameters for colorbar
    #This is the function that does flux normalisation based on s/c location
    #vmin, vmax=plt_func.flux_norm(time[0], time[-1])   #change from log10 to actual values.
    clrmap ='viridis'
    vmin = np.quantile(flux[flux > 0.], 0.05)
    vmax = np.quantile(flux[flux > 0.], 0.95)
    scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    #Make figure
    fig = plt.figure()
    im=ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=clrmap,  shading='auto')
    ax.set_yscale('log')
    
    
    #format axis 
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    #ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_title(f'{time_view_start} to {time_view_end}', fontsize=fontsize + 2)
    
    ######### X label formatting ###############
    
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    #normal
    dateFmt = mdates.DateFormatter('%m-%d\n%H:%M')
    ax.xaxis.set_major_formatter(dateFmt)
    
    #For using trajectory data
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(plt_func.ephemeris_fmt_hour_tick))
    #eph_str = '\n'.join(['DOY\n',
     #           r'$R_{sc}$ ($R_{S}$)',
      #          r'$\lambda_{sc}$ ($^{\circ}$)',
       #         r'LT$_{sc}$ (Hrs)'])
    #kwargs = {'xycoords': 'figure fraction',
     #   'fontsize': fontsize-6}
    #kwargs['xy'] = (0.03, 0.009)
    #ax.annotate(eph_str,**kwargs)
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    #For adding horizontal lines at specific frequencies
    if frequency_lines is not None:
        for i in frequency_lines:
            ax.hlines(i, time[0], time[-1], colors = 'darkslategray',linewidth=1,linestyles='--', label='{}kHz'.format(i))
          
     #For plotting polygons onto spectrogram.
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.fromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_, color='black', linestyle='dashed',linewidth=4, alpha=1, fill=False))
        
    plt.close(fig)
    return ax

def plot_pol(ax,time_view_start, time_view_end, file,colour_in=None,frequency_lines=None):
    
    #Load data from .sav file
    time, freq, pol = extract_data(file, time_view_start=time_view_start, \
                                   time_view_end=time_view_end,val='v')
    #Parameters for colorbar
    vmin=-1
    vmax=1
    clrmap ='binary'
    scaleZ = colors.Normalize(vmin=vmin, vmax=vmax)
    
    #Make figure
    fontsize = 20
    fig = plt.figure()
    im=ax.pcolormesh(time, freq, pol, norm=scaleZ, cmap=clrmap, shading='auto')
    ax.set_yscale('log')
    
    
    #format axis 
    ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    #Uncomment to set title
    #ax.set_title(f'{time_view_start} to {time_view_end}', fontsize=fontsize + 2)
    
    ######### X label formatting ###############
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    #normal
    #dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #ax.xaxis.set_major_formatter(dateFmt)
    
    #For using trajectory data
    ax.xaxis.set_major_formatter(plt.FuncFormatter(plt_func.ephemeris_fmt_hour_tick))
    eph_str = '\n'.join(['DOY\n',
                r'$R_{sc}$ ($R_{S}$)',
                r'$\lambda_{sc}$ ($^{\circ}$)',
                r'LT$_{sc}$ (Hrs)'])
    kwargs = {'xycoords': 'figure fraction',
        'fontsize': fontsize-6}
    kwargs['xy'] = (0.03, 0.009)
    ax.annotate(eph_str,**kwargs)
    
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label('Normalized'+'\n Degree of'+'\n Circular Polarization', fontsize=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    #For adding horizontal lines at specific frequencies
    if frequency_lines is not None:
        for i in frequency_lines:
            ax.hlines(i, time[0], time[-1], colors = 'darkslategray',linewidth=1,linestyles='--', label='{}kHz'.format(i))
              
    #For plotting polygons onto spectrogram.
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.fromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_, color=(0.163625, 0.471133, 0.558148), linestyle='dashed',linewidth=4, alpha=1, fill=False))
        
    plt.close(fig)
    return ax


#I commented out the below 10 lines n 25/01/23 at 12:41 to move all plotting commands together
##dates you would like to plot spectrogram for 
#data_start=pd.Timestamp('2006-01-01')
#data_end=pd.Timestamp('2006-01-04')
#year = datetime.strftime(data_start, '%Y')

#if year == '2017':
#    file = input_data_fp + '/SKR_2017_001-258_CJ.sav'
#else: 
#    file =input_data_fp + '/SKR_{}_CJ.sav'.format(year)
    
##Uncomment this for plotting polygons from .json file
##polygon_fp=root+"output_data/ML_lfes.json"
##saved_polys = get_polygons(polygon_fp,data_start, data_end)
##Uncomment this not to plot polygons
#saved_polys=None




        

def plot_b_total(ax, mag_df, start, end, csize=12):
   # mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
    mag_df = mag_df.loc[mag_df['datetime_ut'].between(start, end), :].reset_index(drop=True)
    ax.plot(mag_df['datetime_ut'],mag_df['btotal'], color='dimgrey')
    figure = ax.figure
    figure.subplots_adjust(bottom=0.35)
    fontsize = 20
    
    #limits on axes
    ymax=mag_df['btotal'].max()
    x_start= mag_df['datetime_ut'].iloc[0]
    x_end= mag_df['datetime_ut'].iloc[-1]
    ax.set_ylim(0,ymax) #refine y limits
    ax.set_xlim(x_start, x_end)
    
    #Format and label axes 
    ax.tick_params(labelsize=csize-2)
    
    
    ax.set_ylabel("$B_{TOTAL}$ $(nT)$", fontsize=csize)
   # ax.set_title(f'{x_start} to {x_end}', fontsize=csize+2)
    
   #I commented the 3 lines below if we want to have Btotal as bottom panel with ephemeris underneath
   
    #ax.set_xlabel('Time')
    #dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #ax.xaxis.set_major_formatter(dateFmt)
    
    #For using trajectory data
    ax.xaxis.set_major_formatter(plt.FuncFormatter(plt_func.ephemeris_fmt_hour_tick))
    eph_str = '\n'.join(['DOY\n',
                r'$R_{sc}$ ($R_{S}$)',
                r'$\lambda_{sc}$ ($^{\circ}$)',
                r'LT$_{sc}$ (Hrs)'])
    kwargs = {'xycoords': 'figure fraction',
        'fontsize': fontsize-6}
#    kwargs['xy'] = (0.03, 0.009)
    kwargs['xy'] = (0.03, 0.12)
    ax.annotate(eph_str,**kwargs)
 
    
    # create empty space to fit colorbar in spectrogram panel
    # (effectively line up top/bottom panels)
    #Can use this when you want to add this as extra panel in spectrograms.
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cax.set_facecolor('none')
    for axis in ['top','bottom','left','right']:
       cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])
    return ax


#fig,ax = plt.subplots(1,1,figsize=(16,12))
#data_start=pd.Timestamp('2006-01-01')   #year-month-day
#data_end=pd.Timestamp('2006-01-04')
#year = datetime.strftime(data_start, '%Y')
#fp = output_data_fp + '/trajectory{}.csv'.format(year)
#mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
#intpwr_df=pd.read_csv(output_data_fp + '/intpwr_withtraj{}.csv'.format(year))
#intpwr_df['datetime_ut'] = mag_df['datetime_ut']
#ax=plot_b_total(ax, mag_df, data_start, data_end)
#plt.show()
   
def plot_int_power(ax, intpwr_df, start, end, line_color, freq_range, csize=18):
    intpwr_df_ = intpwr_df.copy().loc[intpwr_df['datetime_ut'].between(start, end), :].reset_index(drop=True)
    ax.plot(intpwr_df_['datetime_ut'],intpwr_df_[freq_range], color=line_color, label=freq_range)
    
    #limits on axes
    #ymax=intpwr_df_['100-600'].max()
    x_start= intpwr_df_['datetime_ut'].iloc[0]
    x_end= intpwr_df_['datetime_ut'].iloc[-1]
    #ax.set_ylim(0,ymax) #refine y limits
    ax.set_xlim(x_start, x_end)
    
    #Format and label axes 
    ax.tick_params(labelsize=csize-2)
    
   # ax.set_xlabel('Time')
    ax.set_ylabel("Power(Wsr$^{-1}$)", fontsize=csize)
   # ax.set_title(f'{x_start} to {x_end}', fontsize=csize+2)
    #ax.set_yscale('log')#FIXME figure out how to make this look good

    dateFmt = mdates.DateFormatter('%m-%d\n%H:%M')
    ax.xaxis.set_major_formatter(dateFmt)

    
    # create empty space to fit colorbar in spectrogram panel
    # (effectively line up top/bottom panels)
    #Can use this when you want to add this as extra panel in spectrograms.
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cax.set_facecolor('none')
    for axis in ['top','bottom','left','right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])
    return ax


if __name__ == "__main__":
    main()
