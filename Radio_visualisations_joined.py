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
from datetime import timedelta, datetime

def main():
    plt.rcParams.update({'font.size': 12})

    config = configparser.ConfigParser()
    config.read('config_LFE_stats.ini')
    input_data_fp = config['filepaths']['input_data']
    output_data_fp= config['filepaths']['output_data']
    LFE_data_directory= config['filepaths']['LFE_data_directory']

    lfe_unet_data = "lfe_detections_unet_2874.csv" #Processed using findDetectionPositions.py (Elizabeth's updated UNet output file)
    lfe_training_data = "lfe_detections_training.csv" 
    lfe_joined_list = "LFEs_joined.csv"

    #dates you would like to plot visualisations for year-month-day
    #data_start=pd.Timestamp('2004-07-24') #Badman figure 3 example
    #data_end=pd.Timestamp('2004-07-30')
    data_start=pd.Timestamp('2012-09-07')
    data_end=pd.Timestamp('2012-09-10')
    
    #data_start=pd.Timestamp('2009-11-02')
    #data_end=pd.Timestamp('2009-11-10')

    year = datetime.strftime(data_start, '%Y')

    print(data_start)
    
    if year == '2017':
        file = input_data_fp + '/SKR_spectrogram_data/SKR_2017_001-258_CJ.sav'
    else: 
        file = input_data_fp + '/SKR_spectrogram_data/SKR_{}_CJ.sav'.format(year)

    saved_polys=None

    plot = {
        "spectrogram_only": False,
        "spectrogram_intpower": False,
        "two_spectrograms": False,
        "spectrogram_polygons": False,  #for this call Unet json file
        "B_time_series": False,
        "joining_process_polygons": False, # SHOWS JOINING PROCESS. a) two separate polys, b) joined polys ----- EX. /Sample_Visualizations/Example_Joining_Process.png
        "spectrogram_joint_polygons": True # Same as "two_spectograms" but now using JOINT .json file ----- EX. /Sample_Visualizations/Sample_Joined_Polygon.png
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
        fp = output_data_fp + '/trajectory/trajectory{}.csv'.format(year)
        mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
        intpwr_df=pd.read_csv(output_data_fp + '/intpwr/intpwr_withtraj{}.csv'.format(year), parse_dates=['datetime_ut'])
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
        fp = output_data_fp + '/trajectory/trajectory{}.csv'.format(year)
        mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
        intpwr_df=pd.read_csv(output_data_fp + '/intpwr/intpwr_withtraj{}.csv'.format(year), parse_dates=['datetime_ut'])
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


        polygon_fp=LFE_data_directory + "/2004001_2017258_catalogue.json"
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
        
    # If I want to visualize polygons across years, must run loop for YEARLY increments 
    if plot['joining_process_polygons']:
        # Do we even need to join? 
        total_df, saved_polys, joint_polys, poly_num, joint_num = comparison(str(data_start), str(data_end), LFE_data_directory, lfe_unet_data, lfe_joined_list)

        if poly_num == joint_num:
            print('No polygons need joining within this time block!')
        else:
            # We need to find EXACT START & END dates of original & joined polygons in a certain time interval
            # Want to find index values in TOTAL DF that 
            totalstart = np.unique(total_df['start'])
            totalend = np.unique(total_df['end'])
            jointstart = np.unique(joint_polys['start'])
            jointend = np.unique(joint_polys['end'])

            difstarts = []
            difends = []

            for element in totalend:
                if element not in jointend:
                    difends.append(element)

            for element in totalstart:
                if element not in jointstart:
                    difstarts.append(element)

            lo = []
            for i, j in zip(difends, difstarts):
                lo.append([i,j])

            # Save start & end times FOR FULL LFE DATA
            newstarts = []
            newends = []
        
            for l in range(len(lo)):
                place = np.where(total_df['end'] == difends[l])[0][0]
                newstarts.append(total_df['start'].iloc[place])
                newends.append(total_df['end'].iloc[place + 1])

            # Define quantiles / paths PER FIGURE
            quantiles0 = [0.75, 0.8, 0.85, 0.9, 0.95]
            quantiles1 = [0.25, 0.2, 0.15, 0.1, 0.05]
            
            # Take each polygon and join along 95% and 5% quantile lines
            for ind in range(len(newstarts)): # len(newstarts)
                # Initialize FIGURE
                num_panels=1
                fontsize=18
                fig_height = 8
                fig_width = 10

                plt.ioff()

                # Two panels - 1) Show 2 separate polygons with 5 different paths
                            #  2) Joint polygons !
                fig, (ax0, ax1) = plt.subplots(2, figsize=(fig_width, fig_height))
                #set width and height space between subplots
                fig.subplots_adjust(wspace=0.1,hspace=0.1)

                # Initialize Polygons
                polygon_fp= LFE_data_directory + "2004001_2017258_catalogue.json"
                data_start=pd.Timestamp(f'{newstarts[ind]}') 
                data_end=pd.Timestamp(f'{newends[ind]}')
                print('2 polygons are within:', data_start, 'and ', data_end)

                saved_polys = get_polygons(polygon_fp, data_start, data_end)

                # ACCOUNTS FOR DAYLIGHT SAVING'S TIME - using european standards
                dst = True
                if dst == True:
                    if data_start >= europe_dst_range(data_start.year)[0] and data_end <=  europe_dst_range(data_start.year)[1]:
                        print('Subtracting an hour from these polygons:', data_start, ' and ', data_end)
                        for i in range(len(saved_polys)):
                            for j in range(len(saved_polys[i])):
                                saved_polys[i][j, 0] = (datetime.fromtimestamp(saved_polys[i][j, 0]) - pd.Timedelta(1, 'hour')).timestamp()
                    else:
                        saved_polys = get_polygons(polygon_fp, data_start, data_end)
                else:
                    pass 
            
                tup = []
                tlow = []
                fup = []
                flow = []
                conditions = []

                for quant0, quant1 in zip(quantiles0, quantiles1):
                    #print('Quantiles:', quant0, quant1)
                    # Polygon 0 - fmax, fmin, tmax
                    max_t0 = max(saved_polys[0][:, 0])
                    min_t0 = min(saved_polys[0][:, 0])
                    cond = np.quantile(saved_polys[0][:, 0], quant0)

                    # Find frequencies that pass threshold condition on POLYGON 0
                    max_f0 = np.max(saved_polys[0][np.where(saved_polys[0][:, 0] >= cond), 1][0]) # Frequency MAX
                    new_polys = saved_polys[0][np.where(saved_polys[0][:, 0] >= cond)] # BAD POINTS
                    old_polys = saved_polys[0][np.where(saved_polys[0][:, 0] < cond)] # GOOD POINTS
                    delete_polys_len0 = len(new_polys)
                    ind_max_t = np.where(max_f0 == saved_polys[0][np.where(saved_polys[0][:, 0] >= cond),1])[1][0] # Index of time MAXIMUM from where to draw initial line
                    max_t0_u = new_polys[ind_max_t, 0] # Actual T MAX Upper VALUE
                    #print('Maximum t & Maximum freq for LEFT POLY:', max_t0_u, max_f0)

                    min_f0 = np.min(saved_polys[0][np.where(saved_polys[0][:, 0] >= cond), 1][0]) # Frequency min
                    ind_min_t = np.where(min_f0 == saved_polys[0][np.where(saved_polys[0][:, 0] >= cond),1])[1][0] # Index of time MINIMUM from where to draw initial line
                    max_t0_l = new_polys[ind_min_t, 0] # Actual T MAX Lower VALUE
                    #print('Maximum t & Minimum freq for LEFT POLY:', max_t0_l, min_f0)

                    # Polygon 1 - fmax, fmin, tmin
                    max_t1 = max(saved_polys[1][:, 0])
                    min_t1 = min(saved_polys[1][:, 0])
                    cond1 = np.quantile(saved_polys[1][:, 0], quant1)

                    # Find frequencies that pass threshold condition on POLYGON 1
                    max_f1 = np.max(saved_polys[1][np.where(saved_polys[1][:, 0] <= cond1), 1][0]) # Frequency MAX
                    new_polys_1 = saved_polys[1][np.where(saved_polys[1][:, 0] <= cond1)] # BAD POINTS
                    old_polys_1 = saved_polys[1][np.where(saved_polys[1][:, 0] > cond1)] # GOOD POINTS
                    delete_polys_len1 = len(new_polys_1)
                    ind_max_t1 = np.where(max_f1 == saved_polys[1][np.where(saved_polys[1][:, 0] <= cond1),1])[1][0]  # Index of time MAXIMUM from where to draw initial line
                    min_t1_u = new_polys_1[ind_max_t1, 0]  # Actual T MIN upper VALUE
                    #print('Minimum t & Maximum freq for RIGHT POLY:', min_t1_u, max_f1)

                    min_f1 = np.min(saved_polys[1][np.where(saved_polys[1][:, 0] <= cond1), 1][0]) # Frequency min
                    ind_min_t1 = np.where(min_f1 == saved_polys[1][np.where(saved_polys[1][:, 0] <= cond1),1])[1][0] # Index of time MINIMUM from where to draw initial line
                    min_t1_l = new_polys_1[ind_min_t1, 0]  # Actual T MIN Lower VALUE
                    #print('Minimum t & Minimum freq for RIGHT POLY:', min_t1_l, min_f1)

                    # Apped to appropriate list
                    tup.append([max_t0_u, min_t1_u])
                    tlow.append([max_t0_l, min_t1_l])
                    fup.append([max_f0, max_f1])
                    flow.append([min_f0, min_f1])
                    conditions.append([cond, cond1])

                #print('tup:', tup)
                #print('poly0 bad points:', new_polys)
                #print('poly1 bad points:', new_polys_1)
                #print('lengths:', delete_polys_len0, delete_polys_len1)
    
                # FIRST PLOT - separate polygons with QUANTILE LINES
                ax0=plot_flux2(ax0, data_start, data_end, file, colour_in=saved_polys, fontsize=fontsize)
            
                
                #colors_labels = ['red', 'orange', 'yellow', 'green', 'purple']
                colors_labels = ["#e3d66c",
    "#ff6cac",
    "#57cc83",
    "#e7ac2a",
    "#47d7f0"] 
                for i in range(len(tup)):
                    #ax0.axvline(x = datetime.fromtimestamp(conditions[i][0]), linestyle = "--")
                    #ax0.axvline(x = datetime.fromtimestamp(conditions[i][1]), linestyle = "--")
                    # Generate NEW DATAPOINTS == len(new_polys)
                    x10 = np.asarray(tup[i])
                    y1 = np.asarray(fup[i])
                    x1 = np.array([datetime.fromtimestamp(tup[i][0]), datetime.fromtimestamp(tup[i][1])])

                    x20 = np.asarray(tlow[i])
                    y2 = np.asarray(flow[i])
                    x2 = np.array([datetime.fromtimestamp(tlow[i][0]), datetime.fromtimestamp(tlow[i][1])])
                    
                    # Define line parameters
                    m1 = (y1[1]-y1[0]) / (x10[1]-x10[0])
                    b1 = (y1 - (m1*x10))[0]
                    m2 = (y2[1]-y2[0]) / (x20[1]-x20[0])
                    b2 = (y2 - (m2*x20))[0]

                    xnew10 = np.linspace(x10[0], x10[-1], delete_polys_len0) #delete_polys_len0
                    ynew1 = m1*xnew10 + b1
                    xnew1 = [datetime.fromtimestamp(xnew10[ind]) for ind in range(len(xnew10))]
                    xnew20 = np.linspace(x20[0], x20[-1], delete_polys_len1) #delete_polys_len1
                    ynew2 = m2*xnew20 + b2
                    xnew2 = [datetime.fromtimestamp(xnew20[ind]) for ind in range(len(xnew20))]
                    
                    # PLOT
                    ax0.plot([datetime.fromtimestamp(tup[i][0]), datetime.fromtimestamp(tup[i][1])], [fup[i][0], fup[i][1]], colors_labels[i], label = f"Conditions: {quantiles0[i]} and {quantiles1[i]}")
                    ax0.plot([datetime.fromtimestamp(tlow[i][0]), datetime.fromtimestamp(tlow[i][1])], [flow[i][0], flow[i][1]], colors_labels[i])
                    #ax0.scatter(xnew1, ynew1, s= 20, c= 'r')
                    #ax0.scatter(xnew2, ynew2, s= 20, c= 'r')

                # Replace masked values with those along connecting lines - only for 95% & 5% which is last
                new_lists_1 = []
                new_lists_2 = []

                for x, y in zip(xnew10, ynew1):
                    new_lists_1.append([x, y])

                new_lists_1 = np.asarray(new_lists_1)
                new_lists_1[:, 0] = [mdates.date2num(datetime.fromtimestamp(i)) for i in new_lists_1[:,0]]

                for x, y in zip(xnew20, ynew2):
                    new_lists_2.append([x,y])

                new_lists_2 = np.asarray(new_lists_2)
                new_lists_2[:, 0] = [mdates.date2num(datetime.fromtimestamp(i)) for i in new_lists_2[:,0]]

                #ax0.set_xlim([datetime.fromtimestamp(tup[0][0]), datetime.fromtimestamp(tlow[0][1])])
                ax0.legend(loc = 'lower left', prop={'size': 8})
                ax0.set_xlim([mdates.date2num(pd.Timestamp(newstarts[ind])), mdates.date2num(pd.Timestamp(newends[ind]))])
                
                # REDefine quantiles / paths PER FIGURE
                quantiles00 = np.array([0.75, 0.8, 0.85, 0.9, 0.95])

                ## AXIS 1 - JOINT POLYGONS
                # START W/ LIST 2
                # TIME BOUNDS
                # Pick b/w dif quantile
                perc_poly0 = 0.95
                q = np.where(quantiles00 == perc_poly0)[0][0]
    
                # Mask values b/w 95% initialization points 
                poly0_low_bound = np.where((saved_polys[0][:, 0] == tlow[q][0]) & (saved_polys[0][:, 1] == flow[q][0]))[0][0]
                poly0_up_bound = np.where((saved_polys[0][:, 0] == tup[q][0]) & (saved_polys[0][:, 1] == fup[q][0]))[0][0]
                mask0_ind = np.arange(poly0_low_bound+1, poly0_up_bound, 1)

                # Create a mask for certain points
                mask0 = np.zeros(np.shape(saved_polys[0])[0], dtype = bool)
                mask0[mask0_ind] = True
                masked_polys0 = saved_polys[0][~mask0]

                # Mask values b/w 5% initialization points 
                poly1_low_bound = np.where((saved_polys[1][:, 0] == tlow[q][1]) & (saved_polys[1][:, 1] == flow[q][1]))[0][0]
                poly1_up_bound = np.where((saved_polys[1][:, 0] == tup[q][1]) & (saved_polys[1][:, 1] == fup[q][1]))[0][0]
                mask1_ind = np.arange(poly1_up_bound+1, poly1_low_bound, 1)

                # Create a mask for certain points
                mask1 = np.zeros(np.shape(saved_polys[1])[0], dtype = bool)
                mask1[mask1_ind] = True
                masked_polys1 = saved_polys[1][~mask1]

                # Attach poly1_low to poly0_low
                ind_poly0_low = np.where((masked_polys0[:, 0] == tlow[q][0]) & (masked_polys0[:, 1] == flow[q][0]))[0][0]
                ind_poly0_up = np.where((masked_polys0[:, 0] == tup[q][0]) & (masked_polys0[:, 1] == fup[q][0]))[0][0]

                ind_poly1_low = np.where((masked_polys1[:, 0] == tlow[q][1]) & (masked_polys1[:, 1] == flow[q][1]))[0][0]
                ind_poly1_up = np.where((masked_polys1[:, 0] == tup[q][1]) & (masked_polys1[:, 1] == fup[q][1]))[0][0]

                # Reorder poly1 so that index 1 is at lowest bound
                masked_polys1 = np.concatenate([masked_polys1[ind_poly1_low:], masked_polys1[:ind_poly1_low]])
                lol = np.insert(masked_polys0, ind_poly0_low+1, masked_polys1, axis = 0)
                lol = [lol]
                
                # PLOT
                ax1 = plot_flux2(ax1, data_start, data_end, file, colour_in = lol, fontsize=fontsize)
        
                # Print plots
                fig.tight_layout()
                plt.show()
        
    if plot["spectrogram_joint_polygons"]:
        #read in data
        fp = output_data_fp + '/trajectory/trajectory{}.csv'.format(year)
        mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
        intpwr_df=pd.read_csv(output_data_fp + '/intpwr/intpwr_withtraj{}.csv'.format(year), parse_dates=['datetime_ut'])
       #set parameters for plot

        num_panels=2
        fontsize=18
        fig_height = 8
        fig_width = 10
       #define figure
        plt.ioff()
        fig,ax = plt.subplots(num_panels,1,figsize=(fig_width, fig_height))
       #set width and height space between subplots
        fig.subplots_adjust(wspace=0.1,hspace=0.1)
       #plot flux
        saved_polys=None
        ax[0]=plot_flux2(ax[0], data_start, data_end, file,colour_in=None, fontsize=fontsize)


        polygon_fp=LFE_data_directory + "2004001_2017258_joint_dst_catalogue.json"
        print("do we read polygon file path correctly?")
        print(polygon_fp)
        saved_polys = get_polygons(polygon_fp,data_start, data_end)
       # breakpoint()

        ax[1]=plot_flux2(ax[1], data_start, data_end, file, colour_in=saved_polys,fontsize=fontsize)
          
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
    fp = output_data_fp + '/trajectory/trajectory{}.csv'.format(year)
    mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
    print(mag_df)
    intpwr_df=pd.read_csv(output_data_fp + '/intpwr/intpwr_withtraj{}.csv'.format(year))
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

# Additional stylistic changes for JOINT POLYGON PLOTTING
def plot_flux2(ax,time_view_start, time_view_end, file, colour_in=None, frequency_lines=None,fontsize=18):
    
    #Load data from .sav file
    time, freq, flux = extract_data(file, time_view_start=time_view_start,\
                                    time_view_end=time_view_end,val='s')
    #Parameters for colorbar
    #This is the function that does flux normalisation based on s/c location
    #vmin, vmax=plt_func.flux_norm(time[0], time[-1])   #change from log10 to actual values.
    clrmap ='gist_gray' ## Changed from 'viridis'
    vmin = np.quantile(flux[flux > 0.], 0.05)
    vmax = np.quantile(flux[flux > 0.], 0.95)
    scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    #Make figure
    fig = plt.figure()
    im=ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=clrmap,  shading='auto')
    ax.set_yscale('log')
    ax.set_facecolor("#F4ECFF") ## Changed from NONE
    
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
            ax.add_patch(Polygon(shape_, color='blue', linestyle='solid',linewidth=2, alpha=1, fill=False)) ## Changed from color = 'black', linestyle = 'dashed', linewidth = 2
        
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

# Comparison func - total lfes match joint?
def comparison(date_start, date_end, LFE_data_directory, lfe_unet_data, lfe_joined_list):
    # define number of polygons in time range
    data_start = pd.Timestamp(date_start)
    data_end = pd.Timestamp(date_end)
    
    # Define Number of NOT JOINED LFEs - supposedly TOTAL
    polygon_fp=LFE_data_directory + "2004001_2017258_catalogue.json"
    full_df = pd.read_csv(LFE_data_directory + lfe_unet_data)
    full_df = full_df.loc[(full_df['start'] >= date_start) & (full_df['start'] <= date_end)]
    saved_polys = get_polygons(polygon_fp, data_start, data_end)
    poly_num = len(saved_polys)
    print('Uncalibrated, total polygons:', poly_num)

    # define number of JOINED LFEs given in 'joined....csv' file 
    joint = pd.read_csv(LFE_data_directory + lfe_joined_list)
    joint_min = joint.loc[(joint['start'] >= date_start) & (joint['start'] <= date_end)]
    joint_num = np.shape(joint_min)[0]
    print('Joined Polygons:', joint_num)

    '''
    if poly_num == joint_num:
        print('No polygons need joining!')
    else:
        print('We need to join some polygons!')
    '''
    
    return full_df, saved_polys, joint_min, poly_num, joint_num


# In Europe, daylight savings is observed between the LAST Sunday of March and the LAST Sunday of October
# In the United States, daylight savings is observed between the SECOND Sunday of March and the FIRST Sunday of November

def find_last_sunday(date):
    days_left = 6 - date.weekday()
    # .weekday() - datetime.date class function that returns an integer value that corresponds to the day of the week
    # I.e. 0 - Monday, 1 - Tuesday, 2 - Wednesday, 3 - Thursday, 4 - Friday, 5 - Saturday, and 6 - Sunday
    if days_left != 0:
        date += timedelta(days = days_left) # Add number of days to obtain LAST SUNDAY of March & October
    return date 

# Defines the months of Daylight Saving Time in Europe (apart from Iceland, Belarus, Turkey, and Russia) for a specific year
# European Summer Time begins at 01:00 UTC on the LAST Sunday of March (between 25 and 31 of March) and ends at 01:00 UTC on the LAST Sunday of October (between 25 and 31 of October)
def europe_dst_range(year):
    DST_start = datetime(year = 1, month = 3, day = 25, hour = 1, minute = 0)
    DST_end = datetime(year = 1, month = 10, day = 25, hour = 1, minute = 0)

    dst_start = find_last_sunday(DST_start.replace(year = year)) # Find explicit dates of DST for specific input year
    dst_end = find_last_sunday(DST_end.replace(year = year))

    return dst_start, dst_end # Defines datetime limits for .json file corrections per year (from 2004 to 2017 - the time period we have Cassini data for)


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