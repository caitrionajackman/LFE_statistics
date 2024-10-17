from dst_correction import *
from Radio_visualisations_OG_wchanges import *
from LFE_statistics import *
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
import json


def main():
    plt.rcParams.update({'font.size': 12})

    config = configparser.ConfigParser()
    config.read('config_LFE_stats.ini')
    input_data_fp = config['filepaths']['input_data']
    output_data_fp= config['filepaths']['output_data']
    LFE_data_directory= config['filepaths']['LFE_data_directory']
    polygon_fp = LFE_data_directory + "2004001_2017258_catalogue.json" # Found at Zenodo link - too large for Github

    lfe_unet_data = "lfe_detections_unet_2874.csv" #Processed using findDetectionPositions.py (Elizabeth's updated UNet output file)
    lfe_training_data = "lfe_detections_training.csv"
    lfe_joined_list = "LFEs_joined.csv" # Can be generated using LFE_statistics.py 
    
    #dates you would like to plot visualisations for year-month-day
    data_str_start = '2004-01-01'
    data_str_end = '2017-09-14'
    data_start=pd.Timestamp(data_str_start)
    data_end=pd.Timestamp(data_str_end)

    year = datetime.strftime(data_start, '%Y')
    #print(data_start)

    if year == '2017':
        file = input_data_fp + '/SKR_spectrogram_data/SKR_2017_001-258_CJ.sav'
    else: 
        file = input_data_fp + '/SKR_spectrogram_data/SKR_{}_CJ.sav'.format(year)

    saved_polys=None

    unet=True

    #Read in LFE list (output of Elizabeth's U-Net run on full Cassini dataset)
    #print('Reading in the LFE list')
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

    def get_polygons(polygon_fp, start, end):
        print("are we getting here and finding polygons?")    
        print(start, end)    
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
    
    # Comparison func - total lfes match joint?
    def comparison(date_start, date_end):
        # define number of polygons in time range
        data_start = pd.Timestamp(date_start)
        data_end = pd.Timestamp(date_end)
        
        # Define Number of NOT JOINED LFEs - AKA TOTAL number of defined polygons in initial .json file
        config = configparser.ConfigParser()
        config.read('config_LFE_stats.ini')
        LFE_data_directory= config['filepaths']['LFE_data_directory']

        polygon_fp = LFE_data_directory + "2004001_2017258_catalogue.json"
        full_df = pd.read_csv(LFE_data_directory + "lfe_detections_unet_2874.csv")
        full_df = full_df.loc[(full_df['start'] >= date_start) & (full_df['start'] <= date_end)]
        saved_polys = get_polygons(polygon_fp, data_start, data_end)
        poly_num = len(saved_polys)
        print('Uncalibrated, total polygons:', poly_num)

        # define number of JOINED LFEs given in 'joined....csv' file 
        joint = pd.read_csv(LFE_data_directory + "LFEs_joined.csv")
        joint_min = joint.loc[(joint['start'] >= date_start) & (joint['start'] <= date_end)]
        joint_num = np.shape(joint_min)[0]
        print('Joined Polygons:', joint_num)
        
        return full_df, saved_polys, joint_min, poly_num, joint_num
    
    # Do we even need to join? 
    total_df, saved_polys, joint_polys, poly_num, joint_num = comparison(data_str_start, data_str_end)
    
    ##########################################################################################################
    # NEW POLYGON ARRAY
    polygon_array = []

    if poly_num == joint_num: # This means that no polygons need joining 
        final_vertices = saved_polys
        print("we don't need to join any polygons!")
    else: # We need to create new array with collection of JOINT vertices
        print('we have to join some polygons!')
        i = 0
        while i < np.shape(joint_polys)[0]: # Length of new saved_polys array
            print(i) # out of 4553
            start_joint = joint_polys['start'][i]
            end_joint = joint_polys['end'][i]
            print(start_joint, end_joint)

            total_ind = np.where((total_df['start'] == start_joint) & (total_df['end'] == end_joint))
            num_spots = len(total_ind[0])

            if num_spots == 1: # No need to JOIN polygons - exist in original .json file
                polygon_vertices = saved_polys[total_ind[0][0]]
                polygon_array.append(polygon_vertices)
            else:
                print('we need to add polygon w/ dates: ', start_joint, 'to ', end_joint)
                quantiles0 = [0.95]
                quantiles1 = [0.05]

                # initial new start & end times of joining
                data_start=pd.Timestamp(f'{start_joint}') 
                data_end=pd.Timestamp(f'{end_joint}')
                ind_polys = get_polygons(polygon_fp, data_start, data_end)

                tup = []
                tlow = []
                fup = []
                flow = []
                conditions = []

                for quant0, quant1 in zip(quantiles0, quantiles1):
                    #print('Quantiles:', quant0, quant1)
                    # Polygon 0 - fmax, fmin, tmax
                    cond = np.quantile(ind_polys[0][:, 0], quant0)

                    # Find frequencies that pass threshold condition on POLYGON 0
                    max_f0 = np.max(ind_polys[0][np.where(ind_polys[0][:, 0] >= cond), 1][0]) # Frequency MAX
                    new_polys = ind_polys[0][np.where(ind_polys[0][:, 0] >= cond)] # BAD POINTS
                    old_polys = ind_polys[0][np.where(ind_polys[0][:, 0] < cond)] # GOOD POINTS
                    delete_polys_len0 = len(new_polys)
                    ind_max_t = np.where(max_f0 == ind_polys[0][np.where(ind_polys[0][:, 0] >= cond),1])[1][0] # Index of time MAXIMUM from where to draw initial line
                    max_t0_u = new_polys[ind_max_t, 0] # Actual T MAX Upper VALUE
                    #print('Maximum t & Maximum freq for LEFT POLY:', max_t0_u, max_f0)

                    min_f0 = np.min(ind_polys[0][np.where(ind_polys[0][:, 0] >= cond), 1][0]) # Frequency min
                    ind_min_t = np.where(min_f0 == ind_polys[0][np.where(ind_polys[0][:, 0] >= cond),1])[1][0] # Index of time MINIMUM from where to draw initial line
                    max_t0_l = new_polys[ind_min_t, 0] # Actual T MAX Lower VALUE
                    #print('Maximum t & Minimum freq for LEFT POLY:', max_t0_l, min_f0)


                    # Polygon 1 - fmax, fmin, tmin
                    cond1 = np.quantile(ind_polys[1][:, 0], quant1)

                    # Find frequencies that pass threshold condition on POLYGON 1
                    max_f1 = np.max(ind_polys[1][np.where(ind_polys[1][:, 0] <= cond1), 1][0]) # Frequency MAX
                    new_polys_1 = ind_polys[1][np.where(ind_polys[1][:, 0] <= cond1)] # BAD POINTS
                    old_polys_1 = ind_polys[1][np.where(ind_polys[1][:, 0] > cond1)] # GOOD POINTS
                    delete_polys_len1 = len(new_polys_1)
                    ind_max_t1 = np.where(max_f1 == ind_polys[1][np.where(ind_polys[1][:, 0] <= cond1),1])[1][0]  # Index of time MAXIMUM from where to draw initial line
                    min_t1_u = new_polys_1[ind_max_t1, 0]  # Actual T MIN upper VALUE
                    #print('Minimum t & Maximum freq for RIGHT POLY:', min_t1_u, max_f1)

                    min_f1 = np.min(ind_polys[1][np.where(ind_polys[1][:, 0] <= cond1), 1][0]) # Frequency min
                    ind_min_t1 = np.where(min_f1 == ind_polys[1][np.where(ind_polys[1][:, 0] <= cond1),1])[1][0] # Index of time MINIMUM from where to draw initial line
                    min_t1_l = new_polys_1[ind_min_t1, 0]  # Actual T MIN Lower VALUE
                    #print('Minimum t & Minimum freq for RIGHT POLY:', min_t1_l, min_f1)

                    # Apped to appropriate list
                    tup.append([max_t0_u, min_t1_u])
                    tlow.append([max_t0_l, min_t1_l])
                    fup.append([max_f0, max_f1])
                    flow.append([min_f0, min_f1])
                    conditions.append([cond, cond1])

                # REDefine quantiles / paths PER FIGURE
                quantiles00 = np.array([0.95])

                ## AXIS 1 - JOINT POLYGONS
                # TIME BOUNDS
                # Pick b/w dif quantile
                q = np.where(quantiles00 == 0.95)[0][0]

                # Mask values b/w 95% initialization points 
                poly0_low_bound = np.where((ind_polys[0][:, 0] == tlow[q][0]) & (ind_polys[0][:, 1] == flow[q][0]))[0][0]
                poly0_up_bound = np.where((ind_polys[0][:, 0] == tup[q][0]) & (ind_polys[0][:, 1] == fup[q][0]))[0][0]
                mask0_ind = np.arange(poly0_low_bound+1, poly0_up_bound, 1)

                # Create a mask for certain points
                mask0 = np.zeros(np.shape(ind_polys[0])[0], dtype = bool)
                mask0[mask0_ind] = True
                masked_polys0 = ind_polys[0][~mask0]

                # Mask values b/w 5% initialization points 
                poly1_low_bound = np.where((ind_polys[1][:, 0] == tlow[q][1]) & (ind_polys[1][:, 1] == flow[q][1]))[0][0]
                poly1_up_bound = np.where((ind_polys[1][:, 0] == tup[q][1]) & (ind_polys[1][:, 1] == fup[q][1]))[0][0]
                mask1_ind = np.arange(poly1_up_bound+1, poly1_low_bound, 1)

                # Create a mask for certain points
                mask1 = np.zeros(np.shape(ind_polys[1])[0], dtype = bool)
                mask1[mask1_ind] = True
                masked_polys1 = ind_polys[1][~mask1]

                # Attach poly1_low to poly0_low
                ind_poly0_low = np.where((masked_polys0[:, 0] == tlow[q][0]) & (masked_polys0[:, 1] == flow[q][0]))[0][0]
                ind_poly0_up = np.where((masked_polys0[:, 0] == tup[q][0]) & (masked_polys0[:, 1] == fup[q][0]))[0][0]

                ind_poly1_low = np.where((masked_polys1[:, 0] == tlow[q][1]) & (masked_polys1[:, 1] == flow[q][1]))[0][0]
                ind_poly1_up = np.where((masked_polys1[:, 0] == tup[q][1]) & (masked_polys1[:, 1] == fup[q][1]))[0][0]

                # Reorder poly1 so that index 1 is at lowest bound
                masked_polys1 = np.concatenate([masked_polys1[ind_poly1_low:], masked_polys1[:ind_poly1_low]])
                test = np.insert(masked_polys0, ind_poly0_low+1, masked_polys1, axis = 0)
                #test = [test]

                # Append to polygon array
                polygon_array.append(test)
            
            i += 1

    # DST corrections to joined polygons
    for polygon in range(len(polygon_array)):
        data_start = np.min([datetime.fromtimestamp(polygon_array[polygon][:, 0][i]) for i in range(len(polygon_array[polygon][:, 0]))])
        data_end = np.max([datetime.fromtimestamp(polygon_array[polygon][:, 0][i]) for i in range(len(polygon_array[polygon][:, 0]))])

        # This would allow for polygons across two years - December DEF not impacted by DST
        if data_start >= europe_dst_range(data_start.year)[0] and data_end <= europe_dst_range(data_start.year)[1]:
            for vertex in range(len(polygon_array[polygon])):
                polygon_array[polygon][vertex, 0] = (datetime.fromtimestamp(polygon_array[polygon][vertex, 0]) - pd.Timedelta(1, 'hour')).timestamp()
        else:
            pass

    # Create a new .json file with JOINED & DST-corrected Polygon Vertices
    # Joint polygons to dictionary
    new_dic = {}
    new_dic['features'] = []

    for shape in range(len(polygon_array)):
        if len(polygon_array[shape]) == 1:
            coords = [tuple(polygon_array[shape][0][j]) for j in range(len(polygon_array[shape][0]))]
            new_dic['features'].append({"geometry" : {"coordinates": [coords]}})
        else:
            coords = [tuple(list(polygon_array[shape][j])) for j in range(len(polygon_array[shape]))]
            new_dic['features'].append({"geometry" : {"coordinates": [coords]}})
    
    # Save .json file
    with open('2004001_2017258_joint_dst_catalogue.json', 'w') as outfile:
        json.dump(new_dic, outfile)


if __name__ == "__main__":
    main()


