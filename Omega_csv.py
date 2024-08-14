import configparser
import pandas as pd

config = configparser.ConfigParser()
config.read('config_LFE_stats.ini')   
data_directory = config['filepaths']['LFE_data_directory'] # Directory where SN_ms_tot_V2.0.csv, SN_d_tot_V2.0.csv, and LFEs_joined.csv are located
input_data_fp= config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']

# Define joined polygon UNET
lfe_joined_list = "LFEs_joined.csv"
one_min_resol_joined = "LFEs_joined_ephemeris.csv" # joined LFE list coverage

join_unet = pd.read_csv(data_directory + lfe_joined_list, index_col = 0) 
LFE_join = pd.read_csv(data_directory + one_min_resol_joined)

join_unet['Range'] = LFE_join['R_KSM']
join_unet['subLST'] = LFE_join['subLST']
join_unet['subLat'] = LFE_join['subLat']
join_unet['x_ksm'] = LFE_join['x_KSM']
join_unet['y_ksm'] = LFE_join['y_KSM']
join_unet['z_ksm'] = LFE_join['z_KSM']

join_unet.to_csv(output_data_fp + "/LFEs_joined_times_range_lst_lat.csv") 
join_unet.to_csv(input_data_fp + "/LFEs_joined_times_range_lst_lat.csv") 