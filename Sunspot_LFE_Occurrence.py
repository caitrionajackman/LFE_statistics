import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from matplotlib.transforms import ScaledTranslation
import configparser

def moving_median(x, y, window_size):
    y_new = []
    # Define window_size, begin at x[0] and find median of y[0:0+window_size]
    for i in range(len(x)):
        if i + window_size < len(x):
            y_new.append(np.median(y[i:i + window_size]))
        else:
            y_new.append(np.median(y[i:]))
    return y_new

def Sunspot_LFE_Relation(data_directory):
    # Load in Sunspot Data
    sunspots = pd.read_csv(data_directory + "SN_d_tot_V2.0.csv",  delimiter = ";")
    sunspots.columns = ['year', 'month', 'day', 'frac_year', 'count', 'std', 'observations', 'def/prov']

    # Daily sunspot number exists
    sunspots = sunspots.iloc[np.where(sunspots['count'] > 0)[0]].reset_index()
    years = np.unique(sunspots['year'])
    months = np.unique(sunspots['month'])

    # Find Monthly Average Sunspot Count
    monthly_avg = pd.DataFrame(list(), columns = ['sunspot_monthly', 'year', 'month'])
    for year in years:
        for month in months:
            sunspot_count = np.mean(sunspots.loc[(sunspots['month'] == month) & (sunspots['year'] == year)]['count'])
            data = []
            data.append([sunspot_count, year, month])
            new = pd.DataFrame(data, columns = ['sunspot_monthly', 'year', 'month'])
            monthly_avg = pd.concat([monthly_avg, new], ignore_index = True, axis = 0)
        
    monthly_avg = monthly_avg.iloc[np.where(monthly_avg['sunspot_monthly'] != 0)].reset_index()
    monthly_avg['year_frac'] = monthly_avg['year'] + (monthly_avg['month'] / 12)

    # Read in 13-month smoothed sunspot count
    smoothed_sun = pd.read_csv(data_directory + "SN_ms_tot_V2.0.csv", delimiter = ";")
    smoothed_sun.columns = ['year', 'month', 'year_frac', 'monthly_smoothed_count', 'std', 'num_obs', 'definitive']
    smoothed_sun = smoothed_sun.loc[(smoothed_sun['monthly_smoothed_count'] > 0) & (smoothed_sun['num_obs'] > 0) & (smoothed_sun['definitive'] == 1)].reset_index()

    # Duration of Joined LFEs
    joint_LFEs = pd.read_csv(data_directory + "LFEs_joined.csv", index_col = 0)

    # Initialize new columns
    joint_LFEs['start_year']= np.ones(np.shape(joint_LFEs)[0])
    joint_LFEs['end_year'] =np.ones(np.shape(joint_LFEs)[0])
    joint_LFEs['start_month'] = np.ones(np.shape(joint_LFEs)[0])
    joint_LFEs['end_month'] = np.ones(np.shape(joint_LFEs)[0])

    for i in range(np.shape(joint_LFEs)[0]):
        joint_LFEs['start_year'].iloc[i] = pd.to_datetime(joint_LFEs['start'].iloc[i]).year
        joint_LFEs['end_year'].iloc[i] = pd.to_datetime(joint_LFEs['end'].iloc[i]).year
        joint_LFEs['start_month'].iloc[i] = pd.to_datetime(joint_LFEs['start'].iloc[i]).month
        joint_LFEs['end_month'].iloc[i] = pd.to_datetime(joint_LFEs['end'].iloc[i]).month

    # Duration event plotted at end date
    joint_LFEs['end_frac'] = joint_LFEs['end_year'] + (joint_LFEs['end_month'] / 12)
    joint_LFEs['start_frac'] = joint_LFEs['start_year'] + (joint_LFEs['start_month'] / 12)
    joint_LFEs['days'] = ((joint_LFEs['duration'] / 60) / 60) /24
    joint_LFEs['hours'] = ((joint_LFEs['duration'] / 60) / 60) 

    # Yearly average of LFE duration
    year_avg = pd.DataFrame(list(), columns = ['yearly_avg', 'year'])
    years = np.unique(joint_LFEs['start_year'])
    months = np.unique(joint_LFEs['start_month'])

    monthly_aver = []
    x_val = []
    for year in years:
        for month in months:
            val = np.mean(joint_LFEs.loc[(joint_LFEs['start_year'] == year) & (joint_LFEs['start_month'] == month)]['hours'])
            if np.isnan(val) == True:
                pass
            else:
                x_val.append(year + (month / 12))
                monthly_aver.append(np.mean(joint_LFEs.loc[(joint_LFEs['start_year'] == year) & (joint_LFEs['start_month'] == month)]['hours']))

    smoothed = moving_median(x_val, monthly_aver, 12)

    # Occurrence of LFEs (binned into ~ 1 month increments, not 10 days)
    num_LFEs = pd.DataFrame(list(), columns = ['year', 'month', 'num'])
    years = np.unique(joint_LFEs['start_year'])
    months = np.unique(joint_LFEs['start_month'])

    time = []
    events = []
    for year in years:
        for month in months:
            num_events = np.shape(joint_LFEs.loc[(joint_LFEs['start_year'] == year) & (joint_LFEs['start_month'] == month)])[0]
            if num_events == 0:
                pass
            else:
                time.append(year + (month/12))
                events.append(num_events)

    occ_lfe = moving_median(time, events, 20)

    fig, ax = plt.subplots(3, figsize = (12,12))

    # SUNSPOT PLOT
    ax[0].set_title('Solar Sunspot Count')
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Sunspot Number")
    ax[0].scatter(monthly_avg['year_frac'], monthly_avg['sunspot_monthly'],s = 7, color = 'grey')
    ax[0].plot(monthly_avg['year_frac'], monthly_avg['sunspot_monthly'],linewidth = 1, color = 'grey', label = 'Average Monthly Sunspot Number')
    ax[0].plot(smoothed_sun['year_frac'], smoothed_sun['monthly_smoothed_count'], linewidth = 1, color = 'blue', label = '13-month Smoothed Sunspot Number')
    ax[0].set_xlim(2004, 2018)
    ax[0].set_xticks(np.arange(2004, 2018, 1))
    ax[0].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    ax[0].set_ylim(0, np.max(monthly_avg['sunspot_monthly'].loc[(monthly_avg['year'] >=2004) &(monthly_avg['year'] < 2018)]))
    ax[0].legend()

    # Add i), ii), iii)
    ax[0].text(0.0, 1.0, 'i)', transform=(
                ax[0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')

    # DURATION PLOT
    ax[1].set_title("Duration of Joint LFEs ")
    ax[1].set_xlabel("Year")
    ax[1].set_ylabel("Duration of LFE (Hours)")
    ax[1].set_xticks(np.arange(2004, 2018, 1))
    ax[1].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    ax[1].set_yscale('log')
    ax[1].scatter(joint_LFEs['start_frac'], joint_LFEs['hours'], s = 5, marker = '^', color = 'gray')
    ax[1].plot(x_val, smoothed, color = 'blue', linewidth = 2, label = '1-Year Moving Median Smoothed LFE Count')

    '''
    # Add inset
    x1, x2, y1, y2 = np.min(joint_LFEs['start_frac']), np.max(joint_LFEs['start_frac']), np.min(monthly_aver), np.max(monthly_aver)
    axins = ax[1].inset_axes(
        [0.25, 0.75, 0.75, 0.25], xlim=(x1, x2), ylim=(y1, y2), xticklabels = [], yticklabels = [])
    axins.set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    axins.set_xticks(np.arange(2004, 2018, 1))


    axins.scatter(joint_LFEs['start_frac'], joint_LFEs['days'], s = 3, marker = '^', color = 'gray')
    axins.plot(x_val, monthly_aver, color = 'blue', linewidth = 1.5)
    axins.plot(x_val, smoothed, color = 'black', linewidth = 1.5)

    ax[1].indicate_inset_zoom(axins, edgecolor="black")
    '''
    ax[1].set_xlim(np.min(joint_LFEs['start_frac']), np.max(joint_LFEs['start_frac']))
    ax[1].set_ylim(0, 300)
    ax[1].legend()

    # Add i), ii), iii)
    ax[1].text(0.0, 1.0, 'ii)', transform=(
                ax[1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')

    # OCCURRENCE PLOT
    ax[2].set_title("Occurrence Rate for LFEs")
    ax[2].scatter(time, events, s = 7, color = 'grey')
    ax[2].plot(time, events, color = 'grey', linewidth = 1, label = '25 Day Average LFE Occurrence Count')
    ax[2].plot(time, occ_lfe, color = 'blue', label = '1-Year Moving Median Smoothed LFE Occurrence Count')
    ax[2].set_xlabel("Year")
    ax[2].set_xticks(np.arange(2004, 2018, 1))
    ax[2].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    ax[2].set_ylabel("Occurrence (~ every 25 days)")
    ax[2].set_xlim(2004, 2018)
    ax[2].legend()

    # Add i), ii), iii)
    ax[2].text(0.0, 1.0, 'iii)', transform=(
                ax[2].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')

    fig.tight_layout()
    plt.show()

# Actual Code
config = configparser.ConfigParser()
config.read('config_LFE_stats.ini')   
LFE_data_directory= config['filepaths']['LFE_data_directory'] # Directory where SN_ms_tot_V2.0.csv, SN_d_tot_V2.0.csv, and LFEs_joined.csv are located

Sunspot_LFE_Relation(data_directory=LFE_data_directory)