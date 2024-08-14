import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from matplotlib.transforms import ScaledTranslation
import configparser

# KSM is a cartesian Saturn-centric coordinate system where X points from Saturn to the Sun, 
# the X-Z plane contains Saturn's centered magnetic dipole axis, M, and Y completes 
# right handed set.

# Is there any relationship between Cassini's Position and LFE Occurrence/Distribution?

def Cassini_Pos_LFE(data_directory):
    joint_LFEs = pd.read_csv(data_directory + "LFEs_joined.csv", index_col = 0)
    joint_LFEs['hours'] = ((joint_LFEs['duration'] / 60) / 60) 

    # Positional-Dependence PLOTS
    fig, ax = plt.subplots(2, 3, figsize = (20, 12))

    # As a function of x ksm
    ax[0,0].set_title("Duration of LFEs as a function of X Cassini-KSM")
    ax[0,0].set_xlabel("X position (KSM)")
    ax[0,0].set_ylabel("Duration of LFE (Hours)")
    #ax[0].set_xticks(np.arange(2004, 2018, 1))
    #ax[0].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    ax[0,0].set_yscale('log')
    ax[0,0].scatter(joint_LFEs['x_ksm'], joint_LFEs['hours'], s = 5, marker = '^', color = 'gray')
    #ax[0].plot(x_val_top, smoothed_top, color = 'blue', linewidth = 2, label = '1-Year Moving Median Smoothed Long LFE Count')
    # Add i), ii), iii)
    ax[0,0].text(0.0, 1.0, 'i)', transform=(
                ax[0,0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')


    # As a function of x ksm
    ax[1,0].set_title("Occurrence of LFEs as a function of X Cassini-KSM")
    ax[1,0].set_xlabel("X position (KSM)")
    ax[1,0].set_ylabel("Frequency Count of LFEs")
    #ax[0].set_xticks(np.arange(2004, 2018, 1))
    #ax[0].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    #ax[1,0].set_yscale('log')
    ax[1,0].hist(joint_LFEs['x_ksm'], color = 'gray', bins = 100)
    #ax[0].plot(x_val_top, smoothed_top, color = 'blue', linewidth = 2, label = '1-Year Moving Median Smoothed Long LFE Count')
    # Add i), ii), iii)
    ax[1,0].text(0.0, 1.0, 'ii)', transform=(
                ax[1,0].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')

    # Add inset
    x1, x2, y1, y2 = -100, 100, 0, 700
    axins = ax[1,0].inset_axes(
        [0.5, 0.5, 0.5, 0.5], xlim=(x1, x2), ylim=(y1, y2), xticklabels = [], yticklabels = [])
    axins.set_xticklabels(np.arange(-100, 100, 25), rotation = 45)
    axins.set_xticks(np.arange(-100, 100, 25))

    axins.hist(joint_LFEs['x_ksm'], color = 'gray', bins = 100)

    ax[1,0].indicate_inset_zoom(axins, edgecolor="black")


    # As a function of y ksm
    ax[0,1].set_title("Duration of LFEs as a function of Y Cassini-KSM")
    ax[0,1].set_xlabel("Y position (KSM)")
    ax[0,1].set_ylabel("Duration of LFE (Hours)")
    #ax[0].set_xticks(np.arange(2004, 2018, 1))
    #ax[0].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    ax[0,1].set_yscale('log')
    ax[0,1].scatter(joint_LFEs['y_ksm'], joint_LFEs['hours'], s = 5, marker = '^', color = 'gray')
    #ax[0].plot(x_val_top, smoothed_top, color = 'blue', linewidth = 2, label = '1-Year Moving Median Smoothed Long LFE Count')
    # Add i), ii), iii)
    ax[0,1].text(0.0, 1.0, 'iii)', transform=(
                ax[0,1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')


    # As a function of y ksm
    ax[1,1].set_title("Occurrence of LFEs as a function of Y Cassini-KSM")
    ax[1,1].set_xlabel("Y position (KSM)")
    ax[1,1].set_ylabel("Frequency Count of LFEs")
    #ax[0].set_xticks(np.arange(2004, 2018, 1))
    #ax[0].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    #ax[1,0].set_yscale('log')
    ax[1,1].hist(joint_LFEs['y_ksm'], color = 'gray', bins = 100)
    #ax[0].plot(x_val_top, smoothed_top, color = 'blue', linewidth = 2, label = '1-Year Moving Median Smoothed Long LFE Count')
    # Add i), ii), iii)
    ax[1,1].text(0.0, 1.0, 'iv)', transform=(
                ax[1,1].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')

    # Add inset
    x1, x2, y1, y2 = -200, 50, 0, 1250
    axins = ax[1,1].inset_axes(
        [0.05, 0.5, 0.5, 0.5], xlim=(x1, x2), ylim=(y1, y2), xticklabels = [], yticklabels = [])
    axins.set_xticklabels(np.arange(-200, 50, 25), rotation = 45)
    axins.set_xticks(np.arange(-200, 50, 25))

    axins.hist(joint_LFEs['y_ksm'], color = 'gray', bins = 100)

    ax[1,1].indicate_inset_zoom(axins, edgecolor="black")



    # As a function of z ksm
    ax[0,2].set_title("Duration of LFEs as a function of Z Cassini-KSM")
    ax[0,2].set_xlabel("Z position (KSM)")
    ax[0,2].set_ylabel("Duration of LFE (Hours)")
    #ax[0.set_xticks(np.arange(2004, 2018, 1))
    #ax[0].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    ax[0,2].set_yscale('log')
    ax[0,2].scatter(joint_LFEs['z_ksm'], joint_LFEs['hours'], s = 5, marker = '^', color = 'gray')
    #ax[0].plot(x_val_top, smoothed_top, color = 'blue', linewidth = 2, label = '1-Year Moving Median Smoothed Long LFE Count')
    # Add i), ii), iii)
    ax[0,2].text(0.0, 1.0, 'v)', transform=(
                ax[0,2].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')


    # As a function of z ksm
    ax[1,2].set_title("Occurrence of LFEs as a function of Z Cassini-KSM")
    ax[1,2].set_xlabel("Z position (KSM)")
    ax[1,2].set_ylabel("Frequency Count of LFEs")
    #ax[0].set_xticks(np.arange(2004, 2018, 1))
    #ax[0].set_xticklabels(np.arange(2004, 2018, 1), rotation = 45)
    #ax[1,0].set_yscale('log')
    ax[1,2].hist(joint_LFEs['z_ksm'], color = 'gray', bins = 100)
    #ax[0].plot(x_val_top, smoothed_top, color = 'blue', linewidth = 2, label = '1-Year Moving Median Smoothed Long LFE Count')
    # Add i), ii), iii)
    ax[1,2].text(0.0, 1.0, 'vi)', transform=(
                ax[1,2].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=15, va='bottom', fontfamily='serif')


    fig.tight_layout()
    plt.show()

# Actual Code
config = configparser.ConfigParser()
config.read('config_LFE_stats.ini')   
LFE_data_directory = config['filepaths']['LFE_data_directory'] # wherever file "LFEs_joined.csv" is locatied

Cassini_Pos_LFE(data_directory = LFE_data_directory) # ----- EX. /Sample_Visualizations/Cassini_Position_LFE.jpeg