import numpy as np

import matplotlib.dates as mdates
from matplotlib.path import Path

from scipy.io import readsav

def get_sav_data(year):
    """Select sav data for a chosen year.
    
    Parses through an SKR year file to obtain the time, frequency and flux data.
    
    Parameters
    ----------
    year: int
        Year of the file wanted.
    
    Returns
    -------
    time: numpy.array
        Time series in 3 minute step.

    freq: numpy.array
        Midpoint values of Cassini frequency bins.

    flux: numpy.array (freq.shape, time.shape)
        Magnetic flux values for the chosen year.
    """
    file_skr = f'raw_SKR/SKR_{year}_CJ.sav'
    raw_skr = readsav(file_skr)
    flux, time_doy, freq = raw_skr['s'].copy(), raw_skr['t'], raw_skr['f']
    flux[flux == 0] = np.nan  # replace 0 with nans

    time = (time_doy * 24 * 3600).astype('timedelta64[s]') + np.datetime64('1997') - np.timedelta64(1, 'D') + np.timedelta64(1, 's')
    time = time.astype('datetime64[m]')
    time = time.astype('datetime64[s]')

    return time, freq, flux

def get_poly_coords(year, poly_tf_list):
    """Select polygons for a chosen year from a list of polygons.

    Parameters
    ----------
    year: int
        Year chosen
    
    poly_tf_list: list
        List of polygon time-frequency coordinates
    
    Returns
    -------
    polygon_coordinates: list
        List of polygon time-frequency coordinates for the chosen year.
    """
    year_range = 1

    year_low, year_high = mdates.date2num(np.datetime64(f'{year}')), mdates.date2num(np.datetime64(f'{year + year_range}'))

    polygon_coordinates = []
    for poly_coords in poly_tf_list:
    
        poly_time = poly_coords[:,0]
        time_mask = (poly_time >= year_low) & (poly_time <= year_high)

        if any(time_mask):
            polygon_coordinates.append(poly_coords[time_mask])

    return polygon_coordinates