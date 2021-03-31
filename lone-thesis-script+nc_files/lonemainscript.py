# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:33:40 2020

@author: lcmok
"""

#this library contains most functions/packages used in the data analysis of Lone Mokkenstorms thesis. Some scripts were adapted of the version by Wouter Neisingh: https://github.com/hcwinsemius/satellite-cookbook/tree/master/NSIDC-AMSRE

import pyproj
import xarray as xr
import numpy as np
import os
import subprocess
import shutil
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
import scipy
from math import factorial
from numpy import polyfit
from scipy import signal
import glob
from pandas.core.common import flatten
import seaborn as sns
from scipy import signal
from sklearn.metrics import r2_score

def proj_coord(coord, proj_in, proj_out):
    """
    Returns a packaged tuple (x, y) coordinate in projection proj_out
    from one packaged tuple (x, y) coordinatein projection proj_in
    Inputs:
        coord: tuple (x, y)
        proj_in: pyproj.Proj format projection
        proj_out: pyproj.Proj format projection
    Outputs:
        tuple (x, y)
        
    """
    x, y = coord
    xi, yi = pyproj.transform(proj_in, proj_out, x, y)
    return xi, yi
    

def proj_coords(coords, proj_in, proj_out):
    """
    project a list of coordinates, return a list.
    Inputs:
        coords: list of tuples (x, y)
        proj_in: pyproj.Proj format projection
        proj_out: pyproj.Proj format projection
    Outputs:
        list of tuples (x, y)
    
    """
    return [proj_coord(coord, proj_in, proj_out) for coord in coords]

def select_bounds(ds, bounds):
    """
    selects xarray ds along a provided bounding box
    assuming slicing should be done over coordinate axes x and y
    """
    
    xs = slice(bounds[0][0], bounds[1][0])
    ys = slice(bounds[1][1], bounds[0][1])
    # select over x and y axis
    return ds.sel(x=xs, y=ys)

def make_amsre_url(date, res, freq, HV, AD):
    """
    Prepares a url for AMSRE data to download.
    url_template - str url with placeholders for date (%Y.%m.%d), resolution (:d, km), date (%Y%j),
    frequency (str), polarisation ('H'/'V'), ascending/descending path ('A', 'D')
    """

    url_base = 'https://n5eil01u.ecs.nsidc.org/AMSA/AE_L2A.004/{:s}/'
    url_template = os.path.join(url_base,
                                'NSIDC-0630-EASE2_T{:s}km-AQUA_AMSRE-{:s}-{:s}{:s}-{:s}-{:s}-v1.3.nc') 
    datestr1 = date.strftime('%Y%j')
    datestr2 = date.strftime('%Y.%m.%d')
    #     if str(res) == '3.125':
    #         suffix = 'SIR-RSS'
    suffix = 'GRD-RSS'
    return url_template.format(datestr2, str(res), datestr1, freq, HV, AD, suffix)

def make_measures_url(date, res, freq, HV, AD, sat):
    """
    Prepares a url for Measures data to download.
    url_template - str url with placeholders for date (%Y.%m.%d), resolution (:d, km), date (%Y%j),
    frequency (str), polarisation ('H'/'V'), ascending/descending path ('A', 'D')
    """
    
    url_base = 'https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0630.001/{:s}/'
    
    if sat == 'AQUA':
        url_template = os.path.join(url_base,
                                'NSIDC-0630-EASE2_T{:s}km-AQUA_AMSRE-{:s}-{:s}{:s}-{:s}-{:s}-v1.3.nc') 
    elif sat == 'NIMBUS':
        url_template = os.path.join(url_base,
                                'NSIDC-0630-EASE2_T{:s}km-NIMBUS7_SMMR-{:s}-{:s}{:s}-{:s}-{:s}-v1.3.nc') 
    elif sat == 'DMSP':
        url_template = os.path.join(url_base,
                                'NSIDC-0630-EASE2_T{:s}km-F15_SSMI-{:s}-{:s}{:s}-{:s}-{:s}-v1.3.nc') 
    
    datestr1 = date.strftime('%Y%j')
    datestr2 = date.strftime('%Y.%m.%d')

    if sat == 'AQUA':
        if res == '25':
            suffix = 'GRD-RSS'
        elif res == '3.125':
            suffix = 'SIR-RSS'
    elif sat == 'NIMBUS':
        if res == '25':
            suffix = 'GRD-JPL'
        elif res == '3.125':
            suffix = 'SIR-JPL'
    elif sat == 'DMSP':
        suffix = 'GRD-CSU'
    return url_template.format(datestr2, str(res), datestr1, freq, HV, AD, suffix)

def make_measures_download(url, username, password):
    download_template = 'wget --http-user={:s} --http-password={:s} --load-cookies mycookies.txt --save-cookies mycookies.txt --keep-session-cookies --no-check-certificate --auth-no-challenge -r --reject "index.html*" -np -e robots=off {:s}'
    return download_template.format(username, password, url)

def download_measures(freq, res, HV, AD, date, username, password, sat):
    url = make_measures_url(date, res, freq, HV, AD, sat)
    download_string = make_measures_download(url, username, password)
    return url, subprocess.call(download_string.split(' '))  # call the download string in a command-line (use wget.exe! get it from online)

def download_measures_ts(freq, res, HV, AD, start_date, end_date, bounds, fn_out_prefix, username, password, sat):
    """
    Downloads and slices in space, a series of NSIDC daily files, conditioned on user inputs
    """
    # convert bounds to projected coordinates
    step = datetime.timedelta(days=1)  # M

    proj4str = '+proj=cea +lat_0=0 lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m'
    proj_out = pyproj.Proj('epsg:4326')
    # we define a projection object for the projection used in the downloaded grids.
    proj_in = pyproj.Proj(proj4str)

    # here we convert the coordinates in lat-lon into the coordinate system of the downloaded grids.
    bounds_xy = proj_coords(bounds, proj_out, proj_in)
    # points_xy = proj_coords(points_interest, proj_out, proj_in)
    date = start_date
    list_ds = []
    year = date.year  # let's store data per year
    while date <= end_date:
        url, success = download_measures(freq, res, HV, AD, date, username, password, sat)
        fn = url.strip('https://')  # strip https:// from the url to get the local location of the downloaded file
        path = fn.split('/')[0]  # split fn in little pieces on / and keep only the 0-th indexed piece (main folder)
        if success == 0:
            print('Retrieved {:s}'.format(url))
            # the file was successfully downloaded (if not zero, there was a problem or file is simply not available)
            # read file, cut a piece and add it to our list of time steps
            ds = xr.open_dataset(fn, decode_cf=False)
            ds_sel = select_bounds(ds, bounds_xy)
            list_ds.append(ds_sel.load())  # load the actual data so that we can delete the original downloaded files
            ds.close()
            shutil.rmtree(path)  # we're done cutting part of the required grid, so throw away the originally downloaded world grid.
        date += step   # increase by one day to go for the next download day.
        if (year != date.year) or (date > end_date):  # store results if one moves to a new year or the end date is reached
            # concatenate the list of timesteps into a new ds
            if len(list_ds) > 0:
                # only store if any values were found
                ds_year = correct_miss_fill(xr.concat(list_ds, dim='time'))
                # ds_year.data_vars
                encoding = {var: {'zlib': True} for var in ds_year.data_vars if var != 'crs'}
                # store the current dataset into a nice netcdf file
                # fn_out = os.path.abspath(os.path.join(out_path, fn_out_template.format(str(res), freq, HV, AD, year)))
                fn_out = fn_out_prefix + '_' + str(year) + '.nc'
                print('Writing output for year {:d} to {:s}'.format(year, fn_out))
                ds_year.to_netcdf(fn_out, encoding=encoding)
            # prepare a new dataset
            list_ds = []  # empty list
            year = date.year  # update the year

def plot_points(ax, points, **kwargs):
    x_point, y_point = zip(*points)
    ax.plot(x_point, y_point, **kwargs)
    return ax

def correct_miss_fill(ds):
    """
    Returns a properly decoded Climate-and-Forecast conventional ds, after correction of a conflicting attribute
    """
    for d in ds.data_vars:
        try:
            ds[d].attrs.update({'missing_value': ds[d]._FillValue})
        except:
            pass
    return xr.decode_cf(ds)

def c_m_ratio(ds_tb, x, y, x_off, y_off, cmode='Max'):
    if cmode == 'Corr':
        xmin = x-x_off
        xmax = x+x_off
        ymin = y-y_off
        ymax = y+y_off
        
        def cc(ts1, M):
            coef = np.ma.corrcoef(np.ma.masked_invalid(ts1.values.flatten()), np.ma.masked_invalid(M.values.flatten()))[1][0]
            return xr.DataArray(coef)
        
        ds_tb_sel = ds_tb.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        # convert the xarray data-array into a bunch of point time series
        # select series in (M)easurement location
        M = ds_tb_sel.sel(x=x, y=y, method='nearest')
        tb_points = ds_tb_sel.stack(points=('y', 'x')) # .reset_index(['x', 'y'], drop=True) # .transpose('points', 'time')
        # add a coordinate axis to the points
        # apply the function over all points to calculate the trend at each point
        # import pdb;pdb.set_trace()
        coefs = tb_points.groupby('points').apply(cc, M=M)
        # unstack back to lat lon coordinates
        coefs_2d = coefs.unstack('points').rename(dict(points_level_0='y', points_level_1='x')) # get the 2d back and rename axes back to x, y
        #LOWEST CALIBRATION
        # find the x/y index where the correlation is lowest
        idx_y, idx_x = np.where(coefs_2d==coefs_2d.min())
        # select  series in (C)alibration location (with lowest correlation)
        # import pdb;pdb.set_trace()
        #C = ds_tb_sel[:, idx_y, idx_x].squeeze(['x', 'y']).drop(['x', 'y'])  # get rid of the x and y coordinates of calibration pixel
        C = ds_tb_sel[:, idx_y, idx_x].squeeze(['x', 'y'])
        # which has the lowest correlation with the point of interest?
        ratio = C / M
        rat = ratio.values
        time = ratio.time.values
        cal = C.values
        mes = M.values
        # #MEAN CALIBRATION
        # # find the x/y index where the correlation is mean
        # idx_y_mean, idx_x_mean = np.where(coefs_2d==coefs_2d.mean())
        # # select  series in (C)alibration location (with lowest correlation)
        # C_mean = ds_tb_sel[:, idx_y_mean, idx_x_mean].squeeze(['x','y']).drop(['x','y'])  # get rid of the x and y coordinates of calibration pixel
        # ratio_mean = C_mean / M
        return C, M, ratio, rat, cal, mes, time #, C_mean, ratio_mean
    elif cmode == 'Max':
        xmin = x-x_off
        xmax = x+x_off
        ymin = y-y_off
        ymax = y+y_off
        #Generate window
        ds_tb_sel = ds_tb.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        xlist = list(ds_tb_sel.x.values)
        ylist = list(ds_tb_sel.y.values)
        avglist = []
        for i in range(len(xlist)):
            for j in range(len(ylist)):
                tempx =  xlist[i]
                tempy = ylist[j]
                c = ds_tb_sel.sel(x=tempx, y=tempy, method = 'nearest')
                avg = xr.DataArray.mean(c).values
                avglist.append(float(avg))     
        #Find 95 percentile
        prc = float(np.percentile(np.asarray(avglist), 95))
        #Find closest value
        diffunc = lambda list_value : abs(list_value - prc)
        prc = min(avglist, key=diffunc)
        for i in range(len(xlist)):
            for j in range(len(ylist)):
                tempx =  xlist[i]
                tempy = ylist[j]
                c = ds_tb_sel.sel(x=tempx, y=tempy, method = 'nearest')
                avg = float(xr.DataArray.mean(c).values)
                if avg == prc:
                    finalx = tempx
                    finaly = tempy            
        C = ds_tb_sel.sel(x=finalx, y=finaly, method = 'nearest')
        M = ds_tb_sel.sel(x=x, y=y, method = 'nearest')
        ratio = C / M
        rat = ratio.values
        cal = C.values
        mes = M.values
        time = ratio.time.values
        return C, M, ratio, rat, cal, mes, time #, C_mean, ratio_mean
    
    elif cmode == 'Manual':
        # select series in (M)easurement location
        M = ds_tb.sel(x=x, y=y, method='nearest')
        #In case manual C-selection is needed, the same will be done for calibration points
        with open('/home/selkhinifri/Documents/Workspace_Mali/cpoi_bakel.txt') as f:
            cpoints_xy = [tuple(map(float, i.split(','))) for i in f]
        cpoints_x, cpoints_y = zip(*cpoints_xy)
        C = ds_tb.sel(x=cpoints_x, y=cpoints_y, method='nearest')
        ratio = C / M
        rat = ratio.values
        cal = C.values
        mes = M.values
        return C, M, ratio, rat, cal, mes #, C_mean, ratio_mean

def c_m_c_ratio(ds_tb, x, y, x_off, y_off, wcpoints_x, wcpoints_y, cmode='Max'):
    Ccell = ds_tb.sel(x=wcpoints_x, y=wcpoints_y, method='nearest')
    if cmode == 'Corr':
        xmin = x-x_off
        xmax = x+x_off
        ymin = y-y_off
        ymax = y+y_off
        
        def cc(ts1, M):
            coef = np.ma.corrcoef(np.ma.masked_invalid(ts1.values.flatten()), np.ma.masked_invalid(M.values.flatten()))[1][0]
            return xr.DataArray(coef)
        
        ds_tb_sel = ds_tb.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        # convert the xarray data-array into a bunch of point time series
        # select series in (M)easurement location
        M = ds_tb_sel.sel(x=x, y=y, method='nearest')
        tb_points = ds_tb_sel.stack(points=('y', 'x')) # .reset_index(['x', 'y'], drop=True) # .transpose('points', 'time')
        # add a coordinate axis to the points
        # apply the function over all points to calculate the trend at each point
        # import pdb;pdb.set_trace()
        coefs = tb_points.groupby('points').apply(cc, M=M)
        # unstack back to lat lon coordinates
        coefs_2d = coefs.unstack('points').rename(dict(points_level_0='y', points_level_1='x')) # get the 2d back and rename axes back to x, y
        #LOWEST CALIBRATION
        # find the x/y index where the correlation is lowest
        idx_y, idx_x = np.where(coefs_2d==coefs_2d.min())
        # select  series in (C)alibration location (with lowest correlation)
        # import pdb;pdb.set_trace()
        #C = ds_tb_sel[:, idx_y, idx_x].squeeze(['x', 'y']).drop(['x', 'y'])  # get rid of the x and y coordinates of calibration pixel
        C = ds_tb_sel[:, idx_y, idx_x].squeeze(['x', 'y'])
        # which has the lowest correlation with the point of interest?
        ratio = ((C-M) / (C-Ccell))
        rat = ratio.values
        time = ratio.time.values
        cal = C.values
        mes = M.values
        cwet = Ccell.values
        # #MEAN CALIBRATION
        # # find the x/y index where the correlation is mean
        # idx_y_mean, idx_x_mean = np.where(coefs_2d==coefs_2d.mean())
        # # select  series in (C)alibration location (with lowest correlation)
        # C_mean = ds_tb_sel[:, idx_y_mean, idx_x_mean].squeeze(['x','y']).drop(['x','y'])  # get rid of the x and y coordinates of calibration pixel
        # ratio_mean = C_mean / M
        return C, M, ratio, rat, cal, mes, time, Ccell, cwet #, C_mean, ratio_mean
    elif cmode == 'Max':
        xmin = x-x_off
        xmax = x+x_off
        ymin = y-y_off
        ymax = y+y_off
        #Generate window
        ds_tb_sel = ds_tb.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        
        xlist = list(ds_tb_sel.x.values)
        ylist = list(ds_tb_sel.y.values)
        avglist = []
        for i in range(len(xlist)):
            for j in range(len(ylist)):
                tempx =  xlist[i]
                tempy = ylist[j]
                c = ds_tb_sel.sel(x=tempx, y=tempy, method = 'nearest')
                avg = xr.DataArray.mean(c).values
                avglist.append(float(avg))     
        #Find 95 percentile
        prc = float(np.percentile(np.asarray(avglist), 95))
        #Find closest value
        diffunc = lambda list_value : abs(list_value - prc)
        prc = min(avglist, key=diffunc)
        
        for i in range(len(xlist)):
            for j in range(len(ylist)):
                tempx =  xlist[i]
                tempy = ylist[j]
                c = ds_tb_sel.sel(x=tempx, y=tempy, method = 'nearest')
                avg = float(xr.DataArray.mean(c).values)
                if avg == prc:
                    finalx = tempx
                    finaly = tempy            
        C = ds_tb_sel.sel(x=finalx, y=finaly, method = 'nearest')
        M = ds_tb_sel.sel(x=x, y=y, method = 'nearest')
        ratio = (C-M) / (C-Ccell)
        rat = ratio.values
        cal = C.values
        mes = M.values
        cwet = Ccell.values
        time = ratio.time.values
        return C, M, ratio, rat, cal, mes, time, Ccell, cwet # C_mean, ratio_mean
    
    elif cmode == 'Manual':
        # select series in (M)easurement location
        M = ds_tb.sel(x=x, y=y, method='nearest')
        #In case manual C-selection is needed, the same will be done for calibration points
        cpoints_x, cpoints_y = zip(*cpoints_xy)
        C = ds_tb.sel(x=cpoints_x, y=cpoints_y, method='nearest')
        ratio = (C-M) / (C-Ccell)
        rat = ratio.values
        cal = C.values
        mes = M.values
        cwet = Ccell.values
        return C, M, ratio, rat, cal, mes, Ccell, cwet #, C_mean, ratio_mean
    
def createmodel(xtrain, ytrain, xtest, ytest, degree):
    #Create model based on training data  
    model = np.polyfit(xtrain, ytrain, degree)
    poly1d = np.poly1d(model)
    
    #Plot regression line and training data
    x2 = np.linspace(np.min(xtrain), np.max(xtrain))
    y2 = poly1d(x2)
    plt.figure()
    ax2 = plt.subplot()
    ax2.plot(xtrain, ytrain, "o", x2, y2)
    ax2.set_title('Training data and regression line')
    
    #PREDICT VALUES
    predictions = poly1d(xtest)
    
    #Plot predictions against original test data
    plt.figure()
    ax3 = plt.subplot()
    ax3.plot(xtest, ytest, "o", xtest, predictions, "o")
    ax3.set_title('Predictions vs. test data')
    #r2
    rsq = r2_score(ytest, predictions)
    
    #spearman
    from scipy.stats import spearmanr
    coef, p = spearmanr(xtrain,ytrain)
    # interpret the significance
    alpha = 0.05
    if p > alpha:
     	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
     	print('Samples are correlated (reject H0) p=%.3f' % p)
    return predictions, rsq, coef, model, poly1d, x2, y2

def solve_for_y(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    '''----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int -- the length of the window. Must be an odd integer number.
    order : int --- the order of the polynomial used in the filtering.  Must be less then `window_size` - 1.
    deriv: int --- the order of the derivative to compute (default = 0 means only smoothing)
    
    Returns
    ys : ndarray, shape (N) --- the smoothed signal (or it's n-th derivative).
    
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    '''
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as e:
        print(str(e))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
 
def subset(dataset, years, time):
    timestrings = []
    for i in years:
        #Find all timestamps where the year appears
        for j in time:
            if i == j.year:
                timestrings.append(j)
    timestrings = list(flatten(timestrings))
    indexlist = []
    for i in timestrings:
        indexlist.append(time.index(i))
    ss = [dataset[i] for i in indexlist]
    return ss, timestrings
    
def treatmentfunc(origdataset, years):
    #Combines transposition and subsetting in one function
    dataset = np.asarray(origdataset['Discharge'])
    time = origdataset['doy'].tolist()
    ss, timestrings = subset(dataset, years, time)
    return ss, timestrings

def countnan(dataset):
    #Count nan values in dataset
    return (dataset['Discharge'].isna().sum())

def nanfill(y):
    output = []
    for i in range(len(y)):
        if i == 0 or i == len(y)-1 or i == len(y):
            output.append(y[i]) 
        else:
            ss = [y[i-1], y[i], y[i+1]]
            if np.isnan(ss[1]) and np.isnan(ss[0]) == False and np.isnan(ss[2]) == False:
                interp = (ss[0]+ss[2])/2
                output.append(interp)
            else:
                output.append(ss[1])
    return output

def avgfill(data, window):
    #Input: windowsize (int) and numpy array of data
    output = []
    for i in range(len(data)):
        #Account for first values
        if (i >= 0 and i <= window-2): #use data itself if i is positive and smaller than the window size -2 (account for first values)
            output.append(data[i])
        #Filter value is average of 3 preceding values + value in question
        else:
            temp = output[(i-window):i] #preceding values
            temp.append(data[i]) #new value
            output.append(np.nanmean(temp)) #average
    return output

def breakpts(dis):
    #this function identifies breakpoints in the data
    #inputs: discharge data (dis) with a datetime index
    change = []
    y = dis.resample('Y').mean()
    #y = chik.rolling(365, min_periods = 0).mean()
    ytime = np.asarray(y.index)
    y = np.asarray(y)
    #Make change graph
    for i in range(len(y)):
        if i == 0:
            change.append(0)
        else:
            change.append(float(y[i]-y[i-1]))
    change = pd.DataFrame(change)
    change.index=ytime
    BP = []
    ct = 0
    for index, row in change.iterrows():
        if ct == 0:
            temprow = row #save because first entry does not have a previous entry
        #If change changes from positive to negative or vice versa    
        if row[0] > 0 and temprow[0] <= 0:
            BP.append(index)
        elif row[0] <= 0 and temprow[0] > 0:
            BP.append(index)
        temprow = row #save this entry for the next one
        ct = ct+1
        
    #Find integer locations of breakpoints in original data
    BPint = []
    for i in BP:
        ind = np.where(dis.index==i)[0]
        if ind: #if it is not empty
            BPint.append(int(ind))
    return BPint

def calc_ratio(years, points_x, points_y,location):
    #FOR CALCULATION OF RATIO AT A LIST OF POINTS
    #Empty list to generate multi-year series
    df = pd.DataFrame()
    Mf = pd.DataFrame()
    Cf = pd.DataFrame()
    Magf = pd.DataFrame()
    temp = pd.DataFrame()
    Ccoord = []
    Mcoord = []
    
    if location == 'bakel':
        poi=[0]
    else:
        poi=[2]
        
    for n in [x for x in range(len(points_x))]:
        print('CM Point ' + str(n))
        ratseries = []
        cseries = []
        mseries = []
        timeseries = []
        mag = []
        for i in range(len(years)):
            year = years[i]
            endofline = '*'+str(year)+'.nc'
            try:
                fns = ['AMSRE_'+str(year)+'.nc']
            except:
                print('No data file found, please put check the following location: {:s}')
            
            fns.sort()
            ds = xr.open_mfdataset(fns, combine='nested', concat_dim='time')
            #Get CM
            # compute CM ratio
            C, M, ratio, rat, cal, mes, time = c_m_ratio(ds['TB'], points_x[n], points_y[n], 50000.0, 50000.0)
            Ccoord.append(((C.x).astype(float).values.tolist(), (C.y).astype(float).values.tolist()))
            Mcoord.append(((M.x).astype(float).values.tolist(), (M.y).astype(float).values.tolist()))         
            cseries.append(cal)
            mseries.append(mes)
            timeseries.append(time)
        mseries = np.asarray(list(flatten(mseries)))
        cseries = np.asarray(list(flatten(cseries)))
        timeseries = np.asarray(list(flatten(timeseries)))
        
        #Apply filter
        mseries = avgfill(mseries, 3)
        cseries = avgfill(cseries, 3)
        
        #Calculate ratio
        ratseries = np.divide(cseries, mseries)
        
        #Calculate m
        meansignal = np.nanmean(ratseries)
        sdsignal = np.nanstd(ratseries)
        for t in ratseries:
            mag.append((t - meansignal) / sdsignal)
        magseries =  np.asarray(list(flatten(mag)))   
        
        #Turn into dataframe
        df[n] = ratseries
        Mf[n] = mseries
        Cf[n] = cseries
        Magf[n] = magseries
        df.index = timeseries
        Cf.index = timeseries
        Mf.index = timeseries
        Magf.index = timeseries    
    return df, Magf, Cf, Mf, Ccoord, Mcoord, timeseries, ds

def calc_ratio1pt(years, points_x, points_y,location):
    #Empty list to generate multi-year series
    df = pd.DataFrame()
    Mf = pd.DataFrame()
    Cf = pd.DataFrame()
    Magf = pd.DataFrame()
    temp = pd.DataFrame()
    Ccoord = []
    Mcoord = []
    
    if location == 'bakel':
        poi=[0]
    else:
        poi=[2]
        
    for n in poi:
        print('CM Point ' + str(n))
        ratseries = []
        cseries = []
        mseries = []
        timeseries = []
        mag = []
        for i in range(len(years)):
            year = years[i]
            endofline = '*'+str(year)+'.nc'
            try:
                fns = ['AMSRE_'+str(year)+'.nc']
            except:
                print('No data file found, please put check the following location: {:s}')
            
            fns.sort()
            ds = xr.open_mfdataset(fns, combine='nested', concat_dim='time')
            #Get CM
            # compute CM ratio
            C, M, ratio, rat, cal, mes, time = c_m_ratio(ds['TB'], points_x[n], points_y[n], 50000.0, 50000.0)
            Ccoord.append(((C.x).astype(float).values.tolist(), (C.y).astype(float).values.tolist()))
            Mcoord.append(((M.x).astype(float).values.tolist(), (M.y).astype(float).values.tolist()))         
            cseries.append(cal)
            mseries.append(mes)
            timeseries.append(time)
        mseries = np.asarray(list(flatten(mseries)))
        cseries = np.asarray(list(flatten(cseries)))
        timeseries = np.asarray(list(flatten(timeseries)))
        
        #Apply filter
        mseries = avgfill(mseries, 3)
        cseries = avgfill(cseries, 3)
        
        #Calculate ratio
        ratseries = np.divide(cseries, mseries)
        
        #Calculate m
        meansignal = np.nanmean(ratseries)
        sdsignal = np.nanstd(ratseries)
        for t in ratseries:
            mag.append((t - meansignal) / sdsignal)
        magseries =  np.asarray(list(flatten(mag)))   
        
        #Turn into dataframe
        df[n] = ratseries
        Mf[n] = mseries
        Cf[n] = cseries
        Magf[n] = magseries
        df.index = timeseries
        Cf.index = timeseries
        Mf.index = timeseries
        Magf.index = timeseries    
    return df, Magf, Cf, Mf, Ccoord, Mcoord, timeseries, ds

def calc_ratio1ptnew(years, points_x, points_y,location):
    #this function serves as a test to see if regression values go up if unfilled calues are omitted
    #Empty list to generate multi-year series
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    Mf = pd.DataFrame()
    Cf = pd.DataFrame()
    Magf = pd.DataFrame()
    Magf2 = pd.DataFrame()
    temp = pd.DataFrame()
    Ccoord = []
    Mcoord = []
    
    if location == 'bakel':
        poi=[0]
    else:
        poi=[2]
        
    for n in poi:
        print('CM Point ' + str(n))
        ratseries = []
        cseries = []
        mseries = []
        timeseries = []
        mag = []
        
        ratseries2 = []
        cseries2 = []
        mseries2 = []
        timeseries2 = []
        mag2 = []
        
        for i in range(len(years)):
            year = years[i]
            endofline = '*'+str(year)+'.nc'
            try:
                fns = ['AMSRE_'+str(year)+'.nc']
            except:
                print('No data file found, please put check the following location: {:s}')
            
            fns.sort()
            ds = xr.open_mfdataset(fns, combine='nested', concat_dim='time')
            #Get CM
            # compute CM ratio
            C, M, ratio, rat, cal, mes, time = c_m_ratio(ds['TB'], points_x[n], points_y[n], 50000.0, 50000.0)
            Ccoord.append(((C.x).astype(float).values.tolist(), (C.y).astype(float).values.tolist()))
            Mcoord.append(((M.x).astype(float).values.tolist(), (M.y).astype(float).values.tolist()))         
            cseries.append(cal)
            mseries.append(mes)
            timeseries.append(time)
        mseries = np.asarray(list(flatten(mseries)))
        cseries = np.asarray(list(flatten(cseries)))
        timeseries = np.asarray(list(flatten(timeseries)))
        
        #Apply filter
        mseries2 = mseries #unfiltered version
        cseries2 = cseries #unfiltered version
        mseries = avgfill(mseries, 3)
        cseries = avgfill(cseries, 3)
        
        #Calculate ratio
        ratseries = np.divide(cseries, mseries)
        ratseries2 = np.divide(cseries2, mseries2)
        
        #Calculate m
        meansignal = np.nanmean(ratseries)
        sdsignal = np.nanstd(ratseries)
        for t in ratseries:
            mag.append((t - meansignal) / sdsignal)
        magseries =  np.asarray(list(flatten(mag)))   
        
        #calculate m unfilled
        meansignal2 = np.nanmean(ratseries2)
        sdsignal2 = np.nanstd(ratseries2)
        for t in ratseries2:
            mag2.append((t - meansignal2) / sdsignal2)
        magseries2 =  np.asarray(list(flatten(mag2)))   
        
        #Turn into dataframe
        df[n] = ratseries
        Mf[n] = mseries
        Cf[n] = cseries
        Magf[n] = magseries
        
        df2[n] = ratseries2
        Magf2[n] = magseries2
        
        df.index = timeseries
        Cf.index = timeseries
        Mf.index = timeseries
        Magf.index = timeseries  
        
        df2.index = timeseries
        Magf2.index = timeseries    
        
    return df, Magf, Cf, Mf, df2, Magf2, Ccoord, Mcoord, timeseries, ds

def calc_cmcratio(years, points_x, points_y, wcpoints_x, wcpoints_y, timeseries, location):
    #FOR CALCULATION OF RATIO AT A LIST OF POINTS
    cmcdf = pd.DataFrame()
    Cwf = pd.DataFrame()
    cmcCcoord = []
    cmcMcoord = []

    if location == 'bakel':
        poi=[0]
        wcpoint_x = wcpoints_x[0]
        wcpoint_y = wcpoints_y[0]
    else:
        poi=[2]
        wcpoint_x = wcpoints_x[1]
        wcpoint_y = wcpoints_y[1]
    for n in [x for x in range(len(points_x))]:   #UNCHECK if you want to look at a list of locations
        print('CMC Point ' + str(n))
        cmcratseries = []
        cmccseries = []
        cmcmseries = []
        cmcwetseries = []
        for i in range(len(years)):
            year = years[i]
            endofline = '*'+str(year)+'.nc'
            try:
                fns = ['AMSRE_'+str(year)+'.nc']
            except:
                print('No data file found, please put check the following location: {:s}')
            
            fns.sort()
            ds = xr.open_mfdataset(fns, combine='nested', concat_dim='time')
            # compute CMC ratio
            C, M, ratio, rat, cal, mes, time, Ccell, cwet = c_m_c_ratio(ds['TB'], points_x[n], points_y[n], 50000.0, 50000.0, wcpoint_x, wcpoint_y)
            cmcCcoord.append(((C.x).astype(float).values.tolist(), (C.y).astype(float).values.tolist()))
            cmcMcoord.append(((M.x).astype(float).values.tolist(), (M.y).astype(float).values.tolist()))
            
            cmccseries.append(cal)
            cmcmseries.append(mes)
            cmcwetseries.append(cwet)
     
        cmcmseries = np.asarray(list(flatten(cmcmseries)))
        cmccseries = np.asarray(list(flatten(cmccseries)))
        cmcwetseries = np.asarray(list(flatten(cmcwetseries)))
        
        #Apply filter
        cmcmseries = avgfill(cmcmseries, 3)
        cmccseries = avgfill(cmccseries, 3)
        cmcwetseries = avgfill(cmcwetseries,3)
        a = np.subtract(cmccseries, cmcmseries)
        b = np.subtract(cmccseries, cmcwetseries)    
        cmcratseries = np.divide(a,b)
        cmcdf[n] = cmcratseries
        Cwf[n] = cmcwetseries
        Cwf.index = timeseries
    cmcdf.index = timeseries
    return cmcdf, cmcCcoord, cmcMcoord, ds, cmcwetseries, Cwf

def calc_cmcratio1pt(years, points_x, points_y, wcpoints_x, wcpoints_y, timeseries, location):
    #FOR CALCULATION OF RATIO AT ONE POINT
    cmcdf = pd.DataFrame()
    Cwf = pd.DataFrame()
    cmcCcoord = []
    cmcMcoord = []

    if location == 'bakel':
        poi=[0]
        wcpoint_x = wcpoints_x[0]
        wcpoint_y = wcpoints_y[0]
    else:
        poi=[2]
        wcpoint_x = wcpoints_x[1]
        wcpoint_y = wcpoints_y[1]
    for n in poi:
        print('CMC Point ' + str(n))
        cmcratseries = []
        cmccseries = []
        cmcmseries = []
        cmcwetseries = []
        for i in range(len(years)):
            year = years[i]
            endofline = '*'+str(year)+'.nc'
            try:
                fns = ['AMSRE_'+str(year)+'.nc']
            except:
                print('No data file found, please put check the following location: {:s}')
            
            fns.sort()
            ds = xr.open_mfdataset(fns, combine='nested', concat_dim='time')
            # compute CMC ratio
            C, M, ratio, rat, cal, mes, time, Ccell, cwet = c_m_c_ratio(ds['TB'], points_x[n], points_y[n], 50000.0, 50000.0, wcpoint_x, wcpoint_y)
            cmcCcoord.append(((C.x).astype(float).values.tolist(), (C.y).astype(float).values.tolist()))
            cmcMcoord.append(((M.x).astype(float).values.tolist(), (M.y).astype(float).values.tolist()))
            
            cmccseries.append(cal)
            cmcmseries.append(mes)
            cmcwetseries.append(cwet)
     
        cmcmseries = np.asarray(list(flatten(cmcmseries)))
        cmccseries = np.asarray(list(flatten(cmccseries)))
        cmcwetseries = np.asarray(list(flatten(cmcwetseries)))
        
        #Apply filter
        cmcmseries = avgfill(cmcmseries, 3)
        cmccseries = avgfill(cmccseries, 3)
        cmcwetseries = avgfill(cmcwetseries,3)
        a = np.subtract(cmccseries, cmcmseries)
        b = np.subtract(cmccseries, cmcwetseries)    
        cmcratseries = np.divide(a,b)
        cmcdf[n] = cmcratseries
        Cwf[n] = cmcwetseries
        Cwf.index = timeseries
    cmcdf.index = timeseries
    return cmcdf, cmcCcoord, cmcMcoord, ds, cmcwetseries, Cwf

def calc_cmcratio1ptnew(years, points_x, points_y, wcpoints_x, wcpoints_y, timeseries, location):
    #FOR CALCULATION OF RATIO AT ONE POINT
    cmcdf = pd.DataFrame()
    cmcdf2 = pd.DataFrame()
    Cwf = pd.DataFrame()
    cmcCcoord = []
    cmcMcoord = []

    if location == 'bakel':
        poi=[0]
        wcpoint_x = wcpoints_x[0]
        wcpoint_y = wcpoints_y[0]
    else:
        poi=[2]
        wcpoint_x = wcpoints_x[1]
        wcpoint_y = wcpoints_y[1]
    for n in poi:
        print('CMC Point ' + str(n))
        cmcratseries = []
        cmccseries = []
        cmcmseries = []
        cmcwetseries = []
        for i in range(len(years)):
            year = years[i]
            endofline = '*'+str(year)+'.nc'
            try:
                fns = ['AMSRE_'+str(year)+'.nc']
            except:
                print('No data file found, please put check the following location: {:s}')
            
            fns.sort()
            ds = xr.open_mfdataset(fns, combine='nested', concat_dim='time')
            # compute CMC ratio
            C, M, ratio, rat, cal, mes, time, Ccell, cwet = c_m_c_ratio(ds['TB'], points_x[n], points_y[n], 50000.0, 50000.0, wcpoint_x, wcpoint_y)
            cmcCcoord.append(((C.x).astype(float).values.tolist(), (C.y).astype(float).values.tolist()))
            cmcMcoord.append(((M.x).astype(float).values.tolist(), (M.y).astype(float).values.tolist()))
            
            cmccseries.append(cal)
            cmcmseries.append(mes)
            cmcwetseries.append(cwet)
     
        cmcmseries = np.asarray(list(flatten(cmcmseries)))
        cmccseries = np.asarray(list(flatten(cmccseries)))
        cmcwetseries = np.asarray(list(flatten(cmcwetseries)))
        
        #Apply filter
        cmcmseries2 = cmcmseries
        cmccseries2 = cmccseries
        cmcwetseries2 = cmcwetseries
        
        cmcmseries = avgfill(cmcmseries, 3)
        cmccseries = avgfill(cmccseries, 3)
        cmcwetseries = avgfill(cmcwetseries,3)
        a = np.subtract(cmccseries, cmcmseries)
        b = np.subtract(cmccseries, cmcwetseries)
        
        a2 = np.subtract(cmccseries2, cmcmseries2)
        b2 = np.subtract(cmccseries2, cmcwetseries2)
        
        cmcratseries = np.divide(a,b)
        cmcratseries2 = np.divide(a2,b2)
        
        cmcdf2[n] = cmcratseries2
        cmcdf[n] = cmcratseries
        Cwf[n] = cmcwetseries
        Cwf.index = timeseries
    cmcdf.index = timeseries
    cmcdf2.index = timeseries
    return cmcdf, cmcCcoord, cmcMcoord, ds, cmcwetseries, Cwf, cmcdf2

def deseason(data, degree):
    series = data
    
    doylist = []
    for index, row in series.iterrows():
        time = index
        doylist.append((time - datetime.datetime(time.year, 1, 1)).days + 1)
    doylist = pd.DataFrame(doylist)
    doylist.index = series.index
    
    #concat and drop nans
    dataset = pd.concat([series, doylist], axis=1)
    dataset.dropna(inplace=True)
    dataset.columns = ['Q', 'doy']
    # fit polynomial: x^2*b1 + x*b2 + ... + bn
    series= np.asarray(dataset.doy)
    X = series
    y = np.asarray(dataset.Q)
    coef = polyfit(X, y, degree)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)
    # create seasonally adjusted    
    values = y
    diff = list()
    for i in range(len(values)):
        value = values[i] - curve[i]
        diff.append(value)
    return pd.DataFrame(diff, index=dataset.index), curve

def makestationary(dis, cmcdf, Magf, degree):
    collist = []
    mag_detrended = pd.DataFrame(index=timeseries)
    cmc_detrended = pd.DataFrame(index=timeseries)
    dis_detrended = pd.DataFrame(index=timeseries)
    magtrd = pd.DataFrame(index=timeseries)
    cmctrd = pd.DataFrame(index=timeseries)
    distrd = pd.DataFrame(index=timeseries)
    discurve = pd.DataFrame(index=timeseries)
    cmccurve = pd.DataFrame(index=timeseries)
    magcurve = pd.DataFrame(index=timeseries)
        
    #STEP 1########### detrend interannual trend
    #drop na values but keep values paired
    #loop for each VGS
    for i in range(cmcdf.shape[1]):
        collist.append(str(i))
        dataset = pd.concat([dis, cmcdf[i], Magf[i]], axis=1)
        dataset = dataset.dropna()
        dataset.columns = ['Q', 'cmc', 'mag']
        
        #detrend signal and remerge
        distrdnew = pd.DataFrame(signal.detrend(dataset['Q'], bp=breakpts(pd.DataFrame(dataset.Q))), index=dataset.index)
        magtrdnew = pd.DataFrame(signal.detrend(dataset['mag'], bp = breakpts(pd.DataFrame(dataset.mag))), index=dataset.index)
        cmctrdnew = pd.DataFrame(signal.detrend(dataset['cmc'], bp = breakpts(pd.DataFrame(dataset.cmc))), index=dataset.index)  
        distrend = dataset['Q']-distrdnew[0]
        magtrend = dataset['mag']-magtrdnew[0]
        cmctrend = dataset['cmc']-cmctrdnew[0]
        distrdnew = pd.DataFrame(signal.detrend(dataset['Q']), index=dataset.index)
        magtrdnew = pd.DataFrame(signal.detrend(dataset['mag']), index=dataset.index)
        cmctrdnew = pd.DataFrame(signal.detrend(dataset['cmc']), index=dataset.index)  
        distrend = dataset['Q']-distrdnew[0]
        magtrend = dataset['mag']-magtrdnew[0]
        cmctrend = dataset['cmc']-cmctrdnew[0]
        
        #STEP 2########### Now we want to remove seasonality
        #based on https://machinelearningmastery.com/time-series-seasonality-with-python/
        dis_detrendednew, discurvenew = deseason(pd.DataFrame(distrdnew[0]), degree)
        mag_detrendednew, magcurvenew = deseason(pd.DataFrame(magtrdnew[0]), degree)                   
        cmc_detrendednew, cmccurvenew = deseason(pd.DataFrame(cmctrdnew[0]), degree)
        
        #add different points to one df
        dis_detrended = pd.concat([dis_detrended, dis_detrendednew], axis=1)
        mag_detrended = pd.concat([mag_detrended, mag_detrendednew], axis=1)
        cmc_detrended = pd.concat([cmc_detrended, cmc_detrendednew], axis=1)
        #add different points to one df
        distrd = pd.concat([distrd, distrdnew], axis=1)
        magtrd = pd.concat([magtrd, magtrdnew], axis=1)
        cmctrd = pd.concat([cmctrd, cmctrdnew], axis=1)
        #add different points to one df
        discurve = pd.concat([discurve, discurvenew+distrend], axis=1)
        magcurve = pd.concat([magcurve, magcurvenew+magtrend], axis=1)
        cmccurve = pd.concat([cmccurve, cmccurvenew+cmctrend], axis=1)
    
        #add columns that represent pois
        dis_detrended.columns = collist
        mag_detrended.columns = collist
        cmc_detrended.columns = collist  
        discurve.columns = collist
        magcurve.columns = collist
        cmccurve.columns = collist
    return dis_detrended, cmc_detrended, mag_detrended, distrd, cmctrd, magtrd, discurve, cmccurve, magcurve

def makestationary2(cmcdf, degree):
    collist = []
    cmc_detrended = pd.DataFrame(index=cmcdf.index)
    cmctrd = pd.DataFrame(index=cmcdf.index)
    cmccurve = pd.DataFrame(index=cmcdf.index)
        
    #STEP 1########### detrend interannual trend
    #drop na values but keep values paired
    #loop for each VGS
    for i in range(cmcdf.shape[1]):
        collist.append(str(i))
        dataset = pd.DataFrame(cmcdf[i])
        dataset = dataset.dropna()
        dataset.columns = ['cmc']
        
        #detrend signal and remerge
        cmctrdnew = pd.DataFrame(signal.detrend(dataset['cmc'], bp = breakpts(pd.DataFrame(dataset.cmc))), index=dataset.index)  
        cmctrend = dataset['cmc']-cmctrdnew[0]
        
        #STEP 2########### Now we want to remove seasonality
        #based on https://machinelearningmastery.com/time-series-seasonality-with-python/                 
        cmc_detrendednew, cmccurvenew = deseason(pd.DataFrame(cmctrdnew[0]), degree)
        
        #add different points to one df
        cmc_detrended = pd.concat([cmc_detrended, cmc_detrendednew], axis=1)
        #add different points to one df
        cmctrd = pd.concat([cmctrd, cmctrdnew], axis=1)
        #add different points to one df
        cmccurve = pd.concat([cmccurve, cmccurvenew+cmctrend], axis=1)
    
        #add columns that represent pois
        cmc_detrended.columns = collist  
        cmccurve.columns = collist
    return cmc_detrended, cmctrd, cmccurve

def calculate_residuals(model, features, label, predictions):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results

def normal_errors_assumption(model, features, label, predictions, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    # Calculating residuals
    df_results = calculate_residuals(model, features, label, predictions)

    #Shapiro test
    st, p = scipy.stats.shapiro(df_results['Residuals'])
    p = normal_ad(df_results['Residuals'])[1]
    print('Statistics=%.3f, p=%.3f' % (st,p))
    alpha = 0.05
    if p > alpha:
        print('Data looks Gaussian (do not reject H0)')
    else:
        print('Data does not look gaussian (reject H0)')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    scipy.stats.probplot(df_results['Residuals'], dist="norm", plot=plt)

    return(p, df_results)

def homoscedasticity_assumption(model, features, label, predictions):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
      
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label, predictions)
    df_results=df_results.reset_index(drop=True)
    
    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  
    
    st, p = scipy.stats.levene(xt,yt)
    print('Statistics=%.3f, p=%.3f' % (st,p))
    alpha = 0.05
    if p > alpha:
        print('Data looks homoscedastic (do not reject H0)')
    else:
        print('Data does not look homoscedastic (reject H0)')  









# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:32:44 2020

@author: lcmok
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:29:05 2020

@author: lcmok
"""
#THIS MAIN SCRIPT WILL LET YOU EXTRACT THE TB VALUES FROM THE .NC FILES ON YOUR PC. PLEASE ENSURE THE .NC FILES ARE IN YOUR WORKING DIRECTORY AND ARE NAMED AS FOLLOWS:
#A_YYYY.nc -> if you use several platforms within one year, name the next platform B, and so forth.
#The .nc files can be downloaded using the software Wget (Linux/Apple/Windows) and the DownloadSatdata_LM script. Read the documentation there.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy
from math import factorial
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import statsmodels.formula.api as smf
from numpy import polyfit
from scipy import signal
import datetime
import pyproj
import xarray as xr
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from pandas.core.common import flatten
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa as smtsa
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
#import own library
#import nsidc2

#%%
#-------------------------#---------------------------#---------------------------#---------------------------#
#VARIABLES (CHANGE WHERE APPROPIATE)
#---------------------------#---------------------------#---------------------------#---------------------------#

#TO CHANGE:
location = 'bakel' #bakel or kayes
#Add years of interest in chronological order
years = list(range(2002, 2011))


#--------

#The downloaded data is in the projection cylindrical equal area
proj4str = '+proj=cea +lat_0=0 +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m'

#Projection object for the projection used in the downloaded grids.
proj_in = pyproj.Proj(proj4str)

#Target: WGS84 (EPSG code 4326)
proj_out = pyproj.Proj(init='epsg:4326')       #unit is degrees

#Rough bounds of Mali (y, x / long, lat) lower left to upper right
bounds_xy = [(11.91, -12.86),
          (15.4, -9.38,),
         ]
#EXAMPLE OF POIS: 33.84683, -9.954743         
with open('/home/selkhinifri/Documents/Workspace_Mali/poi_'+location+'.txt') as f: #change string to your own location, this refers to a txt file with the coordinate sets. An example for cell C0 is 34.88398, -15.99103
    points = [tuple(map(float, i.split(','))) for i in f]
    points_xy = proj_coords(points, proj_out, proj_in)
    #Unpack points in coordinate system of netcdf
    points_x, points_y = zip(*points_xy)
    
with open('/home/selkhinifri/Documents/Workspace_Mali/wcpoi_manantali.txt') as f: #idem dito
    wcpoints = [tuple(map(float, i.split(','))) for i in f]
    wcpoints_xy = proj_coords(wcpoints, proj_out, proj_in)
    #Unpack points in coordinate system of netcdf
    wcpoints_x, wcpoints_y = zip(*wcpoints_xy)   

#-------------------------#---------------------------#---------------------------#---------------------------#
# CALCULATION OF RATIOS FOR 1 LOCATION ONLY (for several locations, please use the script in the Fig8_TLCC script)
#---------------------------#---------------------------#---------------------------#---------------------------#

#note that for karonga, the downstream poi was the second one in the list due to some problems with the signal. this is why it starts with number 2

# CM
df, Magf, Cf, Mf, Ccoord, Mcoord, timeseries, ds = calc_ratio1pt(years, points_x, points_y,location)

# CMC
cmcdf, cmcCcoord, cmcMcoord, ds2, cmcwet, Cwf = calc_cmcratio1pt(years, points_x, points_y, wcpoints_x, wcpoints_y, timeseries,location)

#%%
#-------------------------#---------------------------#---------------------------#---------------------------#
#DATA CORRECTION/TREATMENT AND SUBSETTING FOR DISCHARGE
#---------------------------#---------------------------#---------------------------#---------------------------#

#Load discharge data and recognize nan values
if location == 'bakel':
    chik = pd.read_csv('/home/selkhinifri/Documents/Workspace_Mali/Bakel.csv', delimiter = ',', header = 0, na_filter=True)
    chik.columns = ['doy', 'Discharge']
    chik['doy'] = pd.to_datetime(chik.doy, dayfirst=True)
    chikss, chiktime = treatmentfunc(chik, years)
    discharge = np.asarray(chikss)
    time = np.asarray(chiktime)
elif location == 'kayes':
    mwaki = pd.read_csv('/home/selkhinifri/Documents/Workspace_Mali/Kayes.csv', delimiter = ';', header = 0, na_values = -999, na_filter=True)
    mwaki['Date'] = pd.to_datetime(mwaki.Date, dayfirst=True)
    mwaki.columns = ['doy','Discharge']
    mwakiss, mwakitime = treatmentfunc(mwaki, years)
    discharge = np.asarray(mwakiss)
    time = np.asarray(mwakitime)
    
dis = pd.DataFrame(discharge)
if location == 'bakel':
    dis.index = chiktime
    poi = 0
elif location == 'kayes':
    dis.index = mwakitime
    poi = 2

#%% REMOVE AWAY DRY SEASON FROM ORIGINAL DATA
#cmcdf[(cmcdf.index.month>4) & (cmcdf.index.month<12)] = np.nan
#Magf[(Magf.index.month>4) & (Magf.index.month<12)] = np.nan
#dis[(dis.index.month>4) & (dis.index.month<12)] = np.nan

#%%-------------------------#---------------------------#---------------------------#---------------------------#
#DATA PLOTTING: CM-ratio and discharge
#---------------------------#---------------------------#---------------------------#---------------------------#

# Create the general figure
fig = plt.figure(figsize=(14,6))
# Plot mwaki data
ax1 = fig.add_subplot(111)
plt.title='Discharge and satellite signals in Bakel'
plt.xlabel('Day of Year')
ax1.plot(Magf[poi], linestyle='-', color='b', label = 'Flood Magnitude')     
ax1.set(ylabel="Flood magnitude")
# Add filtered data in the same figure
ax2 = fig.add_subplot(111, frameon=False, sharex=None)
ax2.plot(dis, linestyle='-', color='r', label = 'Discharge')
ax2.yaxis.set_label_position("right")
ax2.set(ylabel="Discharge (m3/s)")
ax2.yaxis.tick_right()

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines = lines_1 + lines_2
labels = labels_1 + labels_2

ax1.legend(lines, labels, loc=0)
plt.show()
#%%-------------------------#---------------------------#---------------------------#---------------------------#
#DATA PLOTTING: 2D-brightness temperatures in Bakel
#---------------------------#---------------------------#---------------------------#---------------------------#
plt.figure(figsize=(17,12)) 
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.8)
ax.add_feature(cartopy.feature.RIVERS, linestyle='-', alpha=.8)
xi, yi =np.meshgrid(ds.x, ds.y)
Mcoord_xy = proj_coords(Mcoord, proj_in, proj_out)
Ccoord_xy = proj_coords(Ccoord, proj_in, proj_out)

if location == 'bakel':
    wcpointsnew = ds.sel(x=wcpoints_x[0], y=wcpoints_y[0], method='nearest')
    wcx = float(wcpointsnew.x.values)
    wcy = float(wcpointsnew.y.values)
    wcpointsnew = [(wcx, wcy)]
    wcpointsnew = proj_coords(wcpointsnew, proj_in, proj_out)
    #to ensure karonga and bakel can be plotted in one plot
    wcpointsnewsave = wcpointsnew
    Ccoordsave = Ccoord_xy
    Mcoordsave = Mcoord_xy

else:
    wcpointsnew = ds.sel(x=wcpoints_x[1], y=wcpoints_y[1], method='nearest')
    wcx = float(wcpointsnew.x.values)
    wcy = float(wcpointsnew.y.values)
    wcpointsnew = [(wcx, wcy)]
    wcpointsnew = proj_coords(wcpointsnew, proj_in, proj_out)

if location == 'bakel':
    loni, lati = pyproj.transform(proj_in, proj_out, xi, yi)
    p = ax.pcolormesh(loni, lati, ds['TB'].values[5], transform=ccrs.PlateCarree(), cmap='terrain')
    # also plot some points of interest  -> note that this plots both locations in one plot, if the script does not not Ccoordsave and Mccoordsave for example, this is because you are working with one location
    #in that case, please delete the lines below. If you are working with both, please run 'bakel' first and then 'kayes'. In that way, the points will be plotted on the same map.
    ax = plot_points(ax, Ccoord_xy, marker='o', color='r', linewidth=0., transform=ccrs.PlateCarree())
    ax = plot_points(ax, Mcoord_xy, marker='o', color='g', linewidth=0., transform=ccrs.PlateCarree())
    ax = plot_points(ax, wcpointsnew, marker='o', color='k', linewidth=0., transform=ccrs.PlateCarree())
else:
    ax = plot_points(ax, Ccoordsave, marker='o', color='r', linewidth=0., transform=ccrs.PlateCarree())
    ax = plot_points(ax, Mcoordsave, marker='o', color='g', linewidth=0., transform=ccrs.PlateCarree())
    ax = plot_points(ax, wcpointsnewsave, marker='o', color='k', linewidth=0., transform=ccrs.PlateCarree())

plt.colorbar(p, label='Brightness temperature [K]')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle=':')
plt.title = ('Brightness Temperature')
gl.xlabels_top = False
gl.ylabels_right = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
plt.show()
# Gracefully close ds
ds.close()
#plt.savefig('/home/selkhinifri/Documents/Workspace_Mali/Figures/Tb_map.png', dpi = 400)

#%%-------------------------#---------------------------#---------------------------#---------------------------#
#COPYING DATA FOR IN EXCEL
#---------------------------#---------------------------#---------------------------#---------------------------#
# Selecting data for excel -> COPIES TO CLIPBOARD, Cd, M and Cw --> will be used in Fig2_raw_tb.py
f = pd.concat([pd.DataFrame(Cf[poi]), pd.DataFrame(Mf[poi]), pd.DataFrame(cmcwet, index=timeseries)], axis=1)
#f.to_clipboard()
f.to_csv('test1.csv')

#%% -> COPIES TO CLIPBOARD, Q, cmc and m and discharge --> will be used in Fig3_4_Tb_vs_discharge.py
f = pd.concat([pd.DataFrame(dis), pd.DataFrame(cmcdf[poi]), pd.DataFrame(Magf[poi]),], axis=1)
#f.to_clipboard()
f.to_csv('test2.csv')
