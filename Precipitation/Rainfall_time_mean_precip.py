#!/bin/env python 

import scipy.stats
import numpy as np ; import numpy.ma as ma
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from netCDF4 import Dataset
import netCDF4
import time
import pandas as pd
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys, glob, os, re

#----------function allowing to save high-quality figures in different formats---------- 
def save(path, ext='png', close=True, verbose=True):
    
    """
    
    Save a figure from pyplot.
    Parameters
    ----------
    path : string
    The path (and filename, without the extension) to save the
    figure to.
    ext : string (default='png')
    The file extension. This must be supported by the active
    matplotlib backend (see matplotlib.backends module).  Most
    backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
    Whether to close the figure after saving.  If you want to save
    the figure multiple times (e.g., to multiple formats), you
    should NOT close it in between saves or you will have to
    re-plot it.
    verbose : boolean (default=True)
    whether to print information about when and where the image
    has been saved.
    
    """
    # Extract the directory and filename from the given path
    
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
    
    #If the directory does not exist, create it
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # The final path to save to
    
    savepath = os.path.join(directory, filename)
    if verbose:
        print("Saving figure to '%s'..." % savepath),
    
    # Actually save the figure
    
    plt.savefig(savepath)
    
    # Close it
    
    if close:
        plt.close()
    if verbose:
        print("Done")

#----------directory where the data are stored------------------------------------------- 

path = '/home/rebecca/Desktop/Practice/Project/Pre'

##########################################################################################

#--------Here let's read the different netDF files corresponding to four seasons--------- 

ncfile0 = Dataset(path+'/chirps-v2.0.1985-2004.monthly_p25.nc', 'r', format='NETCDF4')
pre = ncfile0.variables['precip'][:,:,:] # MULTIPLY BY 86400 TO CONVERT TO MM/DAY
lat = ncfile0.variables['latitude'][:]
lon = ncfile0.variables['longitude'][:]
time = ncfile0.variables['time']
ncfile0.close()

##############################compute mean ####################################

pre = np.mean(pre,axis=0)  
pre

#-------mask precipitation below 1 mm/day

pre = ma.masked_where(pre <= 0.3, pre)

#=========================Map resources==============================

fig = plt.figure(figsize=(7.40,6.10))
kwargs = {'format': '%.0f'}  # to fix decimals at X numbers after - put **kwargs in plt.cbar 
[lon2d, lat2d] = np.meshgrid(lon, lat)


prj = ccrs.PlateCarree(central_longitude=0.0)

axa = plt.subplot(111, projection=prj)
#axa.add_feature(cfeat.COASTLINE ,edgecolor = 'k')
axa.add_feature(cfeat.BORDERS.with_scale('10m'),linewidth=0.5)
axa.coastlines(resolution='10m',linewidth=0.5);
#axa.add_feature(cfeat.BORDERS, linestyle='-', alpha=.5)
#axa.add_feature(cfeat.OCEAN,edgecolor='k',facecolor='w') # to mask ocean
cs1 = plt.contourf(lon2d,lat2d,pre,levels = np.linspace(1., 9.,11),cmap=plt.cm.rainbow)
axa.set_extent([12 ,31, -14, 6])
axa.set_xticks(range(12,31,15), crs=prj)
axa.set_yticks(range(-14,6,15), crs=prj)
axa.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
axa.yaxis.set_major_formatter(LATITUDE_FORMATTER)
plt.title('RAINFALL CLIMATOLOGY OVER DRC', fontsize=8)
plt.ylabel('')
cb0 = plt.colorbar ( cs1, ax = axa,orientation ='vertical' )

cb0.set_label('[mm/day]',rotation=270) 

save(path+'/figures/Rainfall_time_mean', ext='pdf', close=True, verbose=True)  # save high quality figures