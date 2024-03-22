#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from datetime import datetime, date
import json
import geopandas as gpd
import requests
import matplotlib.pyplot as plt


# In[2]:


# Select modeling domain
domain = 'BEAU'
# data location
datapath = '/nfs/attic/dfh/2020_NPRB/data/'
# SM forcing files 
SMdatapath = datapath+'SMinputs/'+domain+'/'
# dem tif
demtif = SMdatapath+domain+'_dem.tif'

# historical ET data - na
path_hist = '/nfs/attic/dfh/data/climate_na_et/Normal_1991_2020_monthly/'
# future ET data - na
path_fut = '/nfs/attic/dfh/data/climate_na_et/ensemble_8GCMs_ssp585_2071_2100/'
# historical ET data - domain
path_hist_out = datapath+'climate_na_et/Normal_1991_2020_monthly/'+domain+'/'
# future ET data - domain
path_fut_out = datapath+'climate_na_et/ensemble_8GCMs_ssp585_2071_2100/'+domain+'/'


# In[5]:


#historic monthly data
histpath = path_hist_out+'monthly_climatena_dif.nc'
datah = xr.open_dataset(histpath)
datah


# In[4]:


# create date timeseries with the date of the middle day of each month 
dateslist = ['1-1-2006','2-15-2006','3-15-2006','4-15-2006','5-15-2006','6-15-2006',
'7-15-2006','8-15-2006','9-15-2006','10-15-2006','11-15-2006','12-31-2006']
dates = [datetime.strptime(val, '%m-%d-%Y') for val in dateslist]


var = 'tav_dif'
# interpolate monthly et for each pixel
datadaily = np.empty([365,len(datah.y),len(datah.x)])
for i in range(len(datah.y)):
    for j in range(len(datah.x)):
        datapx = datah[var].isel(x=j,y=i)
        if np.isnan(np.sum(datapx)) == True:
            datadaily[:,i,j] = [np.nan]*365
        else:
            df = pd.DataFrame({'time': dates, 'data': datapx})
            df = df.set_index('time').resample('D').interpolate(method ='polynomial',order=2)
            datadaily[:,i,j] = df.data.values
# build xarray data array to save out
datady = xr.DataArray(
    datadaily,
    coords={
        "time":df.index.values,
        "y": datah.y.values,
        "x": datah.x.values,
    },
    dims=["time","y", "x"],
)
# convert to dataset
ds=datady.to_dataset(name = var)


var = 'tmx_dif'
# interpolate monthly et for each pixel
datadaily = np.empty([365,len(datah.y),len(datah.x)])
for i in range(len(datah.y)):
    for j in range(len(datah.x)):
        datapx = datah[var].isel(x=j,y=i)
        if np.isnan(np.sum(datapx)) == True:
            datadaily[:,i,j] = [np.nan]*365
        else:
            df = pd.DataFrame({'time': dates, 'data': datapx})
            df = df.set_index('time').resample('D').interpolate(method ='polynomial',order=2)
            datadaily[:,i,j] = df.data.values
ds[var] = (['time','y', 'x'], datadaily )

var = 'tmn_dif'
# interpolate monthly et for each pixel
datadaily = np.empty([365,len(datah.y),len(datah.x)])
for i in range(len(datah.y)):
    for j in range(len(datah.x)):
        datapx = datah[var].isel(x=j,y=i)
        if np.isnan(np.sum(datapx)) == True:
            datadaily[:,i,j] = [np.nan]*365
        else:
            df = pd.DataFrame({'time': dates, 'data': datapx})
            df = df.set_index('time').resample('D').interpolate(method ='polynomial',order=2)
            datadaily[:,i,j] = df.data.values
ds[var] = (['time','y', 'x'], datadaily )

var = 'pr_dif'
# interpolate monthly et for each pixel
datadaily = np.empty([365,len(datah.y),len(datah.x)])
for i in range(len(datah.y)):
    for j in range(len(datah.x)):
        datapx = datah[var].isel(x=j,y=i)
        if np.isnan(np.sum(datapx)) == True:
            datadaily[:,i,j] = [np.nan]*365
        else:
            df = pd.DataFrame({'time': dates, 'data': datapx})
            df = df.set_index('time').resample('D').interpolate(method ='polynomial',order=2)
            datadaily[:,i,j] = df.data.values
ds[var] = (['time','y', 'x'], datadaily )


# In[ ]:


# save as netcdf
path = path_hist_out+'daily_climatena_dif.nc'
ds.to_netcdf(path)