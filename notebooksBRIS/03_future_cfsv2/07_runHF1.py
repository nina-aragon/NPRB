#!/usr/bin/env python
# coding: utf-8

# In[2]:


# The cells in this notebook run each step in the HydroFlow workflow
# This workflow was developed to function from within the designated SnowModel
# folder for each domain.

# Import all of the python packages used in this workflow.
import scipy
import numpy as np
from collections import OrderedDict
import os, sys
from pylab import *
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from datetime import date, datetime
from datetime import timedelta  
import json
import itertools
import requests
import os

# Choose a domain
domain = 'BRIS'

# Path to the SnowModel folder
SMpath = '/nfs/attic/dfh/2020_NPRB/domain_'+domain+'/snowmodel2023_cfsv2/'
#SMpath = '/scratch/Nina/NPRB/domain_'+domain+'/snowmodel2023_cfsv2/'

#path to NPRB domains
domains_resp = requests.get("https://raw.githubusercontent.com/NPRB/02_preprocess_python/main/NPRB_domains.json")
domains = domains_resp.json()
    
# Define nx and ny for this domain to be used later
nx = domains[domain]['ncols']
ny = domains[domain]['nrows']
clsz = domains[domain]['cellsize']
xll = domains[domain]['xll']
yll = domains[domain]['yll']

start_years = [2018]#list(range(2008,2019))


# In[3]:


# Function to edit fortran files
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


# In[4]:


# Function to write out .ctl file for hydroflow steps
def disc_ctl(var,path,nx,ny,xll,yll,stdt,yrlabel,nt=1):
    '''
    vars can be:
    disc = hydroflow discharge
    tc = time coefficients
    watershed = watershed
    dir = flow direction 
    '''
    dt = datetime.strptime(st_dt, '%Y-%m-%d').strftime('%M:%SZ%d%b%Y')
    if var == 'disc':
        # Capture some variables from this SnowModel simulation
        lines = ['DSET ^disc_'+yrlabel+'.gdat',\
                 'UNDEF -9999.0',\
                 'OPTIONS BINPRECISION float32',\
                 'XDEF '+nx+' LINEAR '+xll+' '+clsz,\
                 'YDEF '+ny+' LINEAR '+yll+' '+clsz,\
                 'ZDEF 1 LEVELS 1',\
                 'TDEF '+str(nt)+' LINEAR '+dt+' 1dy',\
                 'VARS 2',\
                 'slow 1 0 SLOW',\
                 'fast 1 0 FAST',\
                 'ENDVARS']
    elif var == 'tc':
        lines = ['DSET ^tc_'+yrlabel+'.gdat',\
                 'UNDEF -9999.0',\
                 'OPTIONS BINPRECISION float32',\
                 'XDEF '+nx+' LINEAR '+xll+' '+clsz,\
                 'YDEF '+ny+' LINEAR '+yll+' '+clsz,\
                 'ZDEF 1 LEVELS 1',\
                 'TDEF '+str(nt)+' LINEAR '+dt+' 1dy',\
                 'VARS 2',\
                 'tcoef_slow 1 0 TCSLOW',\
                 'tcoef_fast 1 0 TCFAST',\
                 'ENDVARS']
    elif var == 'watershed':
        lines = ['DSET ^watershed.gdat',\
                 'UNDEF -9999.0',\
                 'OPTIONS BINPRECISION float32',\
                 'XDEF '+nx+' LINEAR '+xll+' '+clsz,\
                 'YDEF '+ny+' LINEAR '+yll+' '+clsz,\
                 'ZDEF 1 LEVELS 1',\
                 'TDEF '+str(nt)+' LINEAR '+dt+' 1dy',\
                 'VARS 3',\
                 'dir 1 0 flow direction',\
                 'wshed 1 0 watersheds',\
                 'order 1 0 flow accumulation',\
                 'ENDVARS']
    elif var == 'dir':
        lines = ['DSET ^dir.gdat',\
                 'UNDEF -9999.0',\
                 'OPTIONS BINPRECISION float32',\
                 'XDEF '+nx+' LINEAR '+xll+' '+clsz,\
                 'YDEF '+ny+' LINEAR '+yll+' '+clsz,\
                 'ZDEF 1 LEVELS 1',\
                 'TDEF '+str(nt)+' LINEAR '+dt+' 1dy',\
                 'VARS 1',\
                 'dir 1 0 flow direction',\
                 'ENDVARS']       

    with open(path, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    f.close() 


# In[7]:


for styr in start_years:
    print(styr)
    # file naming/organization 
    yrlabel = str(styr+81)+'-'+str(styr+82)
    #start date    
    st_dt = str(styr)+'-10-01'
    #end date
    ed_dt = str(styr+2)+'-09-30'
    
    # RUN TC

    # Path to single dem/veg file
    dem_veg = SMpath+'topo_vege/dem_veg_'+domain+'_fut.gdat'

    # Path to directions folder
    tc_path = SMpath+'timecoef_fut1/'

    # Path to the flow directions fortran file for editing
    tc_fortran = SMpath+'timecoef_fut1/timecoefs.f'

    # Create the years variable for use later
    print('Working on timecoefs')

    # Output path
    out_path = SMpath+'outputs_fut/'

    # The number of days in the simulation
    num_days = str((datetime.strptime(ed_dt, '%Y-%m-%d')-datetime.strptime(st_dt, '%Y-%m-%d')).days+1)

    # Run the replace line function in order to change the name of the input 
    # and output files in the script
    replace_line(tc_fortran, 9, '      parameter (nx='+nx+',ny='+ny+')\n')
    replace_line(tc_fortran, 11, '      parameter (max_iter='+num_days+')\n')
    replace_line(tc_fortran, 48, '     &  \'../topo_vege/dem_veg_'+domain+'.gdat\',\n')
    replace_line(tc_fortran, 56, '     &  file=\'../outputs_fut/swed_'+yrlabel+'.gdat\',\n')
    replace_line(tc_fortran, 205, '      open (unit=71,file=\'../outputs_fut/tc_'+yrlabel+'.gdat\',\n')

    # Use line magic to change directories to the file locations
    # Run the fortran script and save the output
    get_ipython().run_line_magic('cd', '$tc_path')
    get_ipython().system('gfortran -mcmodel=medium timecoefs.f')
    get_ipython().system('./a.out')

    # write a .ctl file for tcs
    ctlpath = out_path+'tc_'+yrlabel+'.ctl'
    disc_ctl('tc',ctlpath,nx,ny,xll,yll,st_dt,yrlabel,nt=num_days)
    #     # Print out a metadata textfile
    #     meta(domain,name,out_path)
    print('tc gdat created for '+domain+' domain')
    
    
    
    # RUN HF

    # Path to single dem/veg file
    dem_veg = SMpath+'topo_vege/dem_veg_'+domain+'.gdat'

    # Path to hydroflow folder
    hf_path = SMpath+'hydroflow_fut1/'

    # Path to the flow directions fortran file for editing
    hf_fortran = SMpath+'hydroflow_fut1/hydroflow.f'

    # Create the years variable for use later
    print('Working on HydroFlow')

    # Output path
    out_path = SMpath+'outputs_fut/'

    # The number of days in the simulation
    num_days = str((datetime.strptime(ed_dt, '%Y-%m-%d')-datetime.strptime(st_dt, '%Y-%m-%d')).days+1)


    # Run the replace line function in order to change the name of the input 
    # and output files in the script
    replace_line(hf_fortran, 9, '      parameter (nx='+nx+',ny='+ny+',nxny=nx*ny)\n')
    replace_line(hf_fortran, 10, '      parameter (max_iter='+num_days+')\n')
    replace_line(hf_fortran, 37, '      deltax = '+clsz+'\n')
    replace_line(hf_fortran, 38, '      deltay = '+clsz+'\n')
    replace_line(hf_fortran, 49, '      open (unit=41,file=\'../watershed/watershed_index.txt\')\n')
    replace_line(hf_fortran, 64, '     &  file=\'../outputs_fut/rofx_'+yrlabel+'.gdat\',\n')
    replace_line(hf_fortran, 86, '      open (81,file=\'../outputs_fut/tc_'+yrlabel+'.gdat\',\n')
    replace_line(hf_fortran, 280, '      open (unit=71,file=\'../outputs_fut/disc_'+yrlabel+'.gdat\',\n')

    # Use line magic to change directories to the file locations
    # Run the fortran script and save the output
    get_ipython().run_line_magic('cd', '$hf_path')
    get_ipython().system('gfortran -mcmodel=medium hydroflow.f')
    get_ipython().system('./a.out')

    # write a .ctl file for discharge
    ctlpath = out_path+'disc_'+yrlabel+'.ctl'
    disc_ctl('disc',ctlpath,nx,ny,xll,yll,st_dt,yrlabel,nt=num_days)    
    # Print out a metadata textfile
    #meta(domain,name,out_path)
    print('disc gdat created for '+domain+' domain')
    
    # remove tc gdat
    tcfile = out_path+'/tc_'+yrlabel+'.gdat'
    get_ipython().system('rm -f $tcfile')


# In[ ]:




