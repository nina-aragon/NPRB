#!/usr/bin/env python
# coding: utf-8

# ## Baseline HydroFlow

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
domain = 'BEAU'

# Path to the SnowModel folder
SMpath = '/nfs/attic/dfh/2020_NPRB/domain_'+domain+'/snowmodel2023_cfsv2/'

#path to NPRB domains
domains_resp = requests.get("https://raw.githubusercontent.com/NPRB/02_preprocess_python/main/NPRB_domains.json")
domains = domains_resp.json()
    
# Define nx and ny for this domain to be used later
nx = domains[domain]['ncols']
ny = domains[domain]['nrows']
clsz = domains[domain]['cellsize']
xll = domains[domain]['xll']
yll = domains[domain]['yll']


#start calibration date    
st_dt = '2011-10-01'#domains[domain]['st']

#end calibration date
ed_dt = '2018-09-30'#domains[domain]['ed']


# ## Define functions

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
def disc_ctl(var,path,nx,ny,xll,yll,stdt,nt=1):
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
        lines = ['DSET ^disc.gdat',\
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
        lines = ['DSET ^tc.gdat',\
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


# ## Create the time coefficients output file

# In[4]:


# INPUTS:
# OUTPUTS: timecoefs_[domain].gdat

# Path to single dem/veg file
dem_veg = SMpath+'topo_vege/dem_veg_'+domain+'.gdat'

# Path to directions folder
tc_path = SMpath+'timecoef/'

# Path to the flow directions fortran file for editing
tc_fortran = SMpath+'timecoef/timecoefs.f'

# Create the years variable for use later
print('Working on timecoefs')

# Output path
out_path = SMpath+'outputs/wo_assim/'

# The number of days in the simulation
num_days = str((datetime.strptime(ed_dt, '%Y-%m-%d')-datetime.strptime(st_dt, '%Y-%m-%d')).days+1)

# Run the replace line function in order to change the name of the input 
# and output files in the script
replace_line(tc_fortran, 9, '      parameter (nx='+nx+',ny='+ny+')\n')
replace_line(tc_fortran, 11, '      parameter (max_iter='+num_days+')\n')
replace_line(tc_fortran, 48, '     &  \'../topo_vege/dem_veg_'+domain+'.gdat\',\n')
replace_line(tc_fortran, 56, '     &  file=\'../outputs/wo_assim/swed.gdat\',\n')
replace_line(tc_fortran, 205, '      open (unit=71,file=\'../outputs/wo_assim/tc.gdat\',\n')

# Use line magic to change directories to the file locations
# Run the fortran script and save the output
get_ipython().run_line_magic('cd', '$tc_path')
get_ipython().system('gfortran -mcmodel=medium timecoefs.f')
get_ipython().system('./a.out')

# write a .ctl file for tcs
ctlpath = out_path+'tc.ctl'
disc_ctl('tc',ctlpath,nx,ny,xll,yll,st_dt,nt=num_days)
#     # Print out a metadata textfile
#     meta(domain,name,out_path)
print('tc gdat created for '+domain+' domain')


# ## Run HydroFlow

# In[5]:


# INPUTS: dir.gdat, water
# OUTPUTS: 

# Path to single dem/veg file
dem_veg = SMpath+'topo_vege/dem_veg_'+domain+'.gdat'

# Path to hydroflow folder
hf_path = SMpath+'hydroflow/'

# Path to the flow directions fortran file for editing
hf_fortran = SMpath+'hydroflow/hydroflow.f'

# Create the years variable for use later
print('Working on HydroFlow')

# Output path
out_path = SMpath+'outputs/wo_assim/'

# The number of days in the simulation
num_days = str((datetime.strptime(ed_dt, '%Y-%m-%d')-datetime.strptime(st_dt, '%Y-%m-%d')).days+1)


# Run the replace line function in order to change the name of the input 
# and output files in the script
replace_line(hf_fortran, 9, '      parameter (nx='+nx+',ny='+ny+',nxny=nx*ny)\n')
replace_line(hf_fortran, 10, '      parameter (max_iter='+num_days+')\n')
replace_line(hf_fortran, 37, '      deltax = '+clsz+'\n')
replace_line(hf_fortran, 38, '      deltay = '+clsz+'\n')
replace_line(hf_fortran, 49, '      open (unit=41,file=\'../watershed/watershed_index.txt\')\n')
replace_line(hf_fortran, 64, '     &  file=\'../outputs/wo_assim/rofx.gdat\',\n')
replace_line(hf_fortran, 86, '      open (81,file=\'../outputs/wo_assim/tc.gdat\',\n')
replace_line(hf_fortran, 280, '      open (unit=71,file=\'../outputs/wo_assim/disc.gdat\',\n')

# Use line magic to change directories to the file locations
# Run the fortran script and save the output
get_ipython().run_line_magic('cd', '$hf_path')
get_ipython().system('gfortran -mcmodel=medium hydroflow.f')
get_ipython().system('./a.out')

# write a .ctl file for discharge
ctlpath = out_path+'disc.ctl'
disc_ctl('disc',ctlpath,nx,ny,xll,yll,st_dt,nt=num_days)    
# Print out a metadata textfile
#meta(domain,name,out_path)
print('disc gdat created for '+domain+' domain')


# In[ ]:




