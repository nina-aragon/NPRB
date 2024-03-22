#!/usr/bin/env python
# coding: utf-8

# ## Calibrate HydroFlow

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
domain = 'YUKO'

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


# ## Run HydroFlow

# In[8]:


# INPUTS: dir.gdat, water
# OUTPUTS: 

# define slow and fast alpha parameters to be tested
alpha_s = ['10','20','30']
alpha_f = ['2','2','2']

# Path to single dem/veg file
dem_veg = SMpath+'topo_vege/dem_veg_'+domain+'.gdat'

# Path to hydroflow folder
hf_path = SMpath+'hydroflowcal2/'

# Path to the flow directions fortran file for editing
hf_fortran = SMpath+'hydroflowcal2/hydroflow.f'

# Output path
out_path = hf_path

# The number of days in the simulation
nt = str((datetime.strptime(ed_dt, '%Y-%m-%d')-datetime.strptime(st_dt, '%Y-%m-%d')).days+1)

for i in range(len(alpha_s)):
    # Run the replace line function in order to change the name of the input 
    # and output files in the script
    replace_line(hf_fortran, 102, '      alfa_s = '+alpha_s[i]+'\n')
    replace_line(hf_fortran, 104, '      alfa_f = '+alpha_f[i]+'\n')
    replace_line(hf_fortran, 280, '      open (unit=71,file=\'disc_s'+alpha_s[i]+'_f'+alpha_f[i]+'.gdat\',\n')

    # Use line magic to change directories to the file locations
    # Run the fortran script and save the output
    get_ipython().run_line_magic('cd', '$hf_path')
    get_ipython().system('gfortran -mcmodel=medium hydroflow.f')
    get_ipython().system('./a.out')

    # write a .ctl file for discharge
    ctlpath = out_path+'disc_s'+alpha_s[i]+'_f'+alpha_f[i]+'.ctl'
    Qfilename = 'disc_s'+alpha_s[i]+'_f'+alpha_f[i]+'.gdat'

    dt = datetime.strptime(st_dt, '%Y-%m-%d').strftime('%M:%SZ%d%b%Y')

    lines = ['DSET ^'+Qfilename,             'UNDEF -9999.0',             'OPTIONS BINPRECISION float32',             'XDEF '+nx+' LINEAR '+xll+' '+clsz,             'YDEF '+ny+' LINEAR '+yll+' '+clsz,             'ZDEF 1 LEVELS 1',             'TDEF '+str(nt)+' LINEAR '+dt+' 1dy',             'VARS 2',             'slow 1 0 SLOW',             'fast 1 0 FAST',             'ENDVARS']

    with open(ctlpath, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    f.close() 

