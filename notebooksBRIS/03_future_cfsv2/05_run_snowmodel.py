#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
import os
import requests


######## USER INPUT HERE ONLY ##########

# Use this for the 2-year SnowModel run info
start_years_list = list(range(1989,2019))
# print(start_years_list)

# test this out with first 10 years
# start_years_list = [2018]
print(start_years_list)

###########################################

# Define some variables that don't change throughout the snowmodel runs
domain = 'BRIS'

# SM filepath
SMpath = '/nfs/attic/dfh/2020_NPRB/domain_'+domain+'/snowmodel2023_cfsv2/'
#SMpath = '/scratch/Nina/NPRB/domain_'+domain+'/snowmodel2023_cfsv2/'

# #path to NPRB domains
# domains_resp = requests.get("https://raw.githubusercontent.com/NPRB/02_preprocess_python/main/NPRB_domains.json")
# domains = domains_resp.json()

# Other variables
parFile = SMpath+'snowmodel.par'
incFile = SMpath+'code/snowmodel.inc'
compileFile = SMpath+'code/compile_snowmodel.script'
ctlFile = SMpath+'ctl_files/wo_assim/swed.ctl'
codepath = SMpath+'code'
preprocessFile = SMpath+'code/preprocess_code.f'
outputs_user = SMpath+'code/outputs_user.f'
micrometFile = SMpath+'code/micromet_code.f'


# In[6]:


# ### Function to edit text files docs


#function to edit SnowModel Files other than .par
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()



#Edit the par file to set parameters with new values
def edit_par(par_dict,parameter,new_value,parFile):
    lines = open(parFile, 'r').readlines()
    if par_dict[parameter][2] == 14 or par_dict[parameter][2] == 17 \
    or par_dict[parameter][2] == 18 or par_dict[parameter][2] == 19 \
    or par_dict[parameter][2] == 93 or par_dict[parameter][2] == 95 \
    or par_dict[parameter][2] == 97 or par_dict[parameter][2] == 100 \
    or par_dict[parameter][2] == 102 or par_dict[parameter][2] == 104 \
    or par_dict[parameter][2] == 107 or par_dict[parameter][2] == 108 \
    or par_dict[parameter][2] == 147 or par_dict[parameter][2] == 148 \
    or par_dict[parameter][2] == 149:
        text = str(new_value)+'\n'
    else:
        text = str(new_value)+'\t\t\t!'+par_dict[parameter][1]
    lines[par_dict[parameter][2]] = text
    out = open(parFile, 'w')
    out.writelines(lines)
    out.close()


#import baseline .par parameters
with open('/nfs/attic/dfh/2020_NPRB/data/json/par_base.json') as f:
    base = json.load(f)


#function to edit time-related parameters in .par 
def change_dates(styr):
    st = pd.to_datetime(str(styr)+'-10-01',format="%Y-%m-%d")
    ed = pd.to_datetime(str(styr+2)+'-09-30',format="%Y-%m-%d")
    edit_par(base,'iyear_init',str(st.year),parFile)
    edit_par(base,'imonth_init',str(st.month),parFile)
    edit_par(base,'iday_init',str(st.day),parFile)
    edit_par(base,'xhour_init',str(st.hour),parFile)
    edit_par(base,'max_iter',str((ed-st).days*4+4),parFile)
    edit_par(base,'met_input_fname','../../data/SMinputs/'+domain+'/mm_'+domain+'_wy'+str(st.year+81)+'-'+str(st.year+82)+'.dat',parFile)
    edit_par(base,'output_path_wo_assim','outputs_fut/',parFile)
    edit_par(base,'veg_ascii_fname','../../data/SMinputs/'+domain+'/'+domain+'_veg_fut.asc',parFile)


# In[6]:


def compile_snowmodel():
    # Move to code
    get_ipython().run_line_magic('cd', '$codepath')
    # Run compile script 
    get_ipython().system(' ./compile_snowmodel.script')


# In[7]:


def run_snowmodel():
    get_ipython().run_line_magic('cd', '$SMpath')
    get_ipython().system(' ./snowmodel')




for styr in start_years_list:
    print(styr)
    st = pd.to_datetime(str(styr)+'-10-01',format="%Y-%m-%d")
    print('editing .par file')
    
    #change date parameters in .par file
    change_dates(styr)
    
    # Compile snowmodel
    print('compiling snowmodel')
    compile_snowmodel()
    
    # run snowmodel
    print('running rnowmodel')
    run_snowmodel()
    
    # move .gdat files
    print('moving files')
    # starting name
    swedgin = SMpath+'outputs_fut/swed.gdat'
    roffgin = SMpath+'outputs_fut/roff.gdat'
    precgin = SMpath+'outputs_fut/prec.gdat'
    ssubgin = SMpath+'outputs_fut/ssub.gdat'
    tairgin = SMpath+'outputs_fut/tair.gdat'
    # final name
    swedgo = SMpath+'outputs_fut/swed_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat'
    roffgo = SMpath+'outputs_fut/roff_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat'
    precgo = SMpath+'outputs_fut/prec_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat'
    ssubgo = SMpath+'outputs_fut/ssub_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat'
    tairgo = SMpath+'outputs_fut/tair_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat'
    # move
    get_ipython().system(' cp $swedgin $swedgo')
    get_ipython().system(' cp $roffgin $roffgo')
    get_ipython().system(' cp $precgin $precgo')
    get_ipython().system(' cp $ssubgin $ssubgo')
    get_ipython().system(' cp $tairgin $tairgo')

    # move .ctl files
    # starting name
    swedcin = SMpath+'ctl_files/wo_assim/swed.ctl'
    roffcin = SMpath+'ctl_files/wo_assim/roff.ctl'
    preccin = SMpath+'ctl_files/wo_assim/prec.ctl'
    ssubcin = SMpath+'ctl_files/wo_assim/ssub.ctl'
    taircin = SMpath+'ctl_files/wo_assim/tair.ctl'
    # final name
    swedco = SMpath+'outputs_fut/swed_'+str(st.year+81)+'-'+str(st.year+82)+'.ctl'
    roffco = SMpath+'outputs_fut/roff_'+str(st.year+81)+'-'+str(st.year+82)+'.ctl'
    precco = SMpath+'outputs_fut/prec_'+str(st.year+81)+'-'+str(st.year+82)+'.ctl'
    ssubco = SMpath+'outputs_fut/ssub_'+str(st.year+81)+'-'+str(st.year+82)+'.ctl'
    tairco = SMpath+'outputs_fut/tair_'+str(st.year+81)+'-'+str(st.year+82)+'.ctl'
    # edit .ctl
    replace_line(swedcin, 0, 'DSET ^swed_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat\n')
    replace_line(roffcin, 0, 'DSET ^roff_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat\n')
    replace_line(preccin, 0, 'DSET ^prec_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat\n')
    replace_line(ssubcin, 0, 'DSET ^ssub_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat\n')
    replace_line(taircin, 0, 'DSET ^tair_'+str(st.year+81)+'-'+str(st.year+82)+'.gdat\n')
    # move
    get_ipython().system(' cp $swedcin $swedco')
    get_ipython().system(' cp $roffcin $roffco')
    get_ipython().system(' cp $preccin $precco')
    get_ipython().system(' cp $ssubcin $ssubco')
    get_ipython().system(' cp $taircin $tairco')


# In[ ]:




