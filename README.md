# Overview
This repository contains all notebooks and scripts used to model historic (1991-2020) and future (2017-2100) discharge across the 4 arctic HUC4 subregions of Alaska. 

Historic and future datasets were generated using a suite of physically based, spatially distributed weather (MicroMet), energy-balance snow/ice melt (SnowModel), and runoff routing (HydroFlow) models:

* SnowModel - DOI: https://doi.org/10.1175/JHM548.1
* MicroMet - DOI: https://doi.org/10.1175/JHM486.1
* Hydroflow - DOI: https://doi.org/10.1175/JCLI-D-11-00591.1

[![DOI](https://zenodo.org/badge/776077718.svg)](https://zenodo.org/doi/10.5281/zenodo.10904627)

# Source data:
Weather forcing data: NCEP CFSv2: https://developers.google.com/earth-engine/datasets/catalog/NOAA_CFSV2_FOR6H

DEM data - GMTED 2010 elevation data: https://developers.google.com/earth-engine/datasets/catalog/USGS_GMTED2010_FULL

Landcover data - Copernicus landcover: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global

Evapotranspiration data: Hargreave’s reference evapotranspiration was derived from ClimateNA v7.3 1991-2020 period normal. https://adaptwest.databasin.org/pages/adaptwest-climatena/

Future simulations scaled precipitation and temperature CFSv2 weather forcing data using ClimateNA v7.3 data.  The ensemble mean of 8 CMIP6 AOGCMs following emission scenario SSP5-8.5 for the time period 2071-2100 was used to scale the data. https://adaptwest.databasin.org/pages/adaptwest-climatena/

Vegetation classes in the landcover file were updated in future simulations according to modeled changes in vegetation predicted by the ALFRESCO model. The multi-model mode of output for the time period 2070-2099 was used to update the landcover. 
http://data.snap.uaf.edu/data/IEM/Outputs/ALF/Gen_1a/alfresco_relative_spatial_outputs/vegetation_type


# NPRB Workflow: 

## dir: notebooksNPRB

### 01_Baseline_par2json.ipynb
Notebook to save baseline SnowModel parameters to .json. 
Inputs:
* snowmodel.par
Outputs:
* par_base.json

### 02_GEE_ak_huc4.ipynb
Notebook to download arctic domain shapefiles from GEE and define SnowModel bounding boxes for each domain. This notebook saves out a GeoJSON for each arctic domain. 
Inputs:
Outputs:
* Alaska HUC4 shapefiles and json files:
   * [domain].shp
   * [domain].json
* Bounding boxes for each domain
  
### 03_DomainBounds2json.ipynb
Notebook to create a JSON containing bounding parameters for all modeling domains. The output json file can be referenced by later scripts/notebooks using the requests package.
NOTE: This notebook needs to be run twice. On the first run, the user designates all domains, and includes the following details:
* Domain name
* Bounding box with latmax, latmin, lonmax, lonmin
* Start date
* End date
* Station projection (epsg:4326 in US)
* Model projection
After the NPRB_domains.json is pushed to github, 02_GEE_topoveg.ipynb can be run. Use the DEM or landcover ascii files for each domain to fill in the ncols, nrows, xll, yll values for each domain.
Inputs:
Outputs:
* NPRB_domains.json
  
## dir: notebooks_[domain]/01_calibrate_cfsv2

### 01_GEE_topoveg.ipynb
Notebook to pull GMTED 2010 elevation data and Copernicus landcover .tif files from GEE. Outputs include 4 ascii files that are prepped for Snow Model:
* Depression filled dem for SnowModel
* Depression filled dem for HydroFlow
* Landcover with SnowModel landcover classifications
* Latitude grid
* Longitude grid 
These ascii files will provide missing information (ncols, nrows, xll, yll) for the NPRB_domains.json file.
Inputs:
* NPRB_domains.json
Outputs:
* [domain]_dem.tif
* [domain]_dem.asc
* [domain]_demHF.asc
* [domain]_veg.tif
* [domain]_veg.asc
* [domain]_vegHF.asc
* [domain]_grid_lat.tif
* [domain]_grid_lat.asc
* [domain]_grid_lon.tif
* [domain]_grid_lon.asc

### 02_met_data-[domain]_cal.py 
Notebook to pull NCEP CFSv2 meteorological data from GEE and prep .dat files for input into SnowModel. Pulls met data for calibration time period, water years 2012-2018. 
Inputs:
* NPRB_domains.json
Outputs:
* mm_[domain]_wy[start year]-[end year].dat

### 03_get_streamflow_data.ipynb
Notebook to pull USGS stream gauges within each modeling domain. Outputs include streamgage discharge time series .csv, geodataframe of stations in domain with station metadata, and projected geodataframe of stations in domain with station metadata.
Inputs:
* NPRB_domains.json
Outputs:
* [domain]_discharge_cfs_[start date]_[end date].csv
* [domain]_gage_meta_[start date]_[end date].geojson
* [domain]_gage_meta_proj_[start date]_[end date].geojson

### 04_q_data_years.ipynb
Notebook to determine which stations to use for HydroFlow calibration based on data availability and size of contributing watershed. 
Inputs:
* NPRB_domains.json
* [domain].json
Outputs:

### In QGIS
Once you have the huc4 geojson, the streamgage station shapefiles and the filled dem downloaded - you need to do some work in QGIS to determine the upslope area of each stream gauge. 
A detailed description of the workflow can be found here: https://docs.google.com/document/d/1Urdpp41tCZp0ZptHDdRD2HQ3ZznolhWA0Zaa_szcgfc/edit?usp=sharing
Here is the overview:
1. Load DEM, huc4 shapefile, and gage shapefile
2. Reproject huc4 shape to match DEM
2. Clip dem to HUC4
3. Fill Sinks tool
4. Strahler Order tool
5. Using station locations, use Upslope Area tool to save out .tif files with upslope area for each gage. 
6. Rasterize the huc4 shapefile to use for masking later. 
Outputs:
* [gage id].tif files

### 05_convert_upslope_area2netcdf.ipynb
This notebook combines .tif files that were exported from QGIS then into a netcdf.
Inputs:
* NPRB_domains.json
* [gage id].tif files
Outputs:
* upslope.nc

### 06_baseline_SM.py 
Notebook to run SnowModel using baseline .par parameters. Saves out daily prec, tair, swed, swemelt, Q, and snow depth data. 
Inputs:
* mm_[domain]_wy[start year]-[end year].dat
* [domain]_dem.asc
* [domain]_veg.asc
* [domain]_grid_lat.asc
* [domain]_grid_lon.asc
Outputs:
* tair.gdt
* prec.gdat
* roff.gdat
* ssub.gdat
* swed.gdat

### 07_hydroflow_static.ipynb
Notebook to create the static files required for hydroflow. This includes:
* A grads file that combines the dem and vegetation ascii files
* A flow direction file
* A watershed delineation file
Inputs:
* NPRB_domains.json
* [domain]_dem.asc
* [domain]_veg.asc
Outputs:
* dem_veg_[domain].ctl
* dem_veg_[domain].gdat
* dir.ctl
* dir.gdat
* watershed.ctl
* watershed.gdat

### 08_find_domain_watersheds.ipynb
Notebook to rasterize the domain vector file and make a list of the watersheds that fall within the huc4 associated with the modeling domain.
Inputs:
* NPRB_domains.json
* watershed.ctl
* watershed.gdat
* [domain].json
Outputs:
* [domain]huc4_DEM.tif
* watersheds.csv

### 09_climateNA_et_clip2domain.ipynb
Notebook to clip and reproject climate NA et data to domain of interest. All negative ET values are removed. Note: Hargreaves ET equation was applied to temperature data and lookup table was used to compute ET values.
Inputs:
* NPRB_domains.json
* [domain]_dem.tif
* Normal_1991_2020_Eref[month].tif
* ensemble_8GCMs_ssp585_2071_2100_Eref[month].tif
Outputs:
* historic_monthly_et.nc
* future_monthly_et.nc

### 10_climateNA_et_daily.ipynb
Notebook to convert monthly ET data to daily values using second order polynomial interpolation.
Inputs:
* NPRB_domains.json
* [domain]_dem.tif
* historic_monthly_et.nc
* future_monthly_et.nc
Outputs:
* historic_daily_et.nc
* future_daily_et.nc

### 11_roff_remove_et.ipynb
Notebook to remove daily ET from the SnowModel runoff in order to create a modified runoff to input to HydroFlow. Daily ET is also modified to reach a maximum value equal to the daily runoff. 
Inputs:
* NPRB_domains.json
* historic_daily_et.nc
* future_daily_et.nc
* roff.ctl
* roff.gdat
Outputs:
* rofx.ctl
* rofx.gdat
* etx.ctl
* etx.gdat

### 12_fetch_station.ipynb 
Notebook to get SNOTEL and SCAN station SWE, PREC, TEMP metadata and time series data within a modeling domain to be used for SnowModel calibration. At the end of the notebook there is a section to check the data for outliers.
Inputs:
* NPRB_domains.json
Outputs:
* Station_sites_TOBScelcius_[domain]_[start date]_[end date].geojson
* Station_data_TOBScelsius_[start date]_[end date].csv
* Station_sites_PRmeters_[domain]_[start date]_[end date].geojson
* Station_data_PRmeters_[start date]_[end date].csv
* Station_sites_SWEDmeters_[domain]_[start date]_[end date].geojson
* Station_data_SWEDmeters_[start date]_[end date].csv

### 13_calc_prec_cf.ipynb
Notebook to calculate the precipitation correction factor based on the simplified water balance: cf = (specQ+specET)/specP. An overall cf is calculated at the area-weighted average of the gaged parts of each modeling domain. 
Inputs:
* NPRB_domains.json
* [domain]_discharge_cfs_[start date]_[end date].csv
* [domain]_gage_meta_proj_[start date]_[end date].geojson
* upslope.nc
* etx.ctl
Outputs:
* [domain]_precCF_[start date]_[end date]_cfsv2.csv

### 14_build_snowmodel_line_file.ipynb
Notebook to create a file to run SnowModel in line mode for the calibration. This notebook generates input files so that Snowmodel is only run at the cell(s) that correspond to SNOTEL station data. 
Inputs:
* NPRB_domains.json
* [domain]_dem.asc
* [domain]_veg.asc
* [domain]_grid_lat.asc
* Station_sites_SWEDmeters_[domain]_[start date]_[end date].geojson
Outputs:
* [domain]_dem_line.asc
* [domain]_veg_line.asc
* [domain]_grid_lat_line.asc
* [domain]_grid_lon_line.asc
* snowmodel_line_pts.dat

### 15_run_hydroflow_baseline.py
Script to run hydroflow with baseline alpha parameters.
Inputs:
* NPRB_domains.json
* dem_veg.gdat
* watershed.gdat
* dir.gdat
Outputs:
* tc.ctl
* tc.gdat
* disc.ctl
* disc.gdat

### 16_calibrate_SnowModel.py
Notebook to run SnowModel in line mode at the pixels that contain SNOTEL stations. Approximately 10000 calibration parameters are tested. Outputs include a .csv with all combinations of calibration parameters, a .nc file containing skill scores ['R2','MBE','RMSE','NSE','KGE'] for each combination of calibration parameters and a .nc file containing the SM output SWE for each combination of calibration parameters.
Inputs:
* mm_[domain]_wy[start year]-[end year].dat
* [domain]_precCF_[start date]_[end date]_cfsv2.csv
* [domain]_dem.asc
* [domain]_veg.asc
* [domain]_grid_lat.asc
* [domain]_grid_lon.asc
* par_base.json
* Station_sites_SWEDmeters_[domain]_[start date]_[end date].geojson
* Station_data_SWEDmeters_[start date]_[end date].csv
Outputs:
* cal_params_[timestamp].csv
* calibration_[timestamp].nc
* swe_[timestamp].nc

### 17_calibration_postprocess.ipynb
Notebook to evaluate the calibration results and identify the top calibration parameters for SM. At the end of the notebook, the .par file is updated with the top calibration parameters. 
Inputs:
* cal_params_[timestamp].csv
* calibration_[timestamp].nc
* swe_[timestamp].nc
* Station_sites_SWEDmeters_[domain]_[start date]_[end date].geojson
* Station_data_SWEDmeters_[start date]_[end date].csv
Outputs:
* Edited .par file with top calibration parameters

### 18_runSMcal.txt
Run the calibrated SM (using the new .par file) from terminal.
Outputs:
* tair.gdt
* prec.gdat
* roff.gdat
* ssub.gdat
* swed.gdat

### 19_roff_remove_et.ipynb
Notebook to remove daily ET from the calibrated SnowModel runoff in order to create a modified runoff to input to HydroFlow. Daily ET is also modified to reach a maximum value equal to the daily runoff. 
Inputs:
* NPRB_domains.json
* historic_daily_et.nc
* roff.ctl
* roff.gdat
Outputs:
* rofx.ctl
* rofx.gdat
* etx.ctl
* etx.gdat

### 20_calibrate_hydroflow.py
Notebook to calibrate hydroflow by testing multiple values for slow and fast alpha parameters. 
Inputs:
* NPRB_domains.json
* dem_veg.gdat
* watershed.gdat
* dir.gdat
Outputs:
* disc_s[slow alpha]_f[fast alpha].ctl
* disc_s[slow alpha]_f[fast alpha].gdat

### 21_get_gage_index_[domain].ipynb
This notebook requires manual examination of each stream gage to retrieve the i, j pair from the HydroFlow data. Manual corrections are often required to match the station location to the stream channel network that was determined in hydroflow. Stations are plotted on folium maps using satellite imagery to help determine where the station should be located. 
Inputs:
* NPRB_domains.json
* [domain]_discharge_cfs_[start date]_[end date].csv
* [domain]_gage_meta_proj_[start date]_[end date].geojson
* [domain]_dem.asc
* disc.ctl
* disc.gdat
* [domain].json
Outputs:
* [domain]_gage_meta_proj_[start date]_[end date].geojson (with station i,j coordinates)

### 22_calibrate_hydroflow_eval.ipynb
Notebook to determine the best performing slow and fast alpha parameters using KGE scores at all gages of interest. .png images are created for each stream gage that show the observed and modeled streamflow for each set of alpha parameters as well as the corresponding KGE score. 
Inputs:
* NPRB_domains.json
* [domain]_discharge_cfs_[start date]_[end date].csv
* [domain]_gage_meta_proj_[start date]_[end date].geojson
* disc_s[slow alpha]_f[fast alpha].ctl
* disc_s[slow alpha]_f[fast alpha].gdat
Outputs:
* HFcal_USGS_[gage id].png

### 23_coast_mask.ipynb
Notebook to identify coast pixels for each domain and complete a water balance to check the modeling workflow over the calibration period.
Inputs:
* NPRB_domains.json
* dem_veg_[domain].ctl
* watershed.ctl
* watersheds.csv
* prec_cal.ctl
* ssub_cal.ctl
* roff_cal.ctl
* rofx_cal.ctl
* swed_cal.ctl
* etx_cal.ctl
* disc_s[slow alpha]_f[fast alpha].ctl (for top calibration run)
Outputs:
* coast_mask.nc
* [domain]_cal_H2Obudget_table.csv
* [domain]cal_waterbudg.png

## dir: notebooks_[domain]/02_historic_cfsv2

### 01_met_data-BEAU.py
Notebook to pull meteorological data from GEE and prep .dat files for input into SnowModel. Pulls met data for 2-y intervals to be used in creating the long term climatology. 
Inputs:
* NPRB_domains.json
Outputs:
* mm_[domain]_wy[start year]-[end year].dat

### 02_run_snowmodel.py
Notebook to run SnowModel 2-y intervals using calibrated .par parameters. Saves out daily prec, tair, swed, swemelt, Q, and snow depth data. 
Inputs:
* mm_[domain]_wy[start year]-[end year].dat
* [domain]_dem.asc
* [domain]_veg.asc
* [domain]_grid_lat.asc
* [domain]_grid_lon.asc
Outputs:
* tair_[start year]-[end year].gdt
* prec_[start year]-[end year].gdat
* roff_[start year]-[end year].gdat
* ssub_[start year]-[end year].gdat
* swed_[start year]-[end year].gdat

### 03_removeET.ipynnb
Notebook to remove daily ET from the 2-year SnowModel runoff in order to create a modified runoff to input to HydroFlow. Daily ET is also modified to reach a maximum value equal to the daily runoff. 
Inputs:
* NPRB_domains.json
* historic_daily_et.nc
* roff[start year]-[end year].ctl
* roff[start year]-[end year].gdat
Outputs:
* rofx[start year]-[end year].ctl
* rofx[start year]-[end year].gdat
* etx[start year]-[end year].ctl
* etx[start year]-[end year].gdat

### 04_runHF.py
Script to run hydroflow in 2-year intervals with calibrated alpha parameters.
Inputs:
* NPRB_domains.json
* dem_veg.gdat
* watershed.gdat
* dir.gdat
Outputs:
* tc[start year]-[end year].ctl
* tc[start year]-[end year].gdat
* disc[start year]-[end year].ctl
* disc[start year]-[end year].gdat

### 05_results.ipynb
Notebook to extract and concatenate the second year of gridded modeled outputs and save them as a single 30-year .nc file. The gridded discharge file includes a mask indicating coastal pixels. A second coastal discharge .nc file is saved containing daily discharges only at the coastal grid cells. This notebook generates annual summaries and monthly climatologies of disc, prec, et, subb, swed, and tair. Timeseries plots are also generated for each variable.  
Inputs:
* tair_[start year]-[end year].gdat
* prec_[start year]-[end year].gdat
* roff_[start year]-[end year].gdat
* ssub_[start year]-[end year].gdat
* swed_[start year]-[end year].gdat
* disc[start year]-[end year].gdat
Outputs:
* [domain]_tair_wy1991-2020.nc
* [domain]_prec_wy1991-2020.nc
* [domain]_roff_wy1991-2020.nc
* [domain]_ssub_wy1991-2020.nc
* [domain]_swed_wy1991-2020.nc
* [domain]_disc_wy1991-2020.nc
* [domain]_disc_coast_wy1991-2020.nc
* [domain]_annual_summary_wy1991-2020.csv
* [domain]_monthly_climatology_wy1991-2020.csv

## dir: notebooks_[domain]/02_future_cfsv2

### 01_climateNA_clip2domain.ipynb
Notebook to clip and reproject climate NA monthly minimum, maximum and average temperature data and monthly precipitation data to the domain of interest. The change in each of these variables is calculated relative to the past following: T change = future - historic, P change = future/historic. The ensemble mean of 8 CMIP6 AOGCMs following emission scenario SSP5-8.5 for the time period 2071-2100 was used to scale the data. 
Inputs:
* NPRB_domains.json
* [domain]_dem.tif
* Normal_1991_2020_Tave[month].tif
* ensemble_8GCMs_ssp585_2071_2100_Tave[month].tif
* Normal_1991_2020_Tmax[month].tif
* ensemble_8GCMs_ssp585_2071_2100_Tmax[month].tif
* Normal_1991_2020_Tmin[month].tif
* ensemble_8GCMs_ssp585_2071_2100_Tmin[month].tif
* Normal_1991_2020_PPT[month].tif
* ensemble_8GCMs_ssp585_2071_2100_PPT[month].tif
Outputs:
* monthly_climatena_dif.nc

### 02_climateNA_daily.py
Notebook to convert monthly temperature and precipitation change data to daily values using second order polynomial interpolation.
Inputs:
* NPRB_domains.json
* [domain]_dem.tif
* monthly_climatena_dif.nc
Outputs:
* daily_climatena_dif.nc

### 03_future_met.py
Notebook to pull meteorological data from GEE and prep future .dat files for input into SnowModel. 2-year CFSv2 data is scaled to reflect the changes in temperature and precipitation data that was calculated in 02_climateNA_daily.py. The output is 2-year met forcing data spanning water years 2071-2100. 
Inputs:
* NPRB_domains.json
* daily_climatena_dif.nc
Outputs:
* mm_[domain]_wy[start year]-[end year].dat

### 04_landcover.ipynb
Notebook to modify the input landcover file in order to reflect vegetation changes modeled by the ALFRESCO model. The multi-model mode of output for the time period 2070-2099 was used to update the landcover. 
Inputs:
* NPRB_domains.json
* [domain]_veg.tif
* alfresco_relative_vegetation_change_1950-2008_historical.tif
* alfresco_relative_vegetation_change_2070-2099[CMIP6 model].tif
* [domain]_demHF.asc
Outputs:
* [domain]_veg_fut.tif
* [domain]_veg_fut.asc
* dem_veg_[domain]_fut.ctl
* dem_veg_[domain]_fut.gdat

### 05_run_snowmodel.py
Notebook to run future SnowModel 2-y intervals using calibrated .par parameters. Saves out daily prec, tair, swed, swemelt, Q, and snow depth data. 
Inputs:
* mm_[domain]_wy[start year]-[end year].dat
* [domain]_dem.asc
* [domain]_veg_fut.asc
* [domain]_grid_lat.asc
* [domain]_grid_lon.asc
Outputs:
* tair_[start year]-[end year].gdt
* prec_[start year]-[end year].gdat
* roff_[start year]-[end year].gdat
* ssub_[start year]-[end year].gdat
* swed_[start year]-[end year].gdat

### 06_removeET.ipynnb
Notebook to remove daily ET from the future 2-year SnowModel runoff in order to create a modified runoff to input to HydroFlow. Daily ET is also modified to reach a maximum value equal to the daily runoff. 
Inputs:
* NPRB_domains.json
* future_daily_et.nc
* roff[start year]-[end year].ctl
* roff[start year]-[end year].gdat
Outputs:
* rofx[start year]-[end year].ctl
* rofx[start year]-[end year].gdat
* etx[start year]-[end year].ctl
* etx[start year]-[end year].gdat

### 07_runHF.py
Script to run future hydroflow in 2-year intervals with calibrated alpha parameters.
Inputs:
* NPRB_domains.json
* dem_veg_[domain]_fut.gdat
* watershed.gdat
* dir.gdat
Outputs:
* tc[start year]-[end year].ctl
* tc[start year]-[end year].gdat
* disc[start year]-[end year].ctl
* disc[start year]-[end year].gdat

### 08_results.ipynb
Notebook to extract and concatenate the second year of gridded modeled outputs and save them as a single 30-year .nc file. The gridded discharge file includes a mask indicating coastal pixels. A second coastal discharge .nc file is saved containing daily discharges only at the coastal grid cells. This notebook generates annual summaries and monthly climatologies of disc, prec, et, subb, swed, and tair. Time series plots are also generated for each variable.  
Inputs:
* tair_[start year]-[end year].gdat
* prec_[start year]-[end year].gdat
* roff_[start year]-[end year].gdat
* ssub_[start year]-[end year].gdat
* swed_[start year]-[end year].gdat
* disc[start year]-[end year].gdat
Outputs:
* [domain]_tair_wy2071-2100.nc
* [domain]_prec_wy2071-2100.nc
* [domain]_roff_wy2071-2100.nc
* [domain]_ssub_wy2071-2100.nc
* [domain]_swed_wy2071-2100.nc
* [domain]_disc_wy2071-2100.nc
* [domain]_disc_coast_wy2071-2100.nc
* [domain]_annual_summary_wy2071-2100.csv
* [domain]_monthly_climatology_wy2071-2100.csv
