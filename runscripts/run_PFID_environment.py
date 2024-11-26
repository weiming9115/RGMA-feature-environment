import os
import sys
import json
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    os.chdir('/scratch/wmtsai/featenv_analysis/runscripts')    
    from PFID_environment_dilation import *

    # read config jsonc file from ../config
    os.chdir('/scratch/wmtsai/featenv_analysis/config/config_pid_env')
    feature_json = open('feature_list_default.jsonc')
    variable_json = open('varible_list.default.jsonc')
    feature_settings = json.load(feature_json)
    variable_settings = json.load(variable_json)
    
    ##### fixed parameters. do not change ######
    PFID_catalog = {'AR':1, 'FT':2, 'MCS': 3, 'LPS': 4, 
                    'AR-FT': 5, 'AR-MCS': 6, 'AR-LPS': 7,
                    'Front-MCS': 8,' Front-LPS': 9, 'MCS-LPS': 10,
                    'AR-Front-MCS': 11, 'AR-Front-LPS': 12, 'AR-MCS-LPS': 13,
                    'Front-MCS-LPS': 14, 'AR-Front-MCS-LPS': 15, 
                    'Unexp': 16,
                    'DC': 17, 'ND': 18, 'ST': 19, 'DZ': 20}
    ###############

    PFID_select = feature_settings['PFID'][0]['feature_comb']
    PFID_number = PFID_catalog[PFID_select]
    print('selected PFID : {}, number: {}'.format(PFID_select, PFID_number))
    year = feature_settings['PFID'][0]['year']
    month = feature_settings['PFID'][0]['month']

    pid_dir = Path('/scratch/wmtsai/GPM_feature_precip/{}/'.format(year))
    era5_dir = Path('/neelin2020/ERA-5/NC_FILES/{}/'.format(year))
    #####  load all input data #####
    ds = xr.open_dataset(pid_dir / 'GPM-IMERG_feature_precip_{}_{}.nc'.format(year,month)) # PFID
    ds = coordinates_processors(ds).sel(lat=slice(-60,60))
    ds_feat = xr.open_dataset(pid_dir / 'GPM_feature_merged_{}_v4.nc'.format(month)) # feature mask + precip
    ds_feat = coordinates_processors(ds_feat).sel(lat=slice(-60,60))
    ar_tag = ds_feat.ar_tag
    front_c_tag = ds_feat.front_c_tag
    front_w_tag = ds_feat.front_w_tag
    lps_tag = ds_feat.lps_tag
    mcs_tag = ds_feat.mcs_tag 
    prec_gpm = ds_feat.precipitationCal

    # creat a list of data for all variables
    env_vars_files = []
    for i in range(len(variable_settings['variable_inputs'])):
   
        var_info = variable_settings['variable_inputs'][i]
        var = var_info['var_name']
        varname_infile = var_info['varname_infile']
        data_dir = Path(var_info['var_dir'])
        data_str = var_info['file_str'] 
        # modify the default file string with datetime info
        tmp = data_str.replace('X',var).replace('YYYY',year).replace('MM',month)
#        tmp = tmp.replace('DD',day).replace('HH',hour)
        filename = data_dir /'{}'.format(year)/ tmp
        env_vars_files.append([varname_infile, filename])
    ###############

    precip_id = ds.precip_id # get PFID from the inserted dataset
    ds_PFID = precip_id.where(precip_id == PFID_number, 0) # 
    binary_PFID = ds_PFID.where(ds_PFID == 0, 1) # make it as a binary map (0, 1) for labeling

    ## 0. time-loops for a given month
#    for t in range(len(ds_PFID.time)):
    for t in range(1):
        binary_map = binary_PFID.isel(time=t)
        # label precipitation features associated with the AR-FT union
        (label, num) = measure.label(binary_map.values, return_num=True, connectivity=2)
        ds_pf = xr.Dataset(data_vars=dict(p_feat = (['lat','lon'], label)),
                           coords=dict(lat = (['lat'], binary_map.lat.values),
                                       lon = (['lon'], binary_map.lon.values)),
                           attrs=dict(description='precpitation of the specific PFID binary mask'))
        ds_pf['p_feat'] = ds_pf['p_feat'].where(ds_pf['p_feat'] > 0)
        
        print('Time: {}, Number: {}'.format(str(binary_map.time.values), num))

        ds_dilation_object = []
        for num_sel in np.arange(1,num+1):
    
            ds_pf_binary = ds_pf['p_feat'].where(ds_pf['p_feat'] == num_sel, 0)
            ds_pf_binary = ds_pf_binary.where(ds_pf_binary == 0, 1)
    
            if np.sum(ds_pf_binary) >= 20:
                print('processing precipitation feature number: {}'.format(num_sel))

                ds_dilation = generate_dilation_featmask(ds_pf_binary, dilation_ngrids=15)
                ds_dilation_1d = generate_dilated_dataset(ds_dilation, ds_dilation.pf_mask)
                ds_dilation_prec = generate_dilated_dataset(ds_dilation, prec_gpm.isel(time=t))

                ds_dilation_ar = generate_dilated_dataset(ds_dilation, ar_tag.isel(time=t))
                ds_dilation_ft_c = generate_dilated_dataset(ds_dilation, front_c_tag.isel(time=t))
                ds_dilation_ft_w = generate_dilated_dataset(ds_dilation, front_w_tag.isel(time=t))
                ds_dilation_lps = generate_dilated_dataset(ds_dilation, lps_tag.isel(time=t))
                ds_dilation_mcs = generate_dilated_dataset(ds_dilation, mcs_tag.isel(time=t))
             
                ds_dilation_vars = []
                for (var_name, file_name) in env_vars_files:
                    ds_var = xr.open_dataset(file_name)[var_name]
                    ds_var = coordinates_processors(ds_var)
                    ds_dilation_vars.append(generate_dilated_dataset(ds_dilation, ds_var.isel(time=t)))

                ds_dilation_merged = xr.merge([ds_dilation_1d, ds_dilation_prec,
                                               ds_dilation_ar, ds_dilation_ft_c, ds_dilation_ft_w, ds_dilation_lps,
                                               ds_dilation_mcs] + ds_dilation_vars, combine_attrs='no_conflicts')

                ds_dilation_object.append(ds_dilation_merged)
        # merge all the filtered samples into one dataset
        ds_dilation_object_xr = xr.concat(ds_dilation_object, dim=pd.Index(np.arange(len(ds_dilation_object))
                                          , name='samples'), combine_attrs='no_conflicts')    
        ds_dilation_object_xr.attrs['selected_PFID'] = PFID_select
        ds_dilation_object_xr.attrs['created_date'] = str(date.today())
   
    out_dir = Path('/scratch/wmtsai/featenv_analysis/dataset/test')
    ds_dilation_object_xr.to_netcdf(out_dir/ 'ds_dilation_test2.nc')
