import os
ncore = "1"
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore
import sys
import json
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from functools import partial
import warnings

warnings.filterwarnings('ignore')

def process_vars_env_writeout(track, var):

    try:
        ds_env_vars = featenv.get_environment_vars_track(var_name=var, track_id=track,
                                                     lat_range=featenv.box_size_degree, lon_range=featenv.box_size_degree)

    except:
#        raise Exception('check track_id: {}'.format(track))
        print('track_id: {}'.format(track))
        dlon = featenv.lon_env[1] - featenv.lon_env[0]
        dlat = featenv.lat_env[1] - featenv.lat_env[0]
        nlon_grid = int(featenv.box_size_degree/dlon) + 1
        nlat_grid = int(featenv.box_size_degree/dlat) + 1

        data_track = featenv.track_data.sel(tracks=track)
        base_time = data_track['base_time'].values
        length_t = len(base_time)

        ds_env_vars = xr.Dataset(data_vars=dict(var_tmp=(['time','y','x'], np.zeros((length_t,nlat_grid,nlon_grid))*np.nan),
                                        lon=(['time','x'], np.zeros((length_t,nlon_grid))*np.nan),
                                        lat=(['time','y'], np.zeros((length_t,nlat_grid))*np.nan),
                                        base_time=(['time'],base_time)),
                         coords=dict(y=(['y'],np.arange(nlat_grid)),
                                     x=(['x'],np.arange(nlon_grid)),
                                     time=(['time'],np.arange(length_t)))
                        )
        ds_env_vars = ds_env_vars.rename_vars({'var_tmp':var})
        ds_env_vars.coords['tracks'] = [track]

    return ds_env_vars

if __name__ == '__main__':

#    config = int(sys.argv[1]) # year of tracks

    start_time = datetime.now()

    os.chdir('/scratch/wmtsai/featenv_analysis/config/')
    from feature_environment_module import *
    from multiprocessing import Pool

    # read feature and variable settings from .json files
    feature_json = open('feature_list.jsonc')
    variable_json = open('varible_list.jsonc')
    feature_settings = json.load(feature_json)
    variable_settings = json.load(variable_json)

    # call the feature-environemnt module
    featenv = ds_feature_environment()
    print('version: ', featenv.__version__)
    featenv.name = feature_settings['feature'][0]['name']
    featenv.feature_data_sources = feature_settings['feature'][0]['feature_sources']
    featenv.environmental_data_sources = feature_settings['feature'][0]['feature_environment_sources']
    featenv.track_frequency = feature_settings['feature'][0]['track_frequency']
    featenv.env_frequency = feature_settings['feature'][0]['track_frequency']
    featenv.feature_track = eval(feature_settings['feature'][0]['is_feature_track'])
    featenv.feature_mask = eval(feature_settings['feature'][0]['is_feature_mask'])
    featenv.box_size_degree = int(feature_settings['feature'][0]['box_size_degree'])    

    # matching default ERA-5
    featenv.lon_env = np.arange(0,360,0.25)
    featenv.lat_env = np.arange(-90,90.25,0.25)

    # locate and read the preprocessed track file
    featenv.track_data =  xr.open_dataset(feature_settings['feature'][0]['track_data'])
    # check dimensions: coords=[ tracks, time ] variables=[ base_time, meanlat, meanlon ]
    coords_track = []
    vars_track = []
    for i in featenv.track_data.dims:
        coords_track.append(i)
    for i in featenv.track_data.keys():
        vars_track.append(i)    

    a = set(coords_track) 
    b = set(['tracks','time'])
    c = set(vars_track)
    d = set(['base_time','meanlon','meanlat'])
    if (a & b) == {'time','tracks'} and (c & d) == {'base_time','meanlon','meanlat'}:
        print('Track data...ready: {}'.format(feature_settings['feature'][0]['track_data']))
    else:
        sys.exit('Incorret input format...Check the input file') # exit due to an incorrect track file

    # create directories according to the above descriptions
    year_process = str(featenv.track_data.isel(tracks=0).base_time[0].values)[:4] # YYYY
    main_dir = '/scratch/wmtsai/featenv_analysis/dataset/{}/{}/'.format(featenv.name, year_process)
    featenv.create_featenv_directory(main_dir)

    (featenv.track_data).to_netcdf(featenv.track_dir / 'track_geoinfo.nc')

    # extract feat-env data for individual tracks using multiprocessing
    track_sel = featenv.track_data.tracks.values
#    np.random.shuffle(track_sel) # shuffle to avoid I/O traffics due to competition 
    print('--------------------------------------')
    print('total tracks processed: {}'.format(len(track_sel)))
    print('--------------------------------------')

    # set up dictionary for variables, paths, and file strings informed from .json
    for n,var in enumerate(variable_settings['variable_inputs']):
        if n == 0:
            featenv.locate_env_data(var['var_name'], var['var_dir'])
        else:
            featenv.locate_env_data.update({var['var_name']: var['var_dir']})

    for n,var in enumerate(variable_settings['variable_inputs']):
        if n == 0:
            featenv.variable_format(var['var_name'], var['file_str'])
        else:
            featenv.variable_format.update({var['var_name']: var['file_str']})

    for n,var in enumerate(variable_settings['variable_inputs']):
        if n == 0:
            featenv.variable_infile(var['var_name'], var['varname_infile'])
        else:
            featenv.variable_infile.update({var['var_name']: var['varname_infile']})

    # ============= multiprocessing starts ===============

    # loops for designated variables:
    for var in [i for i in featenv.locate_env_data.keys()]:

        print('current variable: {}'.format(var))
        result = process_vars_env_writeout(track_sel[0], var=var)
        result.to_netcdf('/scratch/wmtsai/featenv_analysis/runscripts/test.nc') 
        quit()

        num_process = 12 # assign number of preocesses for this task
        pool = Pool(processes=num_process)
        result_list = pool.map_async(partial(process_vars_env_writeout, var=var), track_sel)
        pool.close()
        pool.join() 

        results = result_list.get()
        # merge all tracks
        data_merged = xr.concat(results, dim='tracks', compat='no_conflicts')   
        data_merged = data_merged.sortby('tracks')

        # writeout datasets
        check3d = [i for i in data_merged.dims if i == 'level']
        if check3d and len(data_merged.dims) > 2:
            out_dir = featenv.env3d_dir
        elif len(data_merged.dims) > 2:
            out_dir = featenv.env2d_dir
        data_merged.to_netcdf(out_dir / '{}_{}.merged.nc'.format(featenv.name, var), encoding={featenv.variable_infile[var]: {'dtype': 'float32'}})
        print(str(out_dir / '{}_{}.merged.nc'.format(featenv.name, var)) + '....saved')

    end_time = datetime.now()
    print('Data processing completed')
    print('Execution time spent: {}'.format(end_time - start_time))

