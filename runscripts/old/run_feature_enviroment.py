import os
ncore = "1"
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore
import sys
import xarray as xr
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from datetime import datetime
from pathlib import Path
import cartopy.crs as ccrs
import warnings

warnings.filterwarnings('ignore')

def process_vars_env_writeout(track):

    print('processing track number: {}'.format(track))
    ds_env_vars = featenv.get_environment_vars_track(track_id=track, lat_range=10, lon_range=10)
    ds_feat_vars = featenv.get_feature_vars_track(track_id=track, lat_range=10, lon_range=10)
    ds_vars = xr.merge([ds_env_vars, ds_feat_vars], compat='override') # some float differeces. TBD

    return track, ds_vars

if __name__ == '__main__':

    year_process = int(sys.argv[1]) # year of tracks

    start_time = datetime.now()

    os.chdir('/scratch/wmtsai/featenv_test/')
    from feature_environment_module import *
    from multiprocessing import Pool

    # call the feature-environemnt module
    featenv = ds_feature_environment()
    print('version: ', featenv.__version__)
    featenv.name = 'LPS_TempestExtremes'
    featenv.feature_data_sources = 'ERA5'
    featenv.environmental_data_sources = 'ERA5'
    featenv.track_frequency = '6-hourly'
    featenv.env_frequency = '6-hourly'
    featenv.feature_track = True
    featenv.feature_mask = False
    featenv.lon_env = np.arange(0,360,0.25)
    featenv.lat_env = np.arange(-90,90.25,0.25)

    print("Feature data sources:", featenv.feature_data_sources)
    print("Environmental data sources:", featenv.environmental_data_sources)

    # create directories according to the above descriptions
    main_dir = '/scratch/wmtsai/featenv_test/{}/{}/'.format(featenv.name, year_process)
    featenv.create_featenv_directory(main_dir)

    # 1. locate environment variables: variable names, direct paths
    env_dir = Path('/neelin2020/ERA-5/NC_FILES/')
    feat_dir = Path('/neelin2020/mcs_flextrkr/')
    tmp_dir = Path('/scratch/wmtsai/ERA-5/NC_FILES/')

    featenv.locate_env_data('T', env_dir)
    featenv.locate_env_data.update({'q': env_dir})
#    featenv.locate_env_data.update({'ua': env_dir})
#    featenv.locate_env_data.update({'va': env_dir})
#    featenv.locate_env_data.update({'omega': env_dir})
    featenv.locate_env_data.update({'sp': env_dir})
    featenv.locate_env_data.update({'2t': env_dir})
    featenv.locate_env_data.update({'2d': env_dir})
    featenv.locate_env_data.update({'mslp': tmp_dir})
    # add satellite gridded data
    featenv.locate_feature_data('precipitation', feat_dir)
    featenv.locate_feature_data.update({'tb': feat_dir})

#    print('Environmental data located: \n',featenv.locate_env_data)
#    print('Feature data located: \n',featenv.locate_feature_data)

    # 1. read preprocessed track file and save into the feature_environment directory
    track_dir = Path('/neelin2020/RGMA_feature_mask/LPS_ERA5')
    track_data = featenv.load_track_data(track_dir / 'ERA5_LPS_tracks_{}.nc'.format(year_process))
    featenv.track_data = track_data
    (featenv.track_data).to_netcdf(featenv.track_dir / '{}_geoinfo.{}.nc'.format(featenv.name, year_process))

    # 2. extract feat-env data for individual tracks using multiprocessing
    track_sel = featenv.track_data.tracks.values[:30]
    np.random.shuffle(track_sel) # faster for multiprocessing with I/O
    print('--------------------------------------')
    print('total tracks processed: {}'.format(len(track_sel)))
    print('--------------------------------------')

    pool = Pool(processes=15) # cpu numbers
    results = list(pool.imap_unordered(process_vars_env_writeout, track_sel, chunksize=2))
    track_list = []
    ds_list = []
    for i in range(len(results)):
        track_list.append(results[i][0])
        ds_list.append(results[i][1])
    ds_merged_xr = xr.concat(ds_list, dim=pd.Index(track_list, name='tracks'))
    ds_merged_xr = ds_merged_xr.sortby('tracks') # restore the track order 

    pool.close()
    pool.join()

    # 3. save feature and environmental variables accordingly
    for var in ds_merged_xr.keys():

        if var != 'base_time':
            ds = ds_merged_xr[var]
            check3d = [i for i in ds.dims if i == 'level']
            if check3d and len(ds.dims) > 2:
                out_dir = featenv.env3d_dir
            elif len(ds.dims) > 2:
                out_dir = featenv.env2d_dir

            print(out_dir)
            ds.to_netcdf(out_dir / '{}_{}_merged.nc'.format(featenv.name, var), encoding={var: {'dtype': 'float32'}})
            print('save file: {}_{}_merged.nc'.format(featenv.name, var))

    end_time = datetime.now()
    print('Data processing completed')
    print('Execution time spent: {}'.format(end_time - start_time))

