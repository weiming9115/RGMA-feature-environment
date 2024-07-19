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
    ds_env_vars = featenv.get_environment_vars_track(var_name=var, track_id=track, lat_range=10, lon_range=10)
    # save into temp directory
    ds_env_vars.to_netcdf(out_dir / '{}_vars_track_{}.nc'.format(featenv.name, str(track).zfill(5)))

if __name__ == '__main__':

    year_process = int(sys.argv[1]) # year of tracks

    start_time = datetime.now()

    os.chdir('/scratch/wmtsai/featenv_test/')
    from feature_environment_module import *
    from multiprocessing import Pool
#    from multiprocessing.pool import ThreadPool

    # call the feature-environemnt module
    featenv = ds_feature_environment()
    print('version: ', featenv.__version__)
    featenv.name = 'MCS_FLEXTRKR_tropics'
    featenv.feature_data_sources = 'GPM-IMERG;MERGE-IR'
    featenv.environmental_data_sources = 'ERA5'
    featenv.track_frequency = 'hourly'
    featenv.env_frequency = 'hourly'
    featenv.feature_track = True
    featenv.feature_mask = False
    featenv.lon_env = np.arange(0,360,0.25)
    featenv.lat_env = np.arange(-90,90.25,0.25)

    print("Feature data sources:", featenv.feature_data_sources)
    print("Environmental data sources:", featenv.environmental_data_sources)

    # create directories according to the above descriptions
    main_dir = '/scratch/wmtsai/featenv_test/{}/{}/'.format(featenv.name, year_process)
    featenv.create_featenv_directory(main_dir)

    # 1. read preprocessed track file and save into the feature_environment directory
    track_dir = Path('/scratch/wmtsai/temp_mcs/mcs_stats/mcs_tracks_non2mcs/featenv_track_input')
    track_data = featenv.load_track_data(track_dir / 'MCS_FLEXTRKR_tropics30NS.{}.nc'.format(year_process))
    featenv.track_data = track_data
    (featenv.track_data).to_netcdf(featenv.track_dir / '{}_geoinfo.{}.nc'.format(featenv.name, year_process))

    # 2. extract feat-env data for individual tracks using multiprocessing
    track_sel = featenv.track_data.tracks.values
    np.random.shuffle(track_sel) # shuffle to avoid I/O trafics
    print('--------------------------------------')
    print('total tracks processed: {}'.format(len(track_sel)))
    print('--------------------------------------')

    # set up variables desired
    # locate environment variables: variable names, direct paths
    env_dir = Path('/neelin2020/ERA-5/NC_FILES/')
    tmp_dir = Path('/scratch/wmtsai/ERA-5/NC_FILES/')

    featenv.locate_env_data('T',env_dir)
#    featenv.locate_env_data.update({'q': env_dir})
#    featenv.locate_env_data.update({'ua': env_dir})
#    featenv.locate_env_data.update({'va': env_dir})
#    featenv.locate_env_data.update({'omega': env_dir})    
#    featenv.locate_env_data.update({'sp': env_dir})
#    featenv.locate_env_data.update({'2t': env_dir})
#    featenv.locate_env_data.update({'2d': env_dir})
#    featenv.locate_env_data.update({'mslp': tmp_dir})

    # loops for designated variables:
    for var in [i for i in featenv.locate_env_data.keys()]:
        out_dir = featenv.envcats_dir / '{}'.format(var)
        os.system('mkdir -p {}'.format(out_dir))

#        process_vars_env_writeout(track_sel[0])

        for n, track_chunked in enumerate(np.array_split(track_sel, len(track_sel)//120)):
            print('chunk number: {}'.format(n))
            num_process = 12 # assign number of preocesses for this task
            pool = Pool(processes=num_process)
       
            chunksize, extra = divmod(len(track_chunked), num_process)
            if extra:
                chunksize+=1
            pool.map_async(process_vars_env_writeout, track_chunked, chunksize=chunksize)
            pool.close()
            pool.join()

    end_time = datetime.now()
    print('Data processing completed')
    print('Execution time spent: {}'.format(end_time - start_time))

