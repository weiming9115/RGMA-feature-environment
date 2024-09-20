import os
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

if __name__ == '__main__':

    start_time = datetime.now()

    os.chdir('/scratch/wmtsai/featenv_test/')
    from feature_environment_module import *

    # call the feature-environemnt module
    featenv = ds_feature_environment()
    featenv.name = 'LPS_TempestExtremes'
    year_process = int(sys.argv[1]) # year of tracks
 
    featenv_dir = Path('/scratch/wmtsai/featenv_test/')
    envtrack_dir = Path(featenv_dir / '{}/{}/feature_catalogs/track'.format(featenv.name,year_process))
    ds_file = xr.open_dataset(list(envtrack_dir.glob('*_vars_track_*'))[0])

    for var in ds_file.keys():
        track_list = []
        ds_list = []
        if (var != 'base_time' ) and (var != 'utc_date'):
            for track in list(envtrack_dir.glob('*_vars_track_*')): # all tracks 
                ds_var = xr.open_dataset(track)[var]
                ds_list.append(ds_var)
                track_list.append(str(track.name)[-8:-3])
            ds_merged = xr.concat(ds_list, pd.Index(track_list, name='tracks'))
            ds_merged = ds_merged.sortby('tracks')

            check3d = [i for i in ds_merged.dims if i == 'level']
            if check3d and len(ds_merged.dims) > 2:
                out_dir = featenv.env3d_dir
            elif len(ds_merged.dims) > 2:
                out_dir = featenv.env2d_dir

            ds_merged.to_netcdf(out_dir / '{}_{}_merged.nc'.format(featenv.name, var), encoding={var: {'dtype': 'float32'}})
            print('save file: {}_{}_merged.nc'.format(featenv.name, var))

    end_time = datetime.now()
    print('Data processing completed')
    print('Execution time spent: {}'.format(end_time - start_time))
