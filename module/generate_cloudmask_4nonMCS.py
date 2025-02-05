import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from datetime import datetime
from pathlib import Path

from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

def convert_Tb2CCSmask(ds_tb_image):

    ccs_binary = ds_tb_image.where(ds_tb_image <= 241, 0)
    ccs_binary = ccs_binary.where(ccs_binary == 0, 1)
    #ccs_binary.plot()
    labeled_image, num_features = ndimage.label(ccs_binary)
    ccs_binary = ccs_binary*0 + labeled_image
    num_sel = labeled_image[20,20] # targeted CCS 
    ccs_mask = ccs_binary.where(ccs_binary == num_sel, 0)
    ccs_mask = ccs_mask.where(ccs_mask == 0, 1)

    return ccs_mask

########################################

if __name__ == "__main__":

    # this script generates the cloudmask for non-MCS tracks identified by PyFLEXTRKR
    # based on the brightness temperature threshold 241 K. 

    year = int(sys.argv[1]) # selected year
    data_dir = Path('/scratch/wmtsai/featenv_analysis/dataset/NonMCS_FLEXTRKR_tropics/{}/environment_catalogs/VARS_2D'.format(year))
    ds_tb = xr.open_dataset(data_dir / 'NonMCS_FLEXTRKR_tropics_tb.merged.nc') # brightness temperature
    print('Input Tb file: {}'.format(data_dir / 'NonMCS_FLEXTRKR_tropics_tb.merged.nc'))

    ds_ccsmask_merge = [] # empty list to save outputs
    for track in ds_tb.tracks.values:
        ccsmask_track_list = []
        for time in ds_tb.time.values:
            tb_image = ds_tb.sel(tracks=track, time=time).tb
            ccs_mask = convert_Tb2CCSmask(tb_image)
            ccsmask_track_list.append(ccs_mask)
        # merge all times for a given track
        ds_ccsmask = xr.concat(ccsmask_track_list, dim=pd.Index(ds_tb.time.values, name='time'))
        ds_ccsmask_merge.append(ds_ccsmask)
    # merge all tracks
    ds_ccsmask_merge = xr.concat(ds_ccsmask_merge, dim=pd.Index(ds_tb.tracks.values, name='tracks'))
    ds_ccsmask_merge_xr = ds_ccsmask_merge.rename('cloudtracknumber_nomergesplit').to_dataset()
    ds_ccsmask_merge_xr.attrs['description'] = "cloud mask identified by contiguous pixels of Tb = 241K"    
    # save into the directory
    ds_ccsmask_merge_xr.to_netcdf(data_dir / 'NonMCS_FLEXTRKR_tropics_cloudtracknumber_nomergesplit.merged.nc')
    print(data_dir / 'NonMCS_FLEXTRKR_tropics_cloudtracknumber_nomergesplit.merged.nc')