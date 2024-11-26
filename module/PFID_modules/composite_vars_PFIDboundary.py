import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import warnings

if __name__ == '__main__':

    sys.path.append('/scratch/wmtsai/featenv_analysis/runscripts')
#    os.chdir('/scratch/wmtsai/featenv_analysis/runscripts')
    from PFID_environment_dilation import *

    # load the PFID catalogue data that contains variables (samples, ngrids, p_level)
    catalog_dir = Path('/scratch/wmtsai/featenv_analysis/dataset/test/')
    ds = xr.open_dataset(catalog_dir / 'ds_dilation_test2_compress.nc').compute()
   
    # loop for each identified PFID object
    ds_composite_merged = [] # (samples, grids_relative_to_boundary, p_level)
    for sample in ds.samples.values:
        print('processing sample number: {}'.format(sample))
        ds_reconst_xr = reconstruct_geocoords(ds.sel(samples=sample))
        ds_composite_xr = get_composite_PFID_boundary(ds_reconst_xr)
        ds_composite_merged.append(ds_composite_xr)
        
    ds_composite_merged_xr = xr.concat(ds_composite_merged, dim=pd.Index(ds.samples.values, name='samples'))
    # writeout 
    ds_composite_merged_xr.to_netcdf(catalog_dir / 'envs_composite_test2.nc')
