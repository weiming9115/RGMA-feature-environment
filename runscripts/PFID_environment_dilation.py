import os
import sys
import xarray as xr
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from pathlib import Path
from skimage import measure
import cartopy.crs as ccrs
import warnings

# for stereogrphic projection (converting lat-lon to physical distance (km))
from geopy.distance import geodesic
from scipy.ndimage import binary_dilation, binary_erosion
warnings.filterwarnings('ignore')

def coordinates_processors(data):
    """
    converting longitude/latitude into lon/lat
    """
    coord_names = []
    for coord_name in data.coords:
        coord_names.append(coord_name)

    if (set(coord_names) & set(['longitude','latitude'])): # if coordinates set this way...
        data2 = data.rename({'latitude': 'lat'})
        data2 = data2.rename({'longitude': 'lon'})
    else:
        data2 = data

    # check if latitutde is decreasing
    if (data2.lat[1] - data2.lat[0]) < 0:
        data2 = data2.reindex(lat=list(reversed(data2.lat))) # flipping latitude accoordingly

    # check if longitude is -180 to 180
    # if so, reconstructing into 0 to 360
    if (data2.lon.min() < 0):
        data2['lon'] = xr.where(data2['lon'] < 0, data2['lon'] + 360, data2['lon'])
        data2 = data2.sortby('lon')

    return data2

def generate_dilation_featmask(ds_pf_binary, dilation_ngrids):

    ds_pf_binary_dilation = binary_dilation(ds_pf_binary, iterations=dilation_ngrids).astype('int')
    ds_dilation = xr.Dataset(data_vars=dict(pf_mask = (['lat','lon'], ds_pf_binary.values),
                                            env_mask = (['lat','lon'], ds_pf_binary_dilation-ds_pf_binary.values)),
                       coords=dict(lat = (['lat'], ds_pf_binary.lat.values),
                                   lon = (['lon'], ds_pf_binary.lon.values)),
                       attrs=dict(description='dilation of the specified PFID binary mask',
                                  dilation_ngrids=dilation_ngrids))
    
    return ds_dilation

def generate_dilated_dataset(ds_dilation, var_input):
        
    # extract feat_mask and env_mask grid points
    (idx_lat, idx_lon) = np.where(ds_dilation.pf_mask + ds_dilation.env_mask == 1)
    ds_list = []
    for i,j in zip(idx_lat, idx_lon):
        ds_list.append(var_input.sel(lat=ds_dilation.lat[i], lon=ds_dilation.lon[j]))
    ds_array = np.asarray(ds_list)
        
    lat_1d = ds_dilation.lat.isel(lat=idx_lat).values
    lon_1d = ds_dilation.lon.isel(lon=idx_lon).values
    
    if len(ds_array.shape) == 1: # (ngrids)
        ds_dilation_1d = xr.Dataset(data_vars = dict(var_temp = (['ngrids'], ds_array),
                                                     lat = (['ngrids'], lat_1d),
                                                     lon = (['ngrids'], lon_1d)),
                                    coords = dict(ngrids = (['ngrids'], np.arange(len(idx_lat)))),
                                    attrs = dict(description = 'extracted grids of feature-environment',
                                                 lat_res = (ds_dilation.lat[1] - ds_dilation.lat[0]).values,
                                                 lon_res = (ds_dilation.lon[1] - ds_dilation.lon[0]).values)
                                   )
    elif len(ds_array.shape) == 2: # (ngrids, level)
        ds_dilation_1d = xr.Dataset(data_vars = dict(var_temp = (['ngrids','level'], ds_array),
                                                     lat = (['ngrids'], lat_1d),
                                                     lon = (['ngrids'], lon_1d)),
                                    coords = dict(ngrids = (['ngrids'], np.arange(len(idx_lat))),
                                                  level = (['level'], var_input.level.values)),
                                    attrs = dict(description = 'extracted grids of feature-environment',
                                                 lat_res = (ds_dilation.lat[1] - ds_dilation.lat[0]).values,
                                                 lon_res = (ds_dilation.lon[1] - ds_dilation.lon[0]).values,
                                                 p_level = len(var_input.level))
                                   )      
    # rename variable
    var_name = [i for i in var_input.to_dataset().keys()][0]
    ds_dilation_1d = ds_dilation_1d.rename_vars({'var_temp': var_name})
    
    return ds_dilation_1d

def reconstruct_geocoords(ds_dilation_1d):

    # converting ds_dilation_1d back to the lat-lon coordinate
    nan_boolen = np.isnan(ds_dilation_1d.pf_mask)
    if np.sum(nan_boolen) >= 1:
        idx_end = np.where(nan_boolen > 0)[0][0]-1  
        ds_dilation_1d = ds_dilation_1d.sel(ngrids=slice(0,idx_end)) # get full valid values by removing NaN 
    
    lat_min = np.nanmin(ds_dilation_1d.lat.values)
    lat_max = np.nanmax(ds_dilation_1d.lat.values)
    lon_min = np.nanmin(ds_dilation_1d.lon.values)
    lon_max = np.nanmax(ds_dilation_1d.lon.values)

    ds_reconst = []
    for var in ds_dilation_1d.keys():
        if (var != 'lon') and (var != 'lat'):
            lon_re = np.arange(lon_min, lon_max + ds_dilation_1d.attrs['lon_res'], ds_dilation_1d.attrs['lon_res'])
            lat_re = np.arange(lat_min, lat_max + ds_dilation_1d.attrs['lat_res'], ds_dilation_1d.attrs['lat_res'])

            ds_var = ds_dilation_1d[var]
            if len(ds_var.dims) == 1: # (ngrids)

                var_map = np.full((len(lat_re), len(lon_re)),np.nan)

                for i in range(len(ds_var.ngrids)): 
                    lon = ds_dilation_1d.isel(ngrids=i).lon.values
                    lat = ds_dilation_1d.isel(ngrids=i).lat.values
                    idx_lon = np.where(lon_re == lon)[0]
                    idx_lat = np.where(lat_re == lat)[0]
                    var_map[idx_lat, idx_lon] = ds_var.isel(ngrids=i)

                ds = xr.Dataset(data_vars=dict(var_temp=(['lat','lon'],var_map)),
                                coords=dict(lon=(['lon'],lon_re),
                                            lat=(['lat'],lat_re)))
                ds = ds.rename_vars({'var_temp': str(var)})

            elif len(ds_var.dims) == 2: # (ngrids, level)

                nlevel = ds_dilation_1d.attrs['p_level']
                var_map = np.full((len(lat_re), len(lon_re), nlevel),np.nan)

                for i in range(len(ds_var.ngrids)):                 
                    lon = ds_dilation_1d.isel(ngrids=i).lon.values
                    lat = ds_dilation_1d.isel(ngrids=i).lat.values
                    idx_lon = np.where(lon_re == lon)[0]
                    idx_lat = np.where(lat_re == lat)[0]
                    var_map[idx_lat, idx_lon, :] = ds_var.isel(ngrids=i)

                ds = xr.Dataset(data_vars=dict(var_temp=(['lat','lon','level'],var_map)),
                                coords=dict(lon=(['lon'],lon_re),
                                            lat=(['lat'],lat_re),
                                            level=(['level'],ds_dilation_1d.level.values)))
                ds = ds.rename_vars({'var_temp': str(var)})        

            ds_reconst.append(ds)
    # merge all variables in the dilation dataset for a single feature
    ds_reconst_xr = xr.merge(ds_reconst)
    
    return ds_reconst_xr

def get_composite_PFID_boundary(ds_reconst_xr):
    """
    return the composite mean values of variables in the shape-file-like dataset
    input: geolocated dataset of a precipitating feature labeled with a given PFID
           , which is generated by the function "reconstruct_geocoords" applied on
           the standard output of PFID_catalogs.
    """
    # composite mean of variables as a function of grid in the radiant direction
    pf_mask = ds_reconst_xr.pf_mask
    pf_mask = pf_mask.where(pf_mask > 0, 0)

    compos_vars_outside = []
    for dilation_ngrids in np.arange(1,16):

        temp1 = binary_dilation(pf_mask, iterations=dilation_ngrids).astype('int')
        if dilation_ngrids == 1:
            temp2 = pf_mask.values
        else:
            temp2 = binary_erosion(pf_mask, iterations=dilation_ngrids-1).astype('int')
        boundary = temp1 - temp2
        boundary_xr = (pf_mask*0 + boundary).rename('boundary')
        # use boundary_xr as mask for composite of vars. outside the precip. feature
        vars_bound = ds_reconst_xr.where(boundary_xr == 1)
        compos_vars_outside.append(vars_bound.mean(('lat','lon')))

    ds_composite_out = xr.concat(compos_vars_outside, pd.Index(np.flip(np.arange(-len(compos_vars_outside),0,1))
                                                                   , name='grid_to_boundary'))

    compos_vars_inside = []
    for dilation_ngrids in np.arange(1,16):

        temp1 = binary_erosion(pf_mask, iterations=dilation_ngrids).astype('int')
        if dilation_ngrids == 1:
            temp2 = pf_mask.values
        else:
            temp2 = binary_erosion(pf_mask, iterations=dilation_ngrids-1).astype('int')
        boundary = temp2 - temp1
        boundary_xr = (pf_mask*0 + boundary).rename('boundary')
        # use boundary_xr as mask for composite of vars. outside the precip. feature
        vars_bound = ds_reconst_xr.where(boundary_xr == 1)
        if np.sum(boundary) > 0:
            compos_vars_inside.append(vars_bound.mean(('lat','lon')))

    ds_composite_in = xr.concat(compos_vars_inside, pd.Index(np.arange(len(compos_vars_inside))
                                                                   , name='grid_to_boundary'))

    # merge two sections together and sortby the relative position to the boundary of the precip. feature
    ds_composite_merged = xr.concat([ds_composite_out, ds_composite_in], dim='grid_to_boundary').sortby('grid_to_boundary')

    return ds_composite_merged
