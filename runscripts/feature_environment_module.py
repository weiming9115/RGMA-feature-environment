import os
#ncore = "1"
#os.environ["OMP_NUM_THREADS"] = ncore
#os.environ["OPENBLAS_NUM_THREADS"] = ncore
#os.environ["MKL_NUM_THREADS"] = ncore
#os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
#os.environ["NUMEXPR_NUM_THREADS"] = ncore
import sys
import xarray as xr
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path
import cartopy.crs as ccrs
import warnings
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

class ds_feature_environment:
    
    __version__ = "1.0beta"
    
    def __init__(self):
        
        self.name = None                          # name of the feature-environment dataset
        self.track_data = None                    # xarray dataset 
        self.object_data = None                   # xarray dataset 
        self.env_data = None                      # xarray dataset
        self.feature_data_sources = None          # e.g., ERA5, GPM-IMERG+MERGE-IR
        self.environmental_data_sources = None    # e.g., ERA5
        self.track_frequency = None               # hourly
        self.env_frequency = None                 # hourly
        self.lon_env = None                       # longitude of the env. data
        self.lat_env = None                       # latitude of the env. data
        self.lon_feature = None                   # longitude of the feature data
        self.lat_feature = None                   # latitude of the feature data
        self.feature_track = None                 # boolen. if input == track format 
        self.feature_mask = None                  # boolen. if input contains feature mask
        self.track_dir = None                     # directory of preprocessed track data saved (a copy under the created directory)
        self.obj_dir = None                       # directory of preprocessed object data saved (a copy under the created directory)
        self.env2d_dir = None                     # 2-D output directory  
        self.env3d_dir = None                     # 3-D output directory 
        self.envderive_dir = None                 # derived output directory (if needed)
        self.envcats_dir = None                   # environment catalogues directory 
        self.box_size_degree = None               # looking window size in the unit of degree (e.g., 10 x 10 deg.) 

    def create_featenv_directory(self, path_dir):
        """
        create subdirectories under the given path_dir
        """

        if Path(path_dir).exists():
            print('the given directory already exists ...')
            main_dir = Path(path_dir)
            self.track_dir = Path( str(main_dir) + '/feature_catalogs/track' )
            self.obj_dir = Path( str(main_dir) + '/feature_catalogs/object' )
            self.envcats_dir = Path( str(main_dir) + '/environment_catalogs' )
            self.env2d_dir = Path( str(main_dir) + '/environment_catalogs/VARS_2D' )
            self.env3d_dir = Path( str(main_dir) + '/environment_catalogs/VARS_3D' )
            self.envderive_dir = Path( str(main_dir) + '/environment_catalogs/VARS_derived' )
        else:
            print('generate feature-environment data directory...')

            main_dir = Path(path_dir)
            os.system('mkdir -p {}'.format(main_dir))

            featcats_dir = main_dir / 'feature_catalogs'
            envcats_dir = main_dir / 'environment_catalogs'
            track_dir = featcats_dir / 'track' 
            obj_dir = featcats_dir / 'object'
            env2d_dir = envcats_dir / 'VARS_2D'
            env3d_dir = envcats_dir / 'VARS_3D'
            envderive_dir = envcats_dir / 'VARS_derived'        

            os.system('mkdir -p {}'.format(featcats_dir))
            os.system('mkdir -p {}'.format(envcats_dir))
            print('Create main directoy: {}'.format(main_dir))
            print('{}'.format(featcats_dir))
            print('{}'.format(envcats_dir))        

            if self.feature_track:
                os.system('mkdir -p {}'.format(track_dir))
                print(track_dir)
                self.track_dir = track_dir
                if self.feature_mask:
                    os.system('mkdir -p {}'.format(track_dir/'2D_mask'))
                    print(feattrack_dir/'2D_mask')
                    self.featmask_dir = track_dir/'2D_mask'

            else:
                os.system('mkdir -p {}'.format(obj_dir))
                print(obj_dir)
                self.obj_dir = obj_dir
                if self.feature_mask:
                    os.system('mkdir -p {}'.format(obj_dir/'2D_mask'))
                    print(obj_dir/'2D_mask')
                    self.featmask_dir = obj_dir/'2D_mask'

            os.system('mkdir -p {}'.format(envcats_dir))
            print(envcats_dir)
            self.envcats_dir = envcats_dir
            
            os.system('mkdir -p {}'.format(env2d_dir))
            print(env2d_dir)
            self.env2d_dir = env2d_dir
            
            os.system('mkdir -p {}'.format(envderive_dir))
            print(envderive_dir)  
            self.envderive_dir = envderive_dir

            os.system('mkdir -p {}'.format(env3d_dir))
            print(env3d_dir)  
            self.env3d_dir = env3d_dir
    
    def load_track_data(self, file_path):
        self.track_data = xr.open_dataset(file_path)
        
        return self.track_data
    
    def load_object_data(self, file_path):
        self.object_data = xr.open_dataset(file_path)
        
        return self.track_data

    def locate_env_data(self, variable_name, path_dir):
        self.locate_env_data = {}
        if len(self.locate_env_data) == 0:
            self.locate_env_data[variable_name] = path_dir            
           
    def locate_feature_data(self, variable_name, path_dir):
        self.locate_feature_data = {}
        if len(self.locate_feature_data) == 0:
            self.locate_feature_data[variable_name] = path_dir    

    def variable_format(self, variable_name, variable_str):
        self.variable_format = {}
        if len(self.variable_format) == 0:
            self.variable_format[variable_name] = variable_str
    
    def variable_infile(self, variable_name, variable_infile):
        self.variable_infile = {}
        if len(self.variable_infile) == 0:
            self.variable_infile[variable_name] = variable_infile        
  
    def get_track_info(self, track_id):
        
        track_info = self.track_data.sel(tracks=track_id)
        
        return track_info
    
    def get_object_info(self, object_id):
        
        obj_info = self.object_data.sel(object_id=object_id)
        
        return obj_info
   
    def get_environment_vars_track(self, var_name, track_id, lat_range, lon_range, p_level=None):
        
        if len(self.locate_env_data) == 0:
            raise ValueError("No environmental data located. Please call locate_env_data() first")
        
        else:
            
            track_info = self.get_track_info(track_id=track_id)
             
            lat_cen = track_info.meanlat.values # lat centroid
            lon_cen = track_info.meanlon.values # lon centroid
         
            # find out when the tracked MCS ends as indicated by NaT
            nat_boolen = np.where(np.isnat(track_info.base_time.values))[0]
            if len(nat_boolen) == 0:
                idx_length = len(track_info.base_time) 
            else:
                idx_length = nat_boolen[0]

            data_chunk = []
            time_chunk = []
            
            for t in range(idx_length):

                time64 = track_info.base_time[t].values
                timestamp = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                time_sel = datetime.utcfromtimestamp(timestamp)
            
                # determine the env_data to be loaded            
                year = str(time_sel.year)
                month = str(time_sel.month).zfill(2)
                day = str(time_sel.day).zfill(2)
                hour = str(time_sel.hour).zfill(2)

                data_var = []
                for var in [var_name]:
                    data_dir = Path(str(self.locate_env_data[var]))
                    data_str = self.variable_format[var]
                    var_infile = self.variable_infile[var]
                    # modify the default file string with datetime info
                    tmp = data_str.replace('X',var).replace('YYYY',year).replace('MM',month)
                    tmp = tmp.replace('DD',day).replace('HH',hour)
                    filename = data_dir /'{}'.format(year)/ tmp

                    with xr.open_dataset(filename) as data_file:
                        data_file = data_file.sel(time=time_sel, method='nearest')[var_infile]
                        data_file = coordinates_processors(data_file)

                        lat_min = lat_cen[t] - lat_range/2
                        lat_max = lat_cen[t] + lat_range/2
                        lon_min = lon_cen[t] - lon_range/2
                        lon_max = lon_cen[t] + lon_range/2

                        # dealing with periodicity
                        if (lon_min >= 0) and (lon_max <= 360):
 
                            data_extract = data_file.sel(lat=slice(lat_min, lat_max),
                                                         lon=slice(lon_min, lon_max))
                            
                            # env. variable grid spacing set in run_feature_environment.py
                            dlon = self.lon_env[1] - self.lon_env[0]
                            dlat = self.lat_env[1] - self.lat_env[0]
                            data_interp = data_extract.interp(lon=np.linspace(data_extract.lon.min(), data_extract.lon.max(),round(lon_range/dlon)+1),
                                                              lat=np.linspace(data_extract.lat.min(), data_extract.lat.max(),round(lat_range/dlat)+1))

                            # converting lat-lon into x-y coordinates
                            data_interp_xy = data_interp.assign_coords(x=("lon", np.arange(len(data_interp.lon)))
                                                                      , y=("lat", np.arange(len(data_interp.lat))))
                            data_interp_xy = data_interp_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop('time')

                        elif (lon_min < 360.) and (lon_max > 360.):

                            dlon = (data_file.lon[1] - data_file.lon[0]).values
                            dlat = (data_file.lat[1] - data_file.lat[0]).values
                            dlon_env = self.lon_env[1] - self.lon_env[0]
                            dlat_env = self.lat_env[1] - self.lat_env[0]
            
                            # get original index of the centroid
                            idx_lon = np.argmin(np.abs(data_file.lon.values - lon_cen[t]))
                            idx_lat = np.argmin(np.abs(data_file.lat.values - lat_cen[t]))
                            longrids_range = round((lon_range/2)/dlon)
                            latgrids_range = round((lat_range/2)/dlat)

                            degree_excess = ( lon_max - 360 )
                            n_grids_shift = round(degree_excess/dlon) # integer
                            # shifting the data to the left
                            data_shift = data_file.roll(lon=-round(n_grids_shift + longrids_range), roll_coords=True)

                            idx_lon_new = idx_lon - (n_grids_shift + longrids_range)
                            idx_lat_new = idx_lat
                            data_extract = data_shift.isel(lon=slice(idx_lon_new-longrids_range, idx_lon_new+longrids_range),
                                                           lat=slice(idx_lat_new-latgrids_range, idx_lat_new+latgrids_range))

                            data_interp = data_extract.interp(lon=np.linspace(data_extract.lon.min(), data_extract.lon.max(),round(lon_range/dlon_env)+1),
                                                              lat=np.linspace(data_extract.lat.min(), data_extract.lat.max(),round(lat_range/dlat_env)+1))

                            # update lat/lon 
                            lon_out = self.lon_env[np.argmin(np.abs(lon_cen[t] - self.lon_env))]
                            lat_out = self.lat_env[np.argmin(np.abs(lat_cen[t] - self.lat_env))]
                            lon_array = np.arange(lon_out - lon_range/2, lon_out + lon_range/2 + dlon_env, dlon_env)
                            lon_array = np.where(lon_array <= 360, lon_array, lon_array-360)
                            lat_array = np.arange(lat_out - lat_range/2, lat_out + lat_range/2 + dlat_env, dlat_env)
                            data_interp.coords['lon'] = lon_array
                            data_interp.coords['lat'] = lat_array
                            data_interp_xy = data_interp.assign_coords(x=("lon", np.arange(len(data_interp.lon)))
                                                                     , y=("lat", np.arange(len(data_interp.lat))))
                            data_interp_xy = data_interp_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop('time')

                        elif (lon_min < 0) and (lon_max >= 0):

                            dlon = (data_file.lon[1] - data_file.lon[0]).values
                            dlat = (data_file.lat[1] - data_file.lat[0]).values
                            dlon_env = self.lon_env[1] - self.lon_env[0]
                            dlat_env = self.lat_env[1] - self.lat_env[0]

                            # get original index of the centroid
                            idx_lon = np.argmin(np.abs(data_file.lon.values - lon_cen[t]))
                            idx_lat = np.argmin(np.abs(data_file.lat.values - lat_cen[t]))
                            longrids_range = round((lon_range/2)/dlon)
                            latgrids_range = round((lat_range/2)/dlat)

                            degree_excess = ( 0. - lon_min )
                            n_grids_shift = round(degree_excess/dlon) # integer
                            # shifting the data to the right
                            data_shift = data_file.roll(lon=round(n_grids_shift + longrids_range), roll_coords=True)

                            idx_lon_new = idx_lon + (n_grids_shift + longrids_range)
                            idx_lat_new = idx_lat
                            data_extract = data_shift.isel(lon=slice(idx_lon_new-longrids_range, idx_lon_new+longrids_range),
                                                           lat=slice(idx_lat_new-latgrids_range, idx_lat_new+latgrids_range))

                            data_interp = data_extract.interp(lon=np.linspace(data_extract.lon.min(), data_extract.lon.max(),round(lon_range/dlon_env)+1),
                                                              lat=np.linspace(data_extract.lat.min(), data_extract.lat.max(),round(lat_range/dlat_env)+1))

                            # update lat/lon
                            lon_out = self.lon_env[np.argmin(np.abs(lon_cen[t] - self.lon_env))]
                            lat_out = self.lat_env[np.argmin(np.abs(lat_cen[t] - self.lat_env))]
                            lon_array = np.arange(lon_out - lon_range/2, lon_out + lon_range/2 + dlon_env, dlon_env)
                            lon_array = np.where(lon_array >= 0, lon_array, 360+lon_array) 
                            lat_array = np.arange(lat_out - lat_range/2, lat_out + lat_range/2 + dlat_env, dlat_env)
                            data_interp.coords['lon'] = lon_array
                            data_interp.coords['lat'] = lat_array
                            data_interp_xy = data_interp.assign_coords(x=("lon", np.arange(len(data_interp.lon)))
                                                                     , y=("lat", np.arange(len(data_interp.lat))))
                            data_interp_xy = data_interp_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop('time')

                        if p_level is not None: # for 3-D data ERA5 only, with vertical dim. named "level"

                            data_interp_xy = data_interp_xy.sel(level=p_level) # update data_extract which is single layer
                            
                        data_var.append(data_interp_xy)

                data_var_merged = xr.merge(data_var) # merge variables into one xr.dataset
                data_chunk.append(data_var_merged)
                time_chunk.append(time_sel)
            
            # add base_time into the dataset
            data_chunk_xr = xr.concat(data_chunk, dim=pd.Index(range(len(data_chunk)),name='time'))
            ds_basetime_xr = xr.Dataset(data_vars=dict(base_time = (['time'], time_chunk)),
                                     coords=dict(time = (['time'], range(len(data_chunk)))))
            data_track_xr = xr.merge([data_chunk_xr, ds_basetime_xr], compat='override')
            data_track_xr.coords['tracks'] = [track_id] # add track_id
           
            return data_track_xr 
        
        
