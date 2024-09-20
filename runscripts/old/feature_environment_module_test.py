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
        self.feature_track = None
        self.feature_mask = None
        self.track_dir = None
        self.obj_dir = None
        self.env2d_dir = None
        self.env3d_dir = None
        self.envderive_dir = None
        self.envcats_dir = None

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
           
    def get_track_info(self, track_number):
        
        track_info = self.track_data.sel(tracks=track_number)
        
        return track_info
    
    def get_object_info(self, object_id):
        
        obj_info = self.object_data.sel(object_id=object_id)
        
        return obj_info
   
    def get_environment_vars_track(self, var_name, track_id, lat_range, lon_range, p_level=None):
        
        if len(self.locate_env_data) == 0:
            raise ValueError("No environmental data located. Please call locate_env_data() first")
        
        else:
            
            track_info = self.get_track_info(track_number=track_id)
             
            lat_cen = track_info.meanlat.values # lat centroid
            lon_cen = track_info.meanlon.values # lon centroid
         
            # find out when the tracked MCS ends as indicated by NaT
            nat_boolen = np.where(np.isnat(track_info.base_time.values))[0]
            if len(nat_boolen) == 0:
                idx_length = len(track_info.base_time)
            else:
                idx_length = nat_boolen[0]

            data_chunk = []
            
            for var in [var_name]:

                data_var = []

                time64 = track_info.base_time[0].values
                timestamp = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                time_start = datetime.utcfromtimestamp(timestamp)

                time64 = track_info.base_time[idx_length-1].values
                timestamp = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                time_end = datetime.utcfromtimestamp(timestamp)
            
                # determine the env_data to be loaded            
                year_st = str(time_start.year)
                month_st = str(time_start.month).zfill(2)
                day_st = str(time_start.day).zfill(2)
                hour_st = str(time_start.hour).zfill(2)

                year_ed = str(time_end.year)
                month_ed = str(time_end.month).zfill(2)
                day_ed = str(time_end.day).zfill(2)
                hour_ed = str(time_end.hour).zfill(2)

                data_dir = Path(str(self.locate_env_data[var]) + '/{}'.format(year_st))
                files = list(data_dir.glob('era-5.{}.{}.*[{}-{}].nc'.format(var,year_st,month_st,month_ed)))
               
                with xr.open_mfdataset(files) as data_files:

                    for t in range(idx_length):

                        time_chunk = []

                        time64 = track_info.base_time[t].values
                        timestamp = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                        time_sel = datetime.utcfromtimestamp(timestamp)
                        data_file = data_files.sel(time=time_sel, method='nearest')
                        data_file = coordinates_processors(data_file)

                        # find nearest grid matching the environment 
                        idx_sel = np.argmin(np.abs(data_file.lon.values - lon_cen[t]))
                        lon_cen_reset = data_file.lon[idx_sel]
                        idx_sel = np.argmin(np.abs(data_file.lat.values - lat_cen[t]))
                        lat_cen_reset = data_file.lat[idx_sel]
                
                        lat_min = lat_cen_reset - lat_range/2
                        lat_max = lat_cen_reset + lat_range/2
                        lon_min = lon_cen_reset - lon_range/2
                        lon_max = lon_cen_reset + lon_range/2

                        # dealing with periodicity
                        if (lon_min >= data_file.lon[0]) and (lon_max <= data_file.lon[-1]):
 
                            data_extract = data_file.sel(lat=slice(lat_min, lat_max),
                                                         lon=slice(lon_min, lon_max))

                        elif (lon_min < data_file.lon[0]) and (lon_max >= data_file.lon[0]):
                            dlon = (data_file.lon[1] - data_file.lon[0]).values
                            data_shift = data_file.roll(lon=2*int(-lon_min/dlon), roll_coords=True)
                            data_extract = data_shift.sel(lat=slice(lat_min, lat_max),
                                                          lon=slice(lon_min+360, lon_max))

                        elif (lon_min <= data_file.lon[-1]) and (lon_max > data_file.lon[-1]):
                            dlon = (data_file.lon[1] - data_file.lon[0]).values
                            data_shift = data_file.roll(lon=2*int((lon_max-360)/dlon), roll_coords=True)
                            data_extract = data_shift.sel(lat=slice(lat_min, lat_max),
                                                          lon=slice(lon_min, lon_max-360))

                        # converting lat-lon into x-y coordinates
                        data_extract_xy = data_extract.assign_coords(x=("lon", np.arange(len(data_extract.lon))), y=("lat", np.arange(len(data_extract.lat))))
                        data_extract_xy = data_extract_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop('time')

                        if p_level is not None: # for 3-D data ERA5 only, with vertical dim. named "level"

                            data_extract_xy = data_extract_xy.sel(level=p_level) # update data_extract which is single layer
                            
                        data_var.append(data_extract_xy)

                    data_var_merged = xr.concat(data_var, pd.Index(range(len(data_var)), dim='time')) # merge variables into one xr.dataset

                    # save into lists
                    data_chunk.append(data_var_merged)
                    time_chunk.append(time_sel)
            
            # add base_time into the dataset
            data_chunk_xr = xr.merge(data_chunk) # merge all variables
            ds_basetime_xr = xr.Dataset(data_vars=dict(base_time = (['time'], time_chunk)),
                                     coords=dict(time = (['time'], range(len(data_chunk)))))
            # merge variables + base_time into a single file
            data_track_xr = xr.merge([data_chunk_xr, ds_basetime_xr])
            
            return data_track_xr 
        
    def get_environment_vars_single(self, object_id, lat_range, lon_range, p_level=None):
        
        if len(self.locate_env_data) == 0:
            raise ValueError("No environmental data located. Please call locate_env_data() first")
        
        else:
            
            obj_info = self.get_object_info(object_id=object_id)
        
            lat_cen = obj_info.meanlat.values # MCS lat centroid
            lon_cen = obj_info.meanlon
            lon_cen = lon_cen.where(lon_cen >= 0, lon_cen+360) # converting to 0-360
            lon_cen = lon_cen.values

            data_chunk = []
                        
            time64 = obj_info.base_time.values
            timestamp = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            time_sel = datetime.utcfromtimestamp(timestamp)

            # determine the env_data to be loaded            
            year = str(time_sel.year)
            month = str(time_sel.month).zfill(2)
            day = str(time_sel.day).zfill(2)
            hour = str(time_sel.hour).zfill(2)

            data_var = []
            for var in [i for i in self.locate_env_data.keys()]:
                data_dir = Path(str(self.locate_env_data[var]))
                filename = data_dir /'{}'.format(year)/ 'era-5.{}.{}.{}.nc'.format(var,year,month)
                data_file = xr.open_dataset(filename)
                data_file = coordinates_processors(data_file)

                # find nearest ERA5 grid for the MCS centroid
                idx_sel = np.argmin(np.abs(data_file.lon.values - lon_cen))
                lon_cen_reset = data_file.lon[idx_sel]
                idx_sel = np.argmin(np.abs(data_file.lat.values - lat_cen))
                lat_cen_reset = data_file.lat[idx_sel]

                lat_min = lat_cen_reset - lat_range/2
                lat_max = lat_cen_reset + lat_range/2
                lon_min = lon_cen_reset - lon_range/2
                lon_max = lon_cen_reset + lon_range/2

                data_extract = data_file.sel(lat=slice(lat_min, lat_max),
                                                    lon=slice(lon_min, lon_max))
                data_extract = data_extract.sel(time=time_sel, method='nearest')

                # x-y grid poiints coordinate not lat-lon 
                dlon = (data_file.lon[1] - data_file.lon[0]).values
                dlat = (data_file.lat[1] - data_file.lat[0]).values
                data_extract_xy = data_extract.interp(lon=np.linspace(data_extract.lon.min(), data_extract.lon.max(),int(lon_range/dlon)+1),
                                          lat=np.linspace(data_extract.lat.min(), data_extract.lat.max(),int(lat_range/dlat)+1))
                # converting lat-lon into x-y coordinates
                data_extract_xy = data_extract_xy.assign_coords(x=("lon", np.arange(len(data_extract_xy.lon))), y=("lat", np.arange(len(data_extract_xy.lat))))
                data_extract_xy = data_extract_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop('time')

                if p_level is not None: # for 3-D data ERA5 only, with vertical dim. named "level"

                    data_extract_xy = data_extract_xy.sel(level=p_level) # update data_extract which is single layer

                data_var.append(data_extract_xy)
            data_var_merged = xr.merge(data_var) # merge variables into one xr.dataset
                                   
        return data_var_merged
        
    def get_feature_vars_track(self, track_id, lat_range, lon_range):
        
        if len(self.locate_feature_data) == 0:
            raise ValueError("No feature data located. Please call locate_feature_data() first")
        
        else:
            
            track_info = self.get_track_info(track_number=track_id)
             
            lat_cen = track_info.meanlat.values # lat centroid
            lon_cen = track_info.meanlon.values # lon centroid
           
            # find out when the tracked MCS ends as indicated by NaT
            idx_end = np.where(np.isnat(track_info.base_time.values))[0][0] 

            data_chunk = []
            time_chunk = []
            
            for t in range(idx_end):

                time64 = track_info.base_time[t].values
                timestamp = (time64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                time_sel = datetime.utcfromtimestamp(timestamp)
            
                # determine the env_data to be loaded            
                year = str(time_sel.year)
                month = str(time_sel.month).zfill(2)
                day = str(time_sel.day).zfill(2)
                hour = str(time_sel.hour).zfill(2)

                data_var = []
                for var in [i for i in self.locate_feature_data.keys()]:
                    data_dir = Path(str(self.locate_feature_data[var]))                    
                    filename = data_dir /'{}0101.0000_{}0101.0000'.format(year,int(year)+1)/ 'mcstrack_{}{}{}_{}30.nc'.format(year,month,day,hour)
                    with xr.open_dataset(filename) as data_file:

                        data_file = data_file[var].sel(time=time_sel, method='nearest')
#                    data_file = xr.open_dataset(filename)[var].sel(time=time_sel, method='nearest') # get the specified variable
                        data_file = coordinates_processors(data_file)
                        # fill gap if any in the original file (e.g., GPM-IMERG, lon = 0.05 to 395.5 where 360 is missing)
                        if (data_file.lon[0] != 0) and (data_file.lon[-1] != 360):
                            dlon = data_file.lon[1] - data_file.lon[0]
                            lon_new = np.arange(0,data_file.lon[-1]+dlon,dlon)
                            array_add = ((data_file.isel(lon=0)+data_file.isel(lon=-1))/2).values
                            array_add = np.reshape(array_add, (len(array_add),1))
                            array_new = np.hstack((array_add, data_file.values))
                            data_file = xr.Dataset(data_vars = dict(tmp_var = (['lat','lon'], array_new)),
                                                   coords = dict(lon = (['lon'], lon_new),
                                                                 lat = (['lat'], data_file.lat.values),
                                                                 time = (['time'], [data_file.time.values])))
                            data_file = data_file.rename({'tmp_var': var})

                        # regrid feature grids into the env. data
                        data_file = data_file.interp(lon=self.lon_env, lat=self.lat_env)

                        # find nearest ERA5 grid for the MCS centroid
                        idx_sel = np.argmin(np.abs(data_file.lon.values - lon_cen[t]))
                        lon_cen_reset = data_file.lon[idx_sel]
                        idx_sel = np.argmin(np.abs(data_file.lat.values - lat_cen[t]))
                        lat_cen_reset = data_file.lat[idx_sel]

                        lat_min = lat_cen_reset - lat_range/2
                        lat_max = lat_cen_reset + lat_range/2
                        lon_min = lon_cen_reset - lon_range/2
                        lon_max = lon_cen_reset + lon_range/2

                        # dealing with periodicity
                        if (lon_min >= data_file.lon[0]) and (lon_max <= data_file.lon[-1]):

                            data_extract = data_file.sel(lat=slice(lat_min, lat_max),
                                                         lon=slice(lon_min, lon_max))

                        elif (lon_min < data_file.lon[0]) and (lon_max >= data_file.lon[0]):
                            dlon = (data_file.lon[1] - data_file.lon[0]).values
                            data_shift = data_file.roll(lon=2*int(-lon_min/dlon), roll_coords=True)
                            data_extract = data_shift.sel(lat=slice(lat_min, lat_max),
                                                          lon=slice(lon_min+360, lon_max))

                        elif (lon_min <= data_file.lon[-1]) and (lon_max > data_file.lon[-1]):
                            dlon = (data_file.lon[1] - data_file.lon[0]).values
                            data_shift = data_file.roll(lon=2*int((lon_max-360)/dlon), roll_coords=True)
                            data_extract = data_shift.sel(lat=slice(lat_min, lat_max),
                                                          lon=slice(lon_min, lon_max-360))

                        # converting lat-lon into x-y coordinates
                        data_extract_xy = data_extract.assign_coords(x=("lon", np.arange(len(data_extract.lon))), y=("lat", np.arange(len(data_extract.lat))))
                        data_extract_xy = data_extract_xy.swap_dims({'lon':'x', 'lat': 'y'}).drop('time')
                        data_var.append(data_extract_xy)
                    
                data_var_merged = xr.merge(data_var) # merge variables into one xr.dataset
                data_chunk.append(data_var_merged)
                time_chunk.append(time_sel)
            
            # add base_time into the dataset
            data_chunk_xr = xr.concat(data_chunk, dim=pd.Index(range(len(data_chunk)),name='time'))
            ds_basetime_xr = xr.Dataset(data_vars=dict(base_time = (['time'], time_chunk)),
                                     coords=dict(time = (['time'], range(len(data_chunk)))))
            data_track_xr = xr.merge([data_chunk_xr, ds_basetime_xr], compat='override')
                                   
            return data_track_xr 

