import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    ######## 0. read the configure file (.yml) to get parameters #######
    data_dir = Path('/scratch/wmtsai/featenv_analysis/mcsinput_scripts')
    config_file = sys.argv[1] # configure file for parameter settings

    with open(data_dir / config_file, 'r') as f:
        config_yml = yaml.load(f, Loader=yaml.SafeLoader)

    dir_mcs_track = Path(config_yml['stats_path'])
    stat_filename = config_yml['databasename'] 
    stat_file = list(dir_mcs_track.glob(stat_filename+'*.nc'))[0]
    print('input file: {}'.format(stat_file))
    (lat_min, lon_min, lat_max, lon_max) = config_yml['geolimits']
    min_duration = config_yml['min_duration']
    mcsstat_robust = config_yml['mcsstat_robust']
    # for output 
    output_path = Path(config_yml['output_path'])
    if output_path.exists() == False:
        os.system('mkdir -p {}'.format(output_path))
    output_name = config_yml['output_name']

    # read data
    data_track = xr.open_dataset(dir_mcs_track / stat_file)
                                 
    ###################################################################

    # 1. subsamplng: geolimits
    meanlat = data_track.meanlat.sel(times=0)
    meanlon = data_track.meanlon.sel(times=0)
    idx_lon = meanlon.where((meanlon > lon_min) & (meanlon < lon_max)).dropna(dim='tracks').tracks.values
    idx_lat = meanlat.where((meanlat > lat_min) & (meanlat < lat_max)).dropna(dim='tracks').tracks.values
    idx_com = np.intersect1d(idx_lon, idx_lat)
    data_sub = data_track.sel(tracks=idx_com)

    # 2. subsampling: MCS duration                              
    time_resolution_hour = data_sub.time_resolution_hour # temporal resolution of the track
    mcs_duration = data_sub.mcs_duration*time_resolution_hour # MCS duration (hours)
    idx = np.where(mcs_duration >= min_duration)[0]
    data_non2mcs = data_sub.isel(tracks=idx) 

    # 3. identify and label MCS life stages (CCS, Init, Grow, Mature, Decay, End)
    # mature is define by the time as max. total rainfal ("total_rain" in the stat file)
    track_list = []

    for track in data_non2mcs.tracks.values:

        tmp = data_non2mcs.sel(tracks=track).mcs_status
        tmp2 = data_non2mcs.sel(tracks=track).total_rain
        idt_mcs_init = np.where(tmp == 1)[0][0]
        idt_mcs_mature = np.where(tmp2 == tmp2.max('times'))[0][0]
        idt_mcs_end = np.where(tmp == 1)[0][-1]

        mcs_duration = data_non2mcs.sel(tracks=track).mcs_duration.values*time_resolution_hour
     
        # mcsstat_robust: status (uninterrupted mcs_status == 1) throghout its all life time
        cond1 = ((idt_mcs_end - idt_mcs_init + 1)*time_resolution_hour == mcs_duration)
        cond2 = (idt_mcs_end > idt_mcs_mature) 
        cond3 = (idt_mcs_init < idt_mcs_mature)
        cond4 = (tmp.sel(times=idt_mcs_end+1) == 0)

        if (mcsstat_robust == True) and (cond1 & cond2 & cond3 & cond4):

            idt_ccs_init = 0 # start as CCS        
            idt_mcs_grow = idt_mcs_init + (idt_mcs_mature - idt_mcs_init)//2
            idt_mcs_decay = idt_mcs_mature + (idt_mcs_end - idt_mcs_mature)//2

            if (idt_mcs_mature > idt_mcs_init + 1) & (idt_mcs_end > idt_mcs_mature + 1):
 
                ds = xr.Dataset(data_vars=dict(
                       idt_ccs_init=(['tracks'], [idt_ccs_init]),
                       idt_mcs_init=(['tracks'], [idt_mcs_init]),
                       idt_mcs_grow=(['tracks'], [idt_mcs_grow]),
                       idt_mcs_mature=(['tracks'], [idt_mcs_mature]),
                       idt_mcs_decay=(['tracks'], [idt_mcs_decay]),
                       idt_mcs_end=(['tracks'], [idt_mcs_end])
                       ),
                       coords=dict(tracks=(['tracks'],[track])))

                track_list.append(ds)

        if (mcsstat_robust == False) and (cond2 & cond3 & cond4):

            idt_ccs_init = 0 # start as CCS
            idt_mcs_grow = idt_mcs_init + (idt_mcs_mature - idt_mcs_init)//2
            idt_mcs_decay = idt_mcs_mature + (idt_mcs_end - idt_mcs_mature)//2

            if (idt_mcs_mature > idt_mcs_init + 1) & (idt_mcs_end > idt_mcs_mature + 1):

                ds = xr.Dataset(data_vars=dict(
                       idt_ccs_init=(['tracks'], [idt_ccs_init]),
                       idt_mcs_init=(['tracks'], [idt_mcs_init]),
                       idt_mcs_grow=(['tracks'], [idt_mcs_grow]),
                       idt_mcs_mature=(['tracks'], [idt_mcs_mature]),
                       idt_mcs_decay=(['tracks'], [idt_mcs_decay]),
                       idt_mcs_end=(['tracks'], [idt_mcs_end])
                       ),
                       coords=dict(tracks=(['tracks'],[track])))

                track_list.append(ds)
          
    data_stablemcs_phase = xr.concat(track_list, dim='tracks') # timestamp information of stable MCSs
    # select stable MCSs from non2mcs 
    data_stablemcs_complete = data_non2mcs.sel(tracks=data_stablemcs_phase.tracks) 
    # merge two datasets into one as output   
    ds_tracks_merged = xr.merge([data_stablemcs_complete, data_stablemcs_phase])

    # 4. save merged dataset into the directory
    output_file = output_name + '.nc'
    ds_tracks_merged.to_netcdf(output_path / output_file)
    print(output_path / output_file)

