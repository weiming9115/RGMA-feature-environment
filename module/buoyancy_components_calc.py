import os
import sys
import time
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from numba import jit
# calculations for thermodynamics
from metpy.calc import thermo
from metpy.units import units
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=False)
def es_calc_bolton(temp):
    # in hPa

    tmelt  = 273.15
    tempc = temp - tmelt
    es = 6.112*np.exp(17.67*tempc/(243.5+tempc))

    return es

@jit(nopython=False)
def es_calc(temp):
    """
    temp [x,p]
    """
    tmelt  = 273.15

    c0=0.6105851e+03
    c1=0.4440316e+02
    c2=0.1430341e+01
    c3=0.2641412e-01
    c4=0.2995057e-03
    c5=0.2031998e-05
    c6=0.6936113e-08
    c7=0.2564861e-11
    c8=-.3704404e-13

    tempc = temp - tmelt
    tempcorig = tempc

    #if tempc < -80:
    es_ltn80c = es_calc_bolton(temp)
    es_ltn80c = np.where(tempc < -80, es_ltn80c, 0)
    
    #else:
    es = c0+tempc*(c1+tempc*(c2+tempc*(c3+tempc*(c4+tempc*(c5+tempc*(c6+tempc*(c7+tempc*c8)))))))
    es_gtn80c = es/100
    es_gtn80c = np.where(tempc >= -80, es_gtn80c, 0)

    # complete es
    es = es_ltn80c + es_gtn80c

    return es

@jit(nopython=False)
def qs_calc(temp, p_level):

    tmelt  = 273.15
    RV=461.5
    RD=287.04

    EPS=RD/RV

    press = p_level * 100. # in Pa
    tempc = temp - tmelt

    es=es_calc(temp) # [x,y,p]
    es=es * 100. #hPa
    
    qs = (EPS * es) / (press + ((EPS-1.)*es))

    return qs

@jit(nopython=False)
def theta_e_calc(temp, q, p_level):

    # following the definitions in Bolton (1980): the calculation of equivalent potential temperature

    pref = 100000.
    tmelt  = 273.15
    CPD=1005.7
    CPV=1870.0
    CPVMCL=2320.0
    RV=461.5
    RD=287.04
    EPS=RD/RV
    ALV0=2.501E6

    press = p_level * 100. # in Pa
    tempc = temp - tmelt # in C

    r = q / (1. - q)

    # get ev in hPa
    ev_hPa = p_level * r / (EPS + r) # hpa

    #get TL
    TL = (2840. / ((3.5*np.log(temp)) - (np.log(ev_hPa)) - 4.805)) + 55.

    #calc chi_e:
    chi_e = 0.2854 * (1. - (0.28 * r))

    theta_e = temp * np.power((pref / press),chi_e) * np.exp(((3.376/TL) - 0.00254) * r * 1000 * (1. + (0.81 * r)))

    return theta_e

@jit(nopython=False)
def layer_average_trapz(var_1d, p_1d):

    var_sum = 0
    for z in range(1,len(var_1d)):
        dx = p_1d[z] - p_1d[z-1]
        var_sum += 1/2*(var_1d[z-1]+var_1d[z])*dx

    return var_sum/(p_1d[-1]-p_1d[0])

@jit(nopython=False)
def BL_measures_calc(T, q, sp, T2m, q2m, p_level):

    # constants
    Lv = 2.5e6 # (J/kg)
    g = 9.81 # (kg/m^2)
    cpd = 1004 # (J/kg/K)
    p0 = 1000  # (hPa)
    Rd = 287.15 # (J/kg)
    
    # find pbl top (100 hPa above the surface)
    # pbl_top_level = find_pbl_top_level(sp, p_level, pbl_depth=100)
    
    # allocate 2-D layer-averaged thetae components
    len_y = T.shape[1]
    len_x = T.shape[2]
    
    thetae_bl_array = np.zeros((len_y, len_x))*np.nan
    thetae_lt_array = np.copy(thetae_bl_array)
    thetae_sat_lt_array = np.copy(thetae_bl_array)

    # loop for lat-lon grids
    for j in np.arange(len_y):
        for i in np.arange(len_x):

            sfc_p = sp[j,i] # surface pressure
            pbl_p = sfc_p - 100

            if pbl_p >= 500: # low-troposphere upper bound greater than 500 hPa
    
                T_at_sf = T2m[j,i]
                q_at_sf = q2m[j,i]
    
                idp_sfc = np.argmin(np.abs(sfc_p - p_level)) 
                if (idp_sfc == len(p_level)-1) and (sfc_p >= 1000): # if surface pressure >= 1000 hPa
                    T_above_sf = T[:,j,i] 
                    q_above_sf = q[:,j,i]
                    p_above_sf = p_level[:]
    
                    # reconstruct the entirle T, q profiles by adding surface quantities
                    T_1d = np.hstack((np.array([T_at_sf]), np.flip(T_above_sf)))
                    q_1d = np.hstack((np.array([q_at_sf]), np.flip(q_above_sf)))
                    pressure_1d = np.hstack((np.array([sfc_p]), np.flip(p_above_sf)))
                                    
                elif (idp_sfc == len(p_level)-1) and (sfc_p < 1000): # surface pressure < 1000 hPa
                    T_above_sf = T[:idp_sfc,j,i] #[top->1000]
                    q_above_sf = q[:idp_sfc,j,i]
                    p_above_sf = p_level[:idp_sfc]
    
                    # reconstruct the entirle T, q profiles by adding surface quantities
                    T_1d = np.hstack((np.array([T_at_sf]), np.flip(T_above_sf)))
                    q_1d = np.hstack((np.array([q_at_sf]), np.flip(q_above_sf)))
                    pressure_1d = np.hstack((np.array([sfc_p]), np.flip(p_above_sf)))          
                        
                else:
                    T_above_sf = T[:idp_sfc+1,j,i] 
                    q_above_sf = q[:idp_sfc+1,j,i]
                    p_above_sf = p_level[:idp_sfc+1]
    
                    # reconstruct the entirle T, q profiles by adding surface quantities
                    T_1d = np.hstack((np.array([T_at_sf]), np.flip(T_above_sf)))
                    q_1d = np.hstack((np.array([q_at_sf]), np.flip(q_above_sf)))
                    pressure_1d = np.hstack((np.array([sfc_p]), np.flip(p_above_sf)))
                    if np.any(np.diff(pressure_1d) > 0):
                        T_1d = np.hstack((np.array([T_at_sf]), np.flip(T_above_sf[:len(p_level)-1])))
                        q_1d = np.hstack((np.array([q_at_sf]), np.flip(q_above_sf[:len(p_level)-1])))
                        pressure_1d = np.hstack((np.array([sfc_p]), np.flip(p_above_sf[:len(p_level)-1])))
                            
                # interpolated points at the pbl top and 500 hPa into P_sfc_to_100   
                pressure_val = np.hstack((pressure_1d, np.array([pbl_p]), np.array([500])))
                pressure_val = np.unique(np.sort(pressure_val))[::-1] # new pressure coord including these two levels
                T_1d_interp = np.interp(pressure_val[::-1], pressure_1d[::-1], T_1d[::-1])[::-1]
                q_1d_interp = np.interp(pressure_val[::-1], pressure_1d[::-1], q_1d[::-1])[::-1]
                                        
                # splitting into boundary layer and lower free troposphere with decreasing the p_coord
                # 1. boundary layer, bl
                idp_pbl = np.where(pressure_val == pbl_p)[0][0]
                q_bl = q_1d_interp[:idp_pbl+1]
                T_bl = T_1d_interp[:idp_pbl+1]
                p_bl = pressure_val[:idp_pbl+1]
                                
                # 2. lower free troposphere, lt
                idp_500 = np.where(pressure_val == 500.)[0][0]
                q_lt = q_1d_interp[idp_pbl:idp_500+1]
                T_lt = T_1d_interp[idp_pbl:idp_500+1]
                p_lt = pressure_val[idp_pbl:idp_500+1]
                           
                # calculating layer-averaged thetae components
                thetae_bl = theta_e_calc(T_bl, q_bl, p_bl)                
                thetae_lt = theta_e_calc(T_lt, q_lt, p_lt)
                qsat_lt = qs_calc(T_lt, p_lt)
                thetae_sat_lt = theta_e_calc(T_lt, qsat_lt, p_lt)

                if (len(thetae_bl) > 1) & (len(thetae_lt) > 1) : # if mutiple p levels to be averaged in layer_average_trapz

                    thetae_bl_avg = layer_average_trapz(np.flip(thetae_bl), np.flip(p_bl)) # negative sign b.c. decreasing p
                    thetae_lt_avg = layer_average_trapz(np.flip(thetae_lt), np.flip(p_lt))
                    thetae_sat_lt_avg = layer_average_trapz(np.flip(thetae_sat_lt), np.flip(p_lt))

                    thetae_bl_array[j,i] = thetae_bl_avg
                    thetae_lt_array[j,i] = thetae_lt_avg
                    thetae_sat_lt_array[j,i] = thetae_sat_lt_avg      
                
                else: # if only one level to be averaged... 
 
                    thetae_bl_array[j,i] = np.nan
                    thetae_lt_array[j,i] = np.nan
                    thetae_sat_lt_array[j,i] = np.nan                        

            else: # some montain areas with PBL lower than 500 hPa
    
                thetae_bl_array[j,i] = np.nan
                thetae_lt_array[j,i] = np.nan
                thetae_sat_lt_array[j,i] = np.nan
    
    # calculate buoyancy estimates
    # 2-d weighting parameters for pbl and lt
    delta_pl=sp-100-500
    delta_pb=100
    wb=(delta_pb/delta_pl)*np.log((delta_pl+delta_pb)/delta_pb)
    wl=1-wb

    # calculate buoyancy estimate
    Buoy_CAPE = (9.81/(340*3)) * wb * ((thetae_bl_array-thetae_sat_lt_array)/thetae_sat_lt_array) * 340
    Buoy_SUBSAT = (9.81/(340*3))* wl * ((thetae_sat_lt_array-thetae_lt_array)/thetae_sat_lt_array) * 340
    Buoy_TOT = Buoy_CAPE - Buoy_SUBSAT

    return (Buoy_CAPE, Buoy_SUBSAT, Buoy_TOT, thetae_bl_array, thetae_lt_array, thetae_sat_lt_array)

if __name__ == '__main__':

    time_start = time.time()

    catalog_name = sys.argv[1]
    year = sys.argv[2]
    featenv_dir = Path('/pscratch/sd/w/wmtsai/featenv_analysis/dataset/{}/{}'.format(catalog_name, year))
    print('feature_environment_dir: ', featenv_dir)
    var3d_dir = featenv_dir / 'environment_catalogs/VARS_3D'
    var2d_dir = featenv_dir / 'environment_catalogs/VARS_2D'

    # load standard outputs of the feature-environment catalogue
    data_T = xr.open_dataset(var3d_dir / '{}_T.merged.nc'.format(catalog_name))
    data_q = xr.open_dataset(var3d_dir / '{}_q.merged.nc'.format(catalog_name))
    data_d2m = xr.open_dataset(var2d_dir / '{}_2d.merged.nc'.format(catalog_name))
    data_t2m = xr.open_dataset(var2d_dir / '{}_2t.merged.nc'.format(catalog_name))
    data_sp = xr.open_dataset(var2d_dir / '{}_sp.merged.nc'.format(catalog_name))

    # loop for tracks
    BL_merged = []
    for track in data_T.tracks.values:
  
#        print('track processing: {}'.format(track))
        BL_phase = []
        # loop for time (phase)
        for t in data_T.time.values:
        
            T = data_T.sel(tracks=track, time=t, level=slice(100,1000)).t
            q = data_q.sel(tracks=track, time=t, level=slice(100,1000)).q
            sp = data_sp.sel(tracks=track, time=t).SP/100 # hPa
            T2m = data_t2m.sel(tracks=track, time=t).VAR_2T
            d2m = data_d2m.sel(tracks=track, time=t).VAR_2D
            # convert dew point to specific humidity (if applicable)
            q2m = thermo.specific_humidity_from_dewpoint(sp*100 * units.pascal, d2m * units.kelvin)
        
            p_level = data_T.level.values
            T = T.values
            q = q.values
            sp = sp.values
            T2m = T2m.values
            q2m = q2m.values

            (Buoy_CAPE, Buoy_SUBSAT, Buoy_TOT, thetae_bl_array
             , thetae_lt_array, thetae_sat_lt_array) = BL_measures_calc(T, q, sp, T2m, q2m, p_level) 
    
            # write out as xarray
            ds = xr.Dataset(data_vars=dict(
                            Buoy_CAPE = (['y','x'], Buoy_CAPE),
                            Buoy_SUBSAT = (['y','x'], Buoy_SUBSAT), 
                            Buoy_TOT = (['y','x'], Buoy_TOT),
                            thetae_bl = (['y','x'], thetae_bl_array),
                            thetae_lt = (['y','x'], thetae_lt_array),
                            thetae_sat_lt = (['y','x'], thetae_sat_lt_array)),
                       
                            coords=dict(x = (['x'], data_T.x.values),
                                        y = (['y'], data_T.y.values))
                           )
        
            BL_phase.append(ds)    
        ds_track = xr.concat(BL_phase, dim=pd.Index(data_T.time.values, name='time'))
        BL_merged.append(ds_track)
    
    # final product and save to VARS_derived
    out_dir = featenv_dir / 'environment_catalogs/VARS_derived'
    BL_merged_xr = xr.concat(BL_merged, dim=pd.Index(data_T.tracks.values, name='tracks'))
    BL_merged_xr.to_netcdf(out_dir / '{}_buoyancy.merged.nc'.format(catalog_name))

    time_end = time.time()
    time_execution = time_end - time_start
    print(str(out_dir / '{}_buoyancy.merged.nc'.format(catalog_name)) + '....saved')
    print('time_execution: ', time_execution)
