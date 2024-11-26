#!bin/bash

for year in {2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020}
do 
   echo 'processing' $year
# edit the ./config/feature_list.jsonc by replacing the track input 
   sed 's/.year./.'$year'./g' ../config/config_track_env/feature_list_default.jsonc > ../config/config_track_env/feature_list.jsonc
   python run_feature_enviroment.py
   sleep 10
done
