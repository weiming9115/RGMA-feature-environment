#!bin/bash

for year in {2001,2002,2003,2004,2005,2006,2007,2008,2009}
do 
   echo 'processing' $year
# edit the ./config/feature_list.jsonc by replacing the track input 
   sed 's/.year./.'$year'./g' ../config/feature_list_default.jsonc > ../config/feature_list.jsonc
   python run_feature_enviroment.py
   sleep 10
done
