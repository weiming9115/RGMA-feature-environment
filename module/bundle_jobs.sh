#!/bin/bash

exp_name=NonMCS_FLEXTRKR_tropics

for year in {2016..2019}
do 
  echo "processing year" $year
  python buoyancy_components_calc.py $exp_name $year
done
