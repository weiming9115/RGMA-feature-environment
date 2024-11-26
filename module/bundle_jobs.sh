#!/bin/bash

exp_name=MCS_FLEXTRKR_tropics

for year in {2001..2023}
do 
  echo "processing year" $year
  python buoyancy_components_calc.py $exp_name $year
done
