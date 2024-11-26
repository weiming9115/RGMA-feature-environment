#!/bin/bash
# Data samples for feature-environment diagnostics
# WT: ~20GB (pr, rlut, huss, ps) for an entire year !!! will make it smaller 
wget -r -nH --cut-dirs=2 -np -e robots=off https://portal.nersc.gov/cfs/m4374/data_sample/
