#!/bin/bash

src_dir='/pscratch/sd/w/wmtsai/featenv_analysis/data'
rsync -avh --exclude 'dataset' --exclude 'data' $src_dir .
