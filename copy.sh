#!/bin/bash

src_dir='/scratch/wmtsai/featenv_analysis/'
rsync -avh --exclude 'dataset' $src_dir .