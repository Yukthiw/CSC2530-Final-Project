#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_no_radar.py config_no_radar.yaml /w/247/yukthiw/logs/ train_no_radar