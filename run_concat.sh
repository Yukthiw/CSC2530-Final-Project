#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py config_concat.yaml /w/247/yukthiw/logs/ train_concat_logs