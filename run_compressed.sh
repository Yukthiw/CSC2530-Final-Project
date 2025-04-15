#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py config_compressed.yaml /w/247/yukthiw/logs/ train_compressed_logs