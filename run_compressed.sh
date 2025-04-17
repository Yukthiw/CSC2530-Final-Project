#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# check to ensure we have the correct number of arguments.
if [ $# -eq 0];
then
    echo "Usage: $0 <config_path>  <log_path> <log_name>"
    exit 1
fi

# set up the variables from arguments.
CONFIG_PATH=$1
LOG_PATH=$2
LOG_NAME=$3

# config path, log path, log name
# python train.py config_compressed.yaml /w/247/yukthiw/logs/ train_compressed_logs
python train.py "$CONFIG_PATH" "$LOG_PATH" "$LOG_NAME"