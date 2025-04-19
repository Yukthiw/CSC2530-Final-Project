#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval_no_radar.py config_no_radar.yaml /w/247/yukthiw/logs/C-LBBDM-No-Radar/checkpoints/ckpt_step_50000.pth /w/247/yukthiw/eval/