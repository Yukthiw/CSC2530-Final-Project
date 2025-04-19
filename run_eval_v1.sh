#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval.py config.yaml /w/247/yukthiw/logs/C-LBBDM-V1/checkpoints/ckpt_step_50000.pth /w/247/yukthiw/eval/