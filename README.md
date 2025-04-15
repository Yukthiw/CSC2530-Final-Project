# CSC2530-Final-Project

## Training

To run training on SLURM cluster submit the training bash script which you would like to run, e.g.:
```
sbatch --partition=gpunodes --nodelist=gpunode{gpu_node_number} --gres gpu:1 -c {num_cores} --mem={CPU_mem_reserved}G --output={slurm_output_name}.log --time=4-00:00:00 run_compressed.sh
```

Note that you may need to modify the output directory of the training, this is where checkpoints, losses and intermediate samples will be saved so make sure sufficient disk space is available.

i.e. In the following example `/w/247/yukthiw/logs/` is set as the output directory, you may need to change this. 
```
#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py config_compressed.yaml /w/247/yukthiw/logs/ train_compressed_logs
```

### Training Parameters

Training parameters such as batch sizes, learning rate, validation interval etc. can be modified by editing the config which you choose to train with.