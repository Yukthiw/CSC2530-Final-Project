#!/bin/bash
# SBATCH --job-name=lidar_weather     # Job name
#SBATCH --output=output_%j.log         # Output file
#SBATCH --error=error_%j.log           # Error file
#SBATCH --time=12:00:00             # Time limit hh:mm:ss
#SBATCH --partition=cpunodes         # Partition (check with `sinfo`)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (adjust if parallel)
#SBATCH --cpus-per-task=4           # CPUs per task
#SBATCH --mem=64G                   # Memory (adjust as needed)
##SBATCH --gres=gpu:1                # Request 1 GPU if needed (remove if CPU-only)

# Load your environment
# source activate snowy_lidar  # or `conda activate snowy_lidar`
# source activate snowy_lidar
source activate /w/246/willdormer/micromamba/envs/snowy_lidar

# INPUT_DIR="/w/331/yukthiw/nuscenes/samples/LIDAR_TOP"
# OUTPUT_DIR="/w/246/willdormer/projects/CompImaging/augmented_pointclouds/samples/LIDAR_TOP"

INPUT_DIR="$1"
OUTPUT_DIR="$2"
MAX_WORKERS="$3"

python3 batch_augment.py "$INPUT_DIR" "$OUTPUT_DIR" "$MAX_WORKERS"