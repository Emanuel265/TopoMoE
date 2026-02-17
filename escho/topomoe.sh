#!/bin/bash
#SBATCH --job-name=topomoe
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00              
#SBATCH --mem=50G                 
#SBATCH --partition=IFIgpuL40S
#SBATCH --output=logs/moe_%j.out
#SBATCH --error=logs/moe_%j.err

export PYTHONPATH=$HOME/deepspeed_custom:$PYTHONPATH

python - <<EOF
import deepspeed
print("Using DeepSpeed from:", deepspeed.__file__)
EOF

deepspeed --num_gpus=2 train.py