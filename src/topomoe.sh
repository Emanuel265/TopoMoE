#!/bin/bash
#SBATCH --job-name=topomoe
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00              
#SBATCH --mem=50G                 
#SBATCH --partition=IFIgpu2070S
#SBATCH --output=logs/moe_%j.out
#SBATCH --error=logs/moe_%j.err

export PYTHONPATH=$HOME/deepspeed_custom:$PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"

python - <<EOF
import deepspeed
print("Using DeepSpeed from:", deepspeed.__file__)
EOF

export NCCL_SOCKET_IFNAME=enp3s0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=$(getent hosts $MASTER_ADDR | awk '{print $1}')
echo "MASTER_ADDR=$MASTER_ADDR"
MASTER_PORT=29500
NNODES=4
GPUS_PER_NODE=1

srun python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/train.py \
    --deepspeed \
    --deepspeed_config src/ds_config.json \
    --use-topomoe