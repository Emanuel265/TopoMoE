#!/bin/bash
#SBATCH --job-name=deepspeed_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --partition=IFIgpuL40S
#SBATCH --output=logs/ds_%j.out
#SBATCH --error=logs/ds_%j.err

set -euo pipefail

NUM_RUNS=1

export MODE=topomoe   # "topomoe" or "deepspeed" or "pip_deepspeed"

if [[ "${MODE}" == "topomoe" ]]; then
    echo "Using TopoMoE"

    export PYTHONPATH="$HOME/deepspeed_base/TopoMoE:${PYTHONPATH:-}"
    export PYTHONNOUSERSITE=1

elif [[ "${MODE}" == "deepspeed" ]]; then
    echo "Using default Deepspeed (local)"

    export PYTHONPATH="$HOME/deepspeed_base/DeepSpeed:${PYTHONPATH:-}"
    export PYTHONNOUSERSITE=1

elif [[ "${MODE}" == "pip_deepspeed" ]]; then
    echo "Using PIP-installed DeepSpeed"

    unset PYTHONPATH
    unset PYTHONNOUSERSITE

else
    echo "ERROR: MODE must be 'topomoe' or 'deepspeed' or 'pip_deepspeed'"
    exit 1
fi

# 1. Force unbuffered output for Python
export PYTHONUNBUFFERED=1

# 2. Quiet down the pip noise
echo "--- Starting Environment Setup ---"
cd "$HOME/deepspeed_base/TopoMoE"
# pip uninstall -y deepspeed -q
# pip install -e . -q
cd ..
echo "--- Setup Complete ---"

# 3. Explicitly flush the echo
echo "Using DeepSpeed from:"
python -c "import deepspeed; print('Deepspeed origin: ' + deepspeed.__file__)"

export NCCL_DEBUG=INFO
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

results=()

if [[ "${NUM_RUNS}" -eq 1 ]]; then
    echo ""
    echo "=============================="
    echo "Single Run"
    echo "=============================="
    deepspeed --num_gpus=2 cifar10.py
    exit 0
fi

for run in $(seq 1 ${NUM_RUNS}); do
    echo ""
    echo "=============================="
    echo "Run ${run}/${NUM_RUNS}"
    echo "=============================="

    output=$(deepspeed --num_gpus=2 train.py)

    echo "${output}"

    # Extract Avg tokens/sec (expects: "Avg tokens/sec: X")
    tps=$(echo "${output}" | grep -E "Avg tokens/sec" | awk '{print $NF}')

    if [[ -z "${tps}" ]]; then
        echo "ERROR: Could not extract Avg tokens/sec"
        exit 1
    fi

    results+=("${tps}")
done

python - <<EOF
import math

# Bash expands this into a Python list of STRINGS
values_str = ["${results[@]}"]

# Split because bash expands as a single string
values_str = values_str[0].split()

# Remove commas and convert to float
values = [float(v.replace(",", "")) for v in values_str]

mean = sum(values) / len(values)
std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

print("\n==============================")
print("DeepSpeed benchmark results")
print("==============================")
for i, v in enumerate(values, 1):
    print(f"Run {i}: {v:.2f} tokens/sec")

print(f"\nMean: {mean:.2f} tokens/sec")
print(f"Std:  {std:.2f} tokens/sec")
EOF
