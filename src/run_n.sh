#!/bin/bash
# launch_runs.sh — submits 3 serial SLURM jobs, each starting after the previous succeeds

SCRIPT="escho/train.py"
DS_CONFIG="escho/ds_minimal_moe_test.json"

JOB_ID=""

for RUN in 1 2 3; do
    # Build dependency flag (empty for the first job)
    if [ -z "$JOB_ID" ]; then
        DEP_FLAG=""
    else
        DEP_FLAG="--dependency=aftercorr:${JOB_ID}"
    fi

    JOB_ID=$(sbatch $DEP_FLAG \
        --job-name="topomoe_run${RUN}" \
        --nodes=2 \
        --gres=gpu:4 \
        --ntasks-per-node=4 \
        --cpus-per-task=4 \
        --time=2:00:00 \
        --mem=50G \
        --partition=IFIgpu2070S \
        --output="logs/moe_run${RUN}_%j.out" \
        --error="logs/moe_run${RUN}_%j.err" \
        --parsable \
        --wrap="
            export PYTHONPATH=\$HOME/deepspeed_custom:\$PYTHONPATH
            echo 'Run ${RUN}/3 — job \$SLURM_JOB_ID'
            deepspeed --hostfile=escho/hostfile ${SCRIPT} \
                --deepspeed_config ${DS_CONFIG} \
                --output_dir results/run_${RUN}
        ")

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: sbatch failed for run ${RUN}. Aborting."
        exit 1
    fi

    echo "Submitted run ${RUN}/3  →  job ${JOB_ID}"
done

echo ""
echo "All 3 jobs queued. Check status with:"
echo "  squeue -u \$USER"

