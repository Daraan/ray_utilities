#!/bin/bash
#SBATCH --job-name=ray_experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --partition=compute
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
# TODO: Output dir of log files

################################################################################
# Generic SLURM Template for Ray Utilities Experiments
#
# This is a simplified template for single-node or small experiments.
# For multi-node PBT experiments, use tune_with_scheduler.sh
#
# Usage:
#   1. Copy this file to a new name (e.g., my_experiment.sh)
#   2. Set PYTHON_SCRIPT to your experiment file
#   3. Adjust SBATCH directives as needed
#   4. Submit: sbatch my_experiment.sh [ARGS...]
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Source common base script
# SLURM copies the script to its spool directory, so we need to find the original location
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # Running under SLURM
    # Get the original script path from SLURM's batch script name
    # This contains just the script name without path, so we reconstruct from scontrol
    ORIGINAL_SCRIPT=$(scontrol show job "${SLURM_JOB_ID}" | grep -oP 'Command=\K[^ ]+' | head -1)

    # If we got the full command, extract just the script path
    # Handle both relative and absolute paths
    if [[ "${ORIGINAL_SCRIPT}" =~ ^/ ]]; then
        # Absolute path
        SCRIPT_DIR="$(dirname "${ORIGINAL_SCRIPT}")"
    else
        # Relative path - combine with submit directory
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/$(dirname "${ORIGINAL_SCRIPT}")"
    fi
else
    # Local testing (not via sbatch)
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
fi
source "${SCRIPT_DIR}/common_base.sh"

# ============================================================================
# Signal Handler for Timeout
# ============================================================================

# SLURM sends SIGUSR1 10 minutes before job timeout (see --signal=SIGUSR1@600)
# This allows graceful shutdown and checkpoint saving
PYTHON_PID=""

handle_timeout() {
    echo "========================================================================"
    echo "WARNING: Received timeout signal (10 minutes remaining)"
    echo "========================================================================"
    echo "Time: $(date)"
    echo "Initiating graceful shutdown..."

    # Forward signal to Python process so it can handle cleanup
    # kill -0 checks if process exists without sending a real signal
    if [ -n "${PYTHON_PID}" ] && kill -0 "${PYTHON_PID}" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to Python process (PID: ${PYTHON_PID})"
        kill -SIGUSR1 "${PYTHON_PID}"
    else
        echo "Warning: Python process not found or already terminated"
    fi

    # Give Python script time to checkpoint and cleanup
    # The script will exit on its own, or we'll be killed by SLURM
}

trap 'handle_timeout' SIGUSR1

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Your experiment script (relative to workspace root)
# Can be overridden by passing as first argument to this script
# Example: sbatch generic_experiment.sh experiments/my_experiment.py --seed 42
if [ $# -gt 0 ]; then
    # Parse Python script from command line arguments
    parse_python_script_from_args "$@" || exit 1
    shift "${PYTHON_SCRIPT_INDEX}"
    # Now $@ contains only Python script arguments
else
    # Use default if no arguments provided
    PYTHON_SCRIPT="experiments/default_training.py"
fi

# Optional: Override workspace directory
# WORKSPACE_DIR="/path/to/your/workspace"

# ============================================================================
# Configuration
# ============================================================================

# Setup environment variables - must be called after PYTHON_SCRIPT is set
setup_environment_vars "${PYTHON_SCRIPT}"

# Ray configuration
RAY_PORT="${RAY_PORT:-6379}"
RAY_OBJECT_STORE="${RAY_OBJECT_STORE:-5}"  # GB
RAY_TMPDIR="${TMPDIR:-/tmp}/ray_client_${SLURM_JOB_ID}"

echo "========================================================================"
echo "Ray Utilities Generic Experiment - Job ${SLURM_JOB_ID}"
echo "========================================================================"
echo "Script:         ${PYTHON_SCRIPT}"
echo "Workspace:      ${WORKSPACE_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "========================================================================"

mkdir -p "${OUTPUT_DIR}" "${RAY_TMPDIR}"

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
    echo "Current directory: $(pwd)"
    echo "Note: Script paths are relative to where sbatch was executed"
    exit 1
fi

# Activate virtual environment
activate_virtualenv

# Verify Python and dependencies
print_python_info

# ============================================================================
# Start Ray Cluster
# ============================================================================

# Get node information
HEAD_NODE=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)
WORKER_NODES=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | tail -n +2)

echo "Head node:      ${HEAD_NODE}"
echo "Worker nodes:   ${WORKER_NODES:-none}"

# Start head node
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')
echo "Starting Ray on ${HEAD_IP}:${RAY_PORT}..."

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    ray start --head \
    --port="${RAY_PORT}" \
    --object-store-memory="${RAY_OBJECT_STORE}000000000" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --temp-dir="${RAY_TMPDIR}" \
    --block &

sleep 10

# Start workers if multi-node
if [ -n "${WORKER_NODES}" ]; then
    for node in ${WORKER_NODES}; do
        echo "Starting worker on ${node}..."
        srun --nodes=1 --ntasks=1 -w "${node}" \
            ray start \
            --address="${HEAD_IP}:${RAY_PORT}" \
            --object-store-memory="${RAY_OBJECT_STORE}000000000" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" \
            --temp-dir="${RAY_TMPDIR}" \
            --block &
    done
fi

sleep 5
ray status || true

# ============================================================================
# Run Experiment
# ============================================================================

# Process Python arguments
setup_python_args "$@"

# Create wrapper script
TEMP_SCRIPT="${RAY_TMPDIR}/experiment_slurm.py"

cat > "${TEMP_SCRIPT}" << EOF
#!/usr/bin/env python3
"""
Auto-generated SLURM wrapper for: ${PYTHON_SCRIPT}

This script:
1. Modifies ray.init() to connect to existing cluster
2. Adds sys.argv passthrough for command-line args
3. Ensures proper cleanup
"""
import sys
import ray

# Modify ray.init behavior
_original_init = ray.init
def _slurm_init(*args, **kwargs):
    # Override to connect to existing cluster
    kwargs.pop('num_cpus', None)
    kwargs.pop('object_store_memory', None)
    return _original_init(address='auto', **kwargs)

ray.init = _slurm_init

# Now run the original experiment
exec(open('${PYTHON_SCRIPT}').read())
EOF

chmod +x "${TEMP_SCRIPT}"

echo "========================================================================"
echo "Running Experiment"
echo "========================================================================"
echo "Arguments:      ${PYTHON_ARGS[@]}"
echo "========================================================================"

# Run with processed args that include --storage_path in background to capture PID
python "${TEMP_SCRIPT}" "${PYTHON_ARGS[@]}" &
PYTHON_PID=$!

# Wait for Python process to complete
wait $PYTHON_PID
EXIT_CODE=$?

echo "Exit code: ${EXIT_CODE}"

# ============================================================================
# Cleanup
# ============================================================================

ray stop || true
wait

# Transfer results if needed
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}/outputs" ]; then
    echo "Transferring results..."
    rsync -avz "${SLURM_TMPDIR}/outputs/" "${OUTPUT_DIR}/" || true
fi

echo "========================================================================"
echo "Job Complete"
echo "========================================================================"
echo "Results: ${OUTPUT_DIR}"
echo "Logs:    $(pwd)/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "========================================================================"

exit ${EXIT_CODE}
