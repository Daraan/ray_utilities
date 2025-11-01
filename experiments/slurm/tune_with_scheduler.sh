#!/bin/bash
#SBATCH --job-name=ray_pbt_tune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --signal=B:SIGUSR1@600
#SBATCH --output=outputs/slurm_logs/%j-%x.out
#SBATCH --error=outputs/slurm_logs/%j-%x.err
#SBATCH --mail-type=END,FAIL

################################################################################
# SLURM Submission Script for Ray Tune Experiments (Legacy/Compatible)
#
# This script uses manual ray start commands for maximum compatibility.
# For Ray 2.49+, consider using tune_symmetric_run.sh instead (simpler setup).
#
# Usage:
#   sbatch tune_with_scheduler.sh <PYTHON_FILE> [PYTHON_ARGS...]
#
# Examples:
#   sbatch tune_with_scheduler.sh experiments/tune_with_scheduler.py --seed 123
#   sbatch tune_with_scheduler.sh experiments/default_training.py --iterations 2000
#   sbatch tune_with_scheduler.sh experiments/tune_batch_size.py --env_type pendulum
#
# Environment Variables (set before sbatch):
#   WORKSPACE_DIR    - Custom workspace directory (default: auto-detected)
#   RAY_OBJECT_STORE - Object store memory in GB (default: 8)
#   RAY_PORT         - Ray head node port (default: 6379)
#
# Python Script Requirements:
#   Your Python script must use address='auto' to connect to the cluster:
#
#   with ray.init(address='auto', runtime_env=runtime_env):
#       results = run_tune(setup)
#
# Advantages over symmetric-run:
#   - Works with any Ray version (including < 2.49)
#   - Proven, stable approach
#   - Fine-grained control over head/worker setup
#
# Disadvantages:
#   - More complex setup
#   - Requires Python script modification (address='auto')
#   - Manual cleanup needed
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
# Configuration (before parsing arguments)
# ============================================================================

# Ray configuration
RAY_PORT="${RAY_PORT:-6379}"
RAY_OBJECT_STORE="${RAY_OBJECT_STORE:-8}"  # GB

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
# Parse Arguments
# ============================================================================
# Parse Arguments
# ============================================================================

if [ $# -eq 0 ]; then
    echo "ERROR: No Python script specified"
    echo "Usage: sbatch tune_with_scheduler.sh <PYTHON_FILE> [PYTHON_ARGS...]"
    echo "Example: sbatch tune_with_scheduler.sh experiments/tune_with_scheduler.py --seed 123"
    exit 1
fi

# Parse Python script from arguments (handles mixed SLURM options and Python script)
parse_python_script_from_args "$@" || exit 1
# Shift away all arguments up to and including the Python script
shift "${PYTHON_SCRIPT_INDEX}"
# Now $@ contains only Python script arguments

# ============================================================================
# Setup Environment (after parsing arguments)
# ============================================================================

# Setup environment variables - must be called after PYTHON_SCRIPT is set
setup_environment_vars "${PYTHON_SCRIPT}"

# Set derived variables
LOG_DIR="${WORKSPACE_DIR}/outputs/slurm_logs"

# ============================================================================
# Setup
# ============================================================================

echo "========================================================================"
echo "Ray Utilities - SLURM Job ${SLURM_JOB_ID}"
echo "========================================================================"
echo "Job name:       ${SLURM_JOB_NAME}"
echo "RUN_ID:         ${RUN_ID}"
echo "Entry point:    ${ENTRY_POINT} (ID: ${ENTRY_POINT_ID})"
echo "Nodes:          ${SLURM_JOB_NUM_NODES}"
echo "CPUs per node:  ${SLURM_CPUS_PER_TASK}"
echo "Workspace:      ${WORKSPACE_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Log dir:        ${LOG_DIR}"
echo "========================================================================"

# Create necessary directories
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}"

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
    echo "Current directory: $(pwd)"
    echo "Note: Script paths are relative to where sbatch was executed"
    exit 1
fi

# Load required modules (adjust for your cluster)

# Activate virtual environment
activate_virtualenv

# Verify Python and dependencies
print_python_info

# ============================================================================
# Ray Cluster Setup
# ============================================================================

# Get node information
HEAD_NODE=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)
WORKER_NODES=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | tail -n +2)

echo "========================================================================"
echo "Starting Ray Cluster"
echo "========================================================================"
echo "Head node:      ${HEAD_NODE}"
echo "Worker nodes:   ${WORKER_NODES:-none}"
echo "========================================================================"

# Get head node IP address
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')
HEAD_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "Head node IP:   ${HEAD_IP}"
echo "Head address:   ${HEAD_ADDRESS}"

# Start Ray head node
echo "Starting Ray head node on ${HEAD_NODE}..."
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    ray start --head \
    --port="${RAY_PORT}" \
    --dashboard-port=8265 \
    --object-store-memory="${RAY_OBJECT_STORE}000000000" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --temp-dir="${RAY_TMPDIR}" \
    --block &

# Wait for head node to be ready
sleep 10

# Start Ray worker nodes
if [ -n "${WORKER_NODES}" ]; then
    echo "Starting Ray worker nodes..."
    for node in ${WORKER_NODES}; do
        echo "  - Starting worker on ${node}"
        srun --nodes=1 --ntasks=1 -w "${node}" \
            ray start \
            --address="${HEAD_ADDRESS}" \
            --object-store-memory="${RAY_OBJECT_STORE}000000000" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" \
            --temp-dir="${RAY_TMPDIR}" \
            --block &
    done
fi

# Wait for cluster to be ready
sleep 5

# Verify Ray cluster
echo "========================================================================"
echo "Ray Cluster Status"
echo "========================================================================"
ray status || echo "WARNING: Could not get ray status (cluster may still be initializing)"

# ============================================================================
# Run Experiment
# ============================================================================

# Setup environment variables and process Python arguments
#setup_python_args "$@"
PYTHON_ARGS=("$@")

echo "========================================================================"
echo "Starting Experiment"
echo "========================================================================"
echo "Script:         ${PYTHON_SCRIPT}"
echo "Arguments:      ${PYTHON_ARGS[@]}"
echo "RUN_ID:         ${RUN_ID}"
echo "========================================================================"


# Run the Python script with remaining arguments in background to capture PID
python "${PYTHON_SCRIPT}" "${PYTHON_ARGS[@]}" &
PYTHON_PID=$!

# Wait for Python process to complete
wait $PYTHON_PID
EXIT_CODE=$?

echo "========================================================================"
echo "Experiment Exit Code: ${EXIT_CODE}"
echo "========================================================================"

# ============================================================================
# Cleanup and File Transfer
# ============================================================================

echo "Cleaning up Ray cluster..."
ray stop || echo "WARNING: Ray stop failed or cluster already stopped"

# Wait for all background jobs
wait

# Optional: Sync results if using local storage on compute nodes
# Uncomment if your SLURM nodes use local storage that needs to be transferred
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}/outputs" ]; then
    echo "Transferring results from ${SLURM_TMPDIR}/outputs to ${OUTPUT_DIR}..."
    rsync -avz "${SLURM_TMPDIR}/outputs/" "${OUTPUT_DIR}/" || echo "WARNING: rsync failed"
fi

echo "========================================================================"
echo "Job Complete"
echo "========================================================================"
echo "Results location: ${OUTPUT_DIR}"
echo "Logs location:    ${LOG_DIR}"
echo "Job output:       $(pwd)/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "========================================================================"

sleep 2

exit ${EXIT_CODE}
