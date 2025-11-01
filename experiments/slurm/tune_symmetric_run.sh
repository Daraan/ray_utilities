#!/bin/bash
#SBATCH --job-name=ray_tune
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
# SLURM Submission Script for Ray Tune with ray symmetric-run (Ray 2.49+)
#
# This script uses the new `ray symmetric-run` command which simplifies
# Ray cluster management on SLURM by automatically handling head/worker setup.
#
# Usage:
#   sbatch tune_symmetric_run.sh <PYTHON_FILE> [PYTHON_ARGS...]
#
# Examples:
#   sbatch tune_symmetric_run.sh experiments/tune_with_scheduler.py --seed 123
#   sbatch tune_symmetric_run.sh experiments/default_training.py --iterations 2000
#
# Environment Variables (set before sbatch):
#   WORKSPACE_DIR    - Custom workspace directory (default: auto-detected)
#   RAY_TMPDIR       - Ray temporary directory (default: $SLURM_TMPDIR/ray)
#   RAY_PORT         - Ray head node port (default: 6379)
#
# Requirements:
#   - Ray 2.49 or above
#   - Single-tenant cluster (symmetric-run doesn't support multi-tenant yet)
#
# Advantages over manual ray start:
#   - Simplified setup: no separate head/worker node management
#   - Automatic cleanup: Ray stops when script completes
#   - Better error handling: cluster startup failures are clearer
#   - Synchronous execution: script only runs after all nodes join
#
# Python Script Compatibility:
#   NO CHANGES NEEDED! Your existing ray.init() calls work as-is:
#
#   with ray.init(num_cpus=24, runtime_env=runtime_env):
#       results = run_tune(setup)
#
#   The ray.init() automatically detects the cluster started by symmetric-run.
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

# SLURM sends SIGUSR1 10 minutes before job timeout to the batch shell (--signal=B:SIGUSR1@600)
# We must forward it to srun, which will then propagate to all tasks
SRUN_PID=""

handle_timeout() {
    echo "========================================================================"
    echo "WARNING: Received timeout signal (10 minutes remaining)"
    echo "========================================================================"
    echo "Time: $(date)"

    # Forward signal to srun process
    if [ -n "${SRUN_PID}" ] && kill -0 "${SRUN_PID}" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to srun (PID: ${SRUN_PID})"
        kill -SIGUSR1 "${SRUN_PID}"
        echo "Signal forwarded. Python script should handle SIGUSR1 for graceful shutdown."
    else
        echo "WARNING: srun process not found or already terminated"
    fi
    echo "========================================================================"
}

trap 'handle_timeout' SIGUSR1

# ============================================================================
# Parse Arguments
# ============================================================================

if [ $# -eq 0 ]; then
    echo "ERROR: No Python script specified"
    echo "Usage: sbatch tune_symmetric_run.sh <PYTHON_FILE> [PYTHON_ARGS...]"
    echo "Example: sbatch tune_symmetric_run.sh experiments/tune_with_scheduler.py --seed 123"
    exit 1
fi

# Parse Python script from arguments (handles mixed SLURM options and Python script)
parse_python_script_from_args "$@" || exit 1
# Shift away all arguments up to and including the Python script
shift "${PYTHON_SCRIPT_INDEX}"
# Now $@ contains only Python script arguments

# ============================================================================
# Configuration
# ============================================================================

# Setup environment variables - must be called after PYTHON_SCRIPT is set
setup_environment_vars "${PYTHON_SCRIPT}"

# Ray configuration
RAY_PORT="${RAY_PORT:-6379}"
RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID}"

# Output directories (likely unused)
LOG_DIR="${WORKSPACE_DIR}/outputs/slurm_logs"

# ============================================================================
# Setup
# ============================================================================

echo "========================================================================"
echo "Ray Utilities - SLURM Job ${SLURM_JOB_ID} (symmetric-run)"
echo "========================================================================"
echo "Job name:       ${SLURM_JOB_NAME}"
echo "RUN_ID:         ${RUN_ID}"
echo "Entry point:    ${ENTRY_POINT} (ID: ${ENTRY_POINT_ID})"
echo "Nodes:          ${SLURM_JOB_NUM_NODES}"
echo "CPUs per node:  ${SLURM_CPUS_PER_TASK}"
echo "GPUs per task:  ${SLURM_GPUS_PER_TASK:-0}"
echo "Workspace:      ${WORKSPACE_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Ray tmpdir:     ${RAY_TMPDIR}"
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

# ============================================================================
# Environment Setup
# ============================================================================

# Activate virtual environment using common function
activate_virtualenv
print_python_info

# Check Ray version requirement
if [ "$RAY_VERSION" = "NOT INSTALLED" ]; then
    echo "ERROR: Ray is not installed"
    exit 1
fi

# Verify Ray 2.49+ for symmetric-run
RAY_MAJOR=$(echo "$RAY_VERSION" | cut -d. -f1)
RAY_MINOR=$(echo "$RAY_VERSION" | cut -d. -f2)
if [ "$RAY_MAJOR" -lt 2 ] || ([ "$RAY_MAJOR" -eq 2 ] && [ "$RAY_MINOR" -lt 49 ]); then
    echo "ERROR: Ray version ${RAY_VERSION} is too old for symmetric-run"
    echo "       symmetric-run requires Ray 2.49 or above"
    echo "       Either upgrade Ray or use tune_with_scheduler.sh instead"
    exit 1
fi

# ============================================================================
# Get Head Node Information
# ============================================================================

# Get all node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Build head node address (hostname:port format for Ray)
ip_head="${head_node}:${RAY_PORT}"
export ip_head

echo "========================================================================"
echo "Ray Cluster Configuration"
echo "========================================================================"
echo "Head node:      ${head_node}"
echo "Head address:   ${ip_head}"
echo "Worker nodes:   ${nodes_array[@]:1}"
echo "========================================================================"

# ============================================================================
# Run Experiment with ray symmetric-run
# ============================================================================

# Process Python arguments
#setup_python_args "$@"
PYTHON_ARGS=("$@")


echo "========================================================================"
echo "Starting Experiment with ray symmetric-run"
echo "========================================================================"
echo "Script:         ${PYTHON_SCRIPT}"
echo "Arguments:      ${PYTHON_ARGS[@]}"
echo "RUN_ID:         ${RUN_ID}"
echo "========================================================================"
echo ""
echo "What happens next:"
echo "  1. Ray starts on all ${SLURM_JOB_NUM_NODES} nodes"
echo "  2. Head node: ${head_node}"
echo "  3. Workers: ${nodes_array[@]:1}"
echo "  4. Python script runs ONLY on head node"
echo "  5. Cluster stops automatically when script completes"
echo "========================================================================"

# Use ray symmetric-run to start Ray cluster and execute script
# - All nodes execute this command via srun
# - Ray automatically starts head on first node, workers on others
# - Python script ONLY executes on head node
# - Cluster automatically stops when script completes
# - Use '--' to separate Ray args from Python script args

srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" \
    ray symmetric-run \
    --address "${ip_head}" \
    --min-nodes "${SLURM_JOB_NUM_NODES}" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${SLURM_GPUS_PER_TASK:-0}" \
    --temp-dir="${RAY_TMPDIR}" \
    -- \
    python -u "${PYTHON_SCRIPT}" "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

echo "========================================================================"
echo "Experiment Exit Code: ${EXIT_CODE}"
echo "========================================================================"

# ============================================================================
# File Transfer (if needed)
# ============================================================================

# Optional: Sync results if using local storage on compute nodes
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}/outputs" ]; then
    echo "Transferring results from ${SLURM_TMPDIR}/outputs to ${OUTPUT_DIR}..."
    rsync -avz "${SLURM_TMPDIR}/outputs/" "${OUTPUT_DIR}/" || echo "WARNING: rsync failed"
fi

echo "========================================================================"
echo "Job Complete"
echo "========================================================================"
echo "Results:        ${OUTPUT_DIR}"
echo "Logs:           ${LOG_DIR}"
echo "Job output:     $(pwd)/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "========================================================================"

exit ${EXIT_CODE}
