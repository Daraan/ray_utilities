#!/bin/bash
#SBATCH --job-name=ray_tune_mt
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
# SLURM Submission Script for Multi-Tenant Ray Clusters
#
# Use this script when multiple users share the same SLURM cluster.
# It uses manual port configuration to avoid conflicts between concurrent jobs.
#
# WARNING: ray symmetric-run does NOT work in multi-tenant environments!
#          This script uses manual ray start commands with custom ports.
#
# Usage:
#   sbatch tune_multi_tenant.sh <PYTHON_FILE> [PYTHON_ARGS...]
#
# Port Configuration:
#   Set USER_PORT_OFFSET environment variable to avoid conflicts.
#   Each user should use a different offset (e.g., 0, 10000, 20000, 30000)
#
# Examples:
#   USER_PORT_OFFSET=0 sbatch tune_multi_tenant.sh experiments/tune.py --seed 42
#   USER_PORT_OFFSET=10000 sbatch tune_multi_tenant.sh experiments/tune.py
#
# Environment Variables:
#   USER_PORT_OFFSET - Port offset for this user (default: SLURM_JOB_ID % 50000)
#   WORKSPACE_DIR    - Workspace directory (default: auto-detected)
#   RAY_OBJECT_STORE - Object store memory in GB (default: 10)
#
# Python Script Requirements:
#   Your Python script must use address='auto' to connect to the cluster:
#
#   with ray.init(address='auto', runtime_env=runtime_env):
#       results = run_tune(setup)
#
# Port Ranges per User (coordinate with your team):
#   User A: USER_PORT_OFFSET=0       (ports 6379-19999)
#   User B: USER_PORT_OFFSET=10000   (ports 16379-29999)
#   User C: USER_PORT_OFFSET=20000   (ports 26379-39999)
#   User D: USER_PORT_OFFSET=30000   (ports 36379-49999)
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

# SLURM sends SIGUSR1 10 minutes before job timeout (see --signal directive)
# This allows graceful shutdown and checkpoint saving
PYTHON_PID=""

handle_timeout() {
    echo "========================================================================"
    echo "WARNING: Received timeout signal (10 minutes remaining)"
    echo "========================================================================"
    echo "Time: $(date)"
    echo "Initiating graceful shutdown..."

    # Forward signal to Python process so it can handle cleanup
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

if [ $# -eq 0 ]; then
    echo "ERROR: No Python script specified"
    echo "Usage: sbatch tune_multi_tenant.sh <PYTHON_FILE> [PYTHON_ARGS...]"
    echo "Example: USER_PORT_OFFSET=10000 sbatch tune_multi_tenant.sh experiments/tune.py"
    exit 1
fi

# Parse Python script from arguments (handles mixed SLURM options and Python script)
parse_python_script_from_args "$@" || exit 1
# Shift away all arguments up to and including the Python script
shift "${PYTHON_SCRIPT_INDEX}"
# Now $@ contains only Python script arguments

# ============================================================================
# Port Configuration for Multi-Tenant Environment
# ============================================================================

# Calculate port offset based on job ID if not specified
if [ -z "${USER_PORT_OFFSET:-}" ]; then
    USER_PORT_OFFSET=$((((SLURM_JOB_ID % 50) * 1000) % 40000))
    echo "INFO: Auto-calculated USER_PORT_OFFSET=${USER_PORT_OFFSET} from job ID"
    echo "      To avoid conflicts, explicitly set USER_PORT_OFFSET before sbatch:"
    echo "      USER_PORT_OFFSET=10000 sbatch tune_multi_tenant.sh ..."
fi

USER_PORT_OFFSET_MINOR=$((USER_PORT_OFFSET / 40))

# Check if there is dws-## (1-17) in SLURM_JOB_NODELIST. Set the USER_PORT_OFFSET to ## * 8000 if so.
# Check if node list contains dws-## pattern and auto-set offset
if [[ "${SLURM_JOB_NODELIST}" =~ dws-([0-9]+) ]]; then
    DWS_NUMBER="${BASH_REMATCH[1]}"
    if [ "${DWS_NUMBER}" -ge 1 ] && [ "${DWS_NUMBER}" -le 17 ]; then
        USER_PORT_OFFSET=$((DWS_NUMBER * 2420))
        echo "INFO: Detected dws-${DWS_NUMBER} in node list, set USER_PORT_OFFSET=${USER_PORT_OFFSET}"
        USER_PORT_OFFSET_MINOR=$((DWS_NUMBER * 40))
    fi
fi


# Ray ports (avoid conflicts with other users)
# See: https://docs.ray.io/en/latest/ray-core/configure.html#ray-ports
RAY_PORT=$((6379 + USER_PORT_OFFSET_MINOR))
NODE_MANAGER_PORT=$((6700 + USER_PORT_OFFSET_MINOR))
OBJECT_MANAGER_PORT=$((6701 + USER_PORT_OFFSET_MINOR))
RAY_CLIENT_SERVER_PORT=$((10001 + USER_PORT_OFFSET_MINOR))
REDIS_SHARD_PORT=$((6702 + USER_PORT_OFFSET_MINOR))
MIN_WORKER_PORT=$((11000 + USER_PORT_OFFSET))
MAX_WORKER_PORT=$((13411 + USER_PORT_OFFSET))
DASHBOARD_PORT=$((8265 + USER_PORT_OFFSET_MINOR))
DASHBOARD_AGENT_GRPC_PORT=$((6703 + USER_PORT_OFFSET_MINOR))

# ============================================================================
# Configuration
# ============================================================================

# Setup environment variables - must be called after PYTHON_SCRIPT is set
setup_environment_vars "${PYTHON_SCRIPT}"

# Ray configuration
RAY_OBJECT_STORE="${RAY_OBJECT_STORE:-10}"  # GB
RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID}"

# Output directories (likely unused)
LOG_DIR="${WORKSPACE_DIR}/outputs/slurm_logs"

# ============================================================================
# Setup
# ============================================================================

echo "========================================================================"
echo "Ray Utilities - Multi-Tenant SLURM Job ${SLURM_JOB_ID}"
echo "========================================================================"
echo "Job name:       ${SLURM_JOB_NAME}"
echo "RUN_ID:         ${RUN_ID}"
echo "Entry point:    ${ENTRY_POINT} (ID: ${ENTRY_POINT_ID})"
echo "Nodes:          ${SLURM_JOB_NUM_NODES}"
echo "CPUs per node:  ${SLURM_CPUS_PER_TASK}"
echo "GPUs per task:  ${SLURM_GPUS_PER_TASK:-0}"
echo "Workspace:      ${WORKSPACE_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "========================================================================"
echo "Port Configuration (offset: ${USER_PORT_OFFSET})"
echo "  Ray port:              ${RAY_PORT}"
echo "  Node manager:          ${NODE_MANAGER_PORT}"
echo "  Object manager:        ${OBJECT_MANAGER_PORT}"
echo "  Client server:         ${RAY_CLIENT_SERVER_PORT}"
echo "  Dashboard:             ${DASHBOARD_PORT}"
echo "  Redis shard:           ${REDIS_SHARD_PORT}"
echo "  Worker ports:          ${MIN_WORKER_PORT}-${MAX_WORKER_PORT}"
echo "========================================================================"

# Check available disk space on /tmp
echo "========================================================================"
echo "Disk Space Check"
echo "========================================================================"
echo "Disk space on /tmp:      $(df -h /tmp)"
df -h
echo "========================================================================"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}"

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
    echo "Current directory: $(pwd)"
    echo "Note: Script paths are relative to where sbatch was executed"
    exit 1
fi

# ============================================================================
# Log experiment overview (for tracking)
# ============================================================================

# Prepare overview file and append job info
OVERVIEW_FILE="${OUTPUT_DIR}/experiment_overview.csv"
{
    echo -e "${SLURM_JOB_ID}\t${RUN_ID}\t${PYTHON_SCRIPT}\t$*"
} >> "${OVERVIEW_FILE}"

# Activate virtual environment
activate_virtualenv

# Verify Python and dependencies
print_python_info

# ============================================================================
# Start Ray Cluster with Custom Ports
# ============================================================================

# Get node information
HEAD_NODE=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)
WORKER_NODES=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | tail -n +2)

# Get head node IP
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')
HEAD_ADDRESS="${HEAD_IP}:${RAY_PORT}"
# Set Ray address for automatic connection
export RAY_ADDRESS="${HEAD_ADDRESS}"

echo "========================================================================"
echo "Ray Cluster Setup"
echo "========================================================================"
echo "Head node:      ${HEAD_NODE} (${HEAD_IP})"
echo "Head address:   ${HEAD_ADDRESS}"
echo "Worker nodes:   ${WORKER_NODES:-none}"
echo "========================================================================"

# Start head node with custom ports
echo "Starting Ray head node with custom ports..."
export RAY_USAGE_STATS_ENABLED=0
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-port="${DASHBOARD_PORT}" \
    --dashboard-agent-grpc-port="${DASHBOARD_AGENT_GRPC_PORT}" \
    --include-dashboard="${RAY_INCLUDE_DASHBOARD:-false}" \
    --node-manager-port="${NODE_MANAGER_PORT}" \
    --object-manager-port="${OBJECT_MANAGER_PORT}" \
    --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
    --redis-shard-ports="${REDIS_SHARD_PORT}" \
    --min-worker-port="${MIN_WORKER_PORT}" \
    --max-worker-port="${MAX_WORKER_PORT}" \
    --object-store-memory="${RAY_OBJECT_STORE}000000000" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${SLURM_GPUS_PER_TASK:-0}" \
    --temp-dir="${RAY_TMPDIR}" \
    --block &

sleep 10

# Start worker nodes
if [ -n "${WORKER_NODES}" ]; then
    echo "Starting Ray worker nodes..."
    for node in ${WORKER_NODES}; do
        echo "  - Starting worker on ${node}"
        srun --nodes=1 --ntasks=1 -w "${node}" \
            ray start \
            --address="${HEAD_ADDRESS}" \
            --node-manager-port="${NODE_MANAGER_PORT}" \
            --object-manager-port="${OBJECT_MANAGER_PORT}" \
            --min-worker-port="${MIN_WORKER_PORT}" \
            --max-worker-port="${MAX_WORKER_PORT}" \
            --object-store-memory="${RAY_OBJECT_STORE}000000000" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" \
            --num-gpus="${SLURM_GPUS_PER_TASK:-0}" \
            --temp-dir="${RAY_TMPDIR}" \
            --block &
    done
fi

sleep 5

echo "========================================================================"
echo "Checking Ray Cluster Status"
echo "========================================================================"
ray status || echo "WARNING: Could not get ray status (cluster may still be initializing)"

# ============================================================================
# Run Experiment
# ============================================================================



#setup_python_args "$@"
PYTHON_ARGS=("$@")

# No upload on slurm we can do that later.
# Replace "offline+upload" and "offline+upload@end" with "offline" in PYTHON_ARGS
for i in "${!PYTHON_ARGS[@]}"; do
    PYTHON_ARGS[$i]="${PYTHON_ARGS[$i]//offline+upload@end/offline}"
    PYTHON_ARGS[$i]="${PYTHON_ARGS[$i]//offline+upload/offline}"
done

# ============================================================================

echo "========================================================================"
echo "Running Experiment"
echo "========================================================================"
echo "Script:         ${PYTHON_SCRIPT}"
echo "Arguments:      ${PYTHON_ARGS[@]}"
echo "Ray address:    ${HEAD_ADDRESS} (via RAY_ADDRESS env var)"
echo "RUN_ID:        ${RUN_ID}"
echo "Entry point:   ${ENTRY_POINT} (ID: ${ENTRY_POINT_ID})"
echo "========================================================================"
echo ""
echo "Python scripts should use ray.init(address='auto') to connect"
echo "========================================================================"

# Run Python script in background to capture PID
python -u "${PYTHON_SCRIPT}" --log_level IMPORTANT_INFO "${PYTHON_ARGS[@]}" &
PYTHON_PID=$!

# Wait for Python process to complete
wait $PYTHON_PID
EXIT_CODE=$?

echo "========================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "========================================================================"

# ============================================================================
# Cleanup
# ============================================================================

echo "Stopping Ray cluster..."
ray stop || echo "WARNING: Ray stop failed (cluster may already be stopped)"

# Wait for all background jobs
timeout 300 wait || true

# Transfer results if needed

if [ "${USE_BACKUP_DUMP:-false}" = "true" ] && [ -n "${BACKUP_DUMP_DIR:-}" ]; then
    echo "Syncing results to backup: ${BACKUP_DUMP_DIR}..."
    rsync -avz "${RAY_TMPDIR}/" "${BACKUP_DUMP_DIR}/" || echo "WARNING: backup sync failed"
fi

echo "========================================================================"
echo "Job Complete"
echo "========================================================================"
echo "Results:        ${OUTPUT_DIR}"
echo "Logs:           ${LOG_DIR}"
echo "Job output:     $(pwd)/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "Backup dump:    ${BACKUP_DUMP_DIR:-none}"
echo "========================================================================"

exit ${EXIT_CODE}
