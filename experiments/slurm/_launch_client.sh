#!/bin/bash
#SBATCH --job-name=ray-client
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@600
#SBATCH --output=outputs/slurm_logs/%j-ray-client-%x.out
#SBATCH --error=outputs/slurm_logs/%j-ray-client-%x.err
#SBATCH --mail-type=END,FAIL


################################################################################
# SLURM Client Script for Ray Cluster
#
# Connects to an existing Ray head node and optionally runs a Python experiment.
# This script does NOT start its own Ray cluster - it connects to a persistent
# head node launched via launch_head.sh.
#
# Usage (Worker Only - Direct Address):
#   RAY_ADDRESS=10.5.88.207:7304 sbatch launch_client.sh
#
# Usage (Worker Only - Via Head Job ID):
#   RAY_HEAD_JOB_ID=12345 sbatch launch_client.sh
#
# Usage (With Python Script):
#   RAY_HEAD_JOB_ID=12345 sbatch launch_client.sh <PYTHON_FILE> [PYTHON_ARGS...]
#
# Usage (Via connect_to_head.sh - Recommended):
#   ./connect_to_head.sh 12345 experiments/tune.py --seed 42
#
# Required Environment Variables (one of):
#   RAY_HEAD_JOB_ID - SLURM job ID of the Ray head node
#   RAY_ADDRESS     - Direct Ray cluster address (e.g., 10.5.88.207:7304)
#
# Optional Environment Variables:
#   WORKSPACE_DIR   - Workspace directory (default: auto-detected)
#
# Connection Process:
#   1. If RAY_ADDRESS provided: Use directly
#   2. Otherwise: Read connection info from outputs/ray_head_${RAY_HEAD_JOB_ID}.info
#   3. Validates head node is still running (if RAY_HEAD_JOB_ID provided)
#   4. Connects to head node as a worker
#   5. If Python script provided: Runs it with ray.init(address='auto')
#   6. If no Python script: Stays connected as worker until job ends
#
# Python Script Requirements:
#   Your Python script must use address='auto' to connect to the cluster:
#
#   with ray.init(address='auto', runtime_env=runtime_env):
#       results = run_tune(setup)
################################################################################

set -e
set -u

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

# Check if common_base.sh exists
if [ -f "${SCRIPT_DIR}/common_base.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/common_base.sh"
elif [ -f "${SCRIPT_DIR}/experiments/slurm/common_base.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/experiments/slurm/common_base.sh"
# Do another search locally
elif [ -f "${SLURM_SUBMIT_DIR}/experiments/slurm/common_base.sh" ]; then
    # shellcheck disable=SC1091
    source "${SLURM_SUBMIT_DIR}/experiments/slurm/common_base.sh"
elif [ -f "./experiments/slurm/common_base.sh" ]; then
    # shellcheck disable=SC1091
    source "./experiments/slurm/common_base.sh"
else
    echo "ERROR: common_base.sh not found"
    echo "Searched in: ${SCRIPT_DIR}, ${SCRIPT_DIR}/experiments/slurm/, ${SLURM_SUBMIT_DIR}/experiments/slurm/ and ./experiments/slurm/"
    exit 1
fi

# ============================================================================
# Signal Handler for Timeout
# ============================================================================

# SLURM sends SIGUSR1 10 minutes before job timeout (see --signal directive)
# This allows graceful shutdown and checkpoint saving
# From the head node script we also send this signal
PYTHON_PID=""

unset PRETIMEOUT_RECEIVED

# shellcheck disable=SC2329
handle_pretimeout() {
    echo "========================================================================"
    echo "WARNING: Received pretimeout signal (SIGUSR2)"
    echo "========================================================================"
    echo "Time: $(date)"
    echo "Pretimeout received - will skip process killing in timeout handler"
    PRETIMEOUT_RECEIVED=1
    export PRETIMEOUT_RECEIVED
}

trap 'handle_pretimeout' SIGUSR2

# shellcheck disable=SC2329
handle_timeout() {
    echo "========================================================================"
    echo "WARNING: Received timeout signal (10 minutes remaining)"
    echo "========================================================================"
    echo "Time: $(date)"
    echo "Initiating graceful shutdown..."

    # Forward signal to Python process so it can handle cleanup

    # NOTE: Parent script sends SIGUSR1 --full we might double kill tune which we do not want.
    if [ -n "${PRETIMEOUT_RECEIVED:-}" ]; then
        echo "Skipping SIGUSR1 forwarding - pretimeout already received"
    elif [ -n "${PYTHON_PID}" ] && kill -0 "${PYTHON_PID}" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to Python process (PID: ${PYTHON_PID})"
        kill -SIGUSR1 "${PYTHON_PID}"
    else
        echo "Warning: Python process not found or already terminated"
    fi
}

trap 'handle_timeout' SIGUSR1

# ============================================================================
# Validate Required Environment Variables
# ============================================================================

if [ -z "${RAY_HEAD_JOB_ID:-}" ] && [ -z "${RAY_ADDRESS:-}" ]; then
    echo "ERROR: Neither RAY_HEAD_JOB_ID nor RAY_ADDRESS environment variable is set"
    echo ""
    echo "This script connects to an existing Ray head node."
    echo "You must specify either the SLURM job ID or the direct address."
    echo ""
    echo "Usage (direct address):"
    echo "  RAY_ADDRESS=10.5.88.207:7304 sbatch launch_client.sh"
    echo ""
    echo "Usage (via head job ID, worker only):"
    echo "  RAY_HEAD_JOB_ID=12345 sbatch launch_client.sh"
    echo ""
    echo "Usage (via head job ID, with Python script):"
    echo "  RAY_HEAD_JOB_ID=12345 sbatch launch_client.sh <PYTHON_FILE> [ARGS...]"
    echo ""
    echo "Or use the helper script (recommended):"
    echo "  ./connect_to_head.sh 12345 experiments/tune.py --seed 42"
    exit 1
fi

# ============================================================================
# Parse Arguments
# ============================================================================

RUN_PYTHON_SCRIPT=false
PYTHON_SCRIPT=""

if [ $# -eq 0 ]; then
    echo "INFO: No Python script specified - will connect as worker only"
    RUN_PYTHON_SCRIPT=false
else
    parse_python_script_from_args "$@" || exit 1
    shift "${PYTHON_SCRIPT_INDEX}"
    RUN_PYTHON_SCRIPT=true
fi

# Parse Ray start options from environment
RAY_START_OPTIONS=()
if [ -n "${RAY_START_OPTS:-}" ]; then
    # Convert space-separated string to array
    IFS=' ' read -r -a RAY_START_OPTIONS <<< "${RAY_START_OPTS}"
fi

# ============================================================================
# Load Ray Head Connection Info
# ============================================================================

# Detect workspace directory
detect_workspace_dir

OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs}"

# Determine connection mode
if [ -n "${RAY_ADDRESS:-}" ]; then
    CONNECTION_MODE="direct"
    echo "========================================================================"
    echo "Ray Client - Connecting to Head Node (Direct Address)"
    echo "========================================================================"
    echo "Client Job ID:    ${SLURM_JOB_ID}"
    echo "Ray Address:      ${RAY_ADDRESS}"
    echo "Mode:             $([ "${RUN_PYTHON_SCRIPT}" = "true" ] && echo "Worker + Python Script" || echo "Worker Only")"
    echo "========================================================================"

    HEAD_IP=$(echo "${RAY_ADDRESS}" | cut -d':' -f1)
    RAY_PORT=$(echo "${RAY_ADDRESS}" | cut -d':' -f2)
    HEAD_NODE="unknown"
else
    CONNECTION_MODE="job_id"
    CONNECTION_FILE="${OUTPUT_DIR}/ray_head_${RAY_HEAD_JOB_ID}.info"

    echo "========================================================================"
    echo "Ray Client - Connecting to Head Node (Via Job ID)"
    echo "========================================================================"
    echo "Client Job ID:    ${SLURM_JOB_ID}"
    echo "Head Job ID:      ${RAY_HEAD_JOB_ID}"
    echo "Connection file:  ${CONNECTION_FILE}"
    echo "Mode:             $([ "${RUN_PYTHON_SCRIPT}" = "true" ] && echo "Worker + Python Script" || echo "Worker Only")"
    echo "========================================================================"

    echo "Waiting for connection file..."
    MAX_WAIT=120
    WAIT_COUNT=0
    while [ ! -f "${CONNECTION_FILE}" ] && [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
        echo "Waiting for head node to create connection file... (${WAIT_COUNT}/${MAX_WAIT}s)"
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ ! -f "${CONNECTION_FILE}" ]; then
        echo "ERROR: Connection file not found after ${MAX_WAIT}s: ${CONNECTION_FILE}"
        echo ""
        echo "The Ray head node (job ${RAY_HEAD_JOB_ID}) may not have started properly,"
        echo "or the connection info was not saved."
        echo ""
        echo "Check if the head node is running:"
        echo "  squeue -j ${RAY_HEAD_JOB_ID}"
        echo ""
        echo "Check head node logs:"
        echo "  tail -f ${OUTPUT_DIR}/slurm_logs/${RAY_HEAD_JOB_ID}-ray-head-*.out"
        exit 1
    fi

    echo "Loading connection info..."
    # shellcheck source=outputs/ray_head_.info
    source "${CONNECTION_FILE}" || {
        echo "ERROR: Failed to source connection file"
        exit 1
    }

    if [ -z "${RAY_ADDRESS:-}" ] || [ -z "${HEAD_NODE:-}" ] || [ -z "${RAY_PORT:-}" ]; then
        echo "ERROR: Connection file is missing required variables"
        echo "Expected: RAY_ADDRESS, HEAD_NODE, RAY_PORT"
        cat "${CONNECTION_FILE}"
        exit 1
    fi

    echo "Connection info loaded successfully:"
    echo "  RAY_ADDRESS:    ${RAY_ADDRESS}"
    echo "  HEAD_NODE:      ${HEAD_NODE}"
    echo "  HEAD_IP:        ${HEAD_IP:-unknown}"
    echo "  RAY_PORT:       ${RAY_PORT}"
    echo "  Started:        ${STARTED:-unknown}"
    echo "========================================================================"

    echo "Verifying head node is still running..."
    if ! squeue -j "${RAY_HEAD_JOB_ID}" -h &>/dev/null; then
        echo "ERROR: Ray head node job ${RAY_HEAD_JOB_ID} is not running"
        echo ""
        echo "Check job status:"
        echo "  squeue -j ${RAY_HEAD_JOB_ID}"
        echo ""
        echo "If the job finished, you need to start a new head node:"
        echo "  sbatch launch_head.sh"
        exit 1
    fi
    echo "Head node job ${RAY_HEAD_JOB_ID} is running"
fi

# Export RAY_ADDRESS for ray.init(address='auto')
export RAY_ADDRESS

# Get port configuration from connection file (if available)
#USER_PORT_OFFSET="${USER_PORT_OFFSET:-0}"
#USER_PORT_OFFSET_MINOR=$((USER_PORT_OFFSET / 40))
#MIN_WORKER_PORT=$((11000 + USER_PORT_OFFSET))
#MAX_WORKER_PORT=$((13411 + USER_PORT_OFFSET))

# ============================================================================
# Configuration
# ============================================================================

# Setup environment variables - must be called after PYTHON_SCRIPT is set
if [ "${RUN_PYTHON_SCRIPT}" = "true" ]; then
    setup_environment_vars "${PYTHON_SCRIPT}"
else
    RUN_ID="worker-only-${SLURM_JOB_ID}"
    ENTRY_POINT="N/A"
    ENTRY_POINT_ID="N/A"
fi

# Ray configuration for worker node
RAY_OBJECT_STORE="${RAY_OBJECT_STORE:-10}"  # GB
RAY_TMPDIR="${TMPDIR:-/tmp}/ray_client_${SLURM_JOB_ID}"

# Output directories (likely unused)
LOG_DIR="${WORKSPACE_DIR}/outputs/slurm_logs"

# ============================================================================
# Setup
# ============================================================================

echo "========================================================================"
echo "Ray Client Job ${SLURM_JOB_ID}"
echo "========================================================================"
echo "Job name:       ${SLURM_JOB_NAME}"
echo "RUN_ID:         ${RUN_ID}"
echo "Entry point:    ${ENTRY_POINT} (ID: ${ENTRY_POINT_ID})"
echo "Node:           ${SLURM_JOB_NODELIST}"
echo "CPUs:           ${SLURM_CPUS_PER_TASK}"
echo "GPUs:           ${SLURM_GPUS_PER_TASK:-0}"
echo "Memory:         ${SLURM_MEM_PER_NODE}MB"
echo "Workspace:      ${WORKSPACE_DIR}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "========================================================================"
echo "Ray Configuration"
echo "  Connecting to:  ${RAY_ADDRESS}"
echo "  Object store:   ${RAY_OBJECT_STORE}GB"
echo "  Extra options:  ${RAY_START_OPTIONS[*]:-none}"
echo "========================================================================"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}"

if [ "${RUN_PYTHON_SCRIPT}" = "true" ]; then
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
        if [ "${CONNECTION_MODE}" = "job_id" ]; then
            echo -e "${SLURM_JOB_ID}\t${RUN_ID}\t${PYTHON_SCRIPT}\t$*\t(connected to head job ${RAY_HEAD_JOB_ID})"
        else
            echo -e "${SLURM_JOB_ID}\t${RUN_ID}\t${PYTHON_SCRIPT}\t$*\t(connected to ${RAY_ADDRESS})"
        fi
    } >> "${OVERVIEW_FILE}"
fi

# Activate virtual environment
activate_virtualenv

# Verify Python and dependencies
print_python_info

# ============================================================================
# Connect to Ray Cluster as Worker
# ============================================================================

echo "========================================================================"
echo "Connecting to Ray Cluster"
echo "========================================================================"
echo "Head address:   ${RAY_ADDRESS}"
echo "This node:      $(hostname)"
echo "========================================================================"

# Test connection to head node
echo "Testing connection to head node..."
if ! ping -c 1 -W 5 "${HEAD_IP}" &>/dev/null; then
    echo "WARNING: Cannot ping head node ${HEAD_IP}"
    echo "This may be normal if ICMP is blocked, continuing anyway..."
fi

# Connect this node to the Ray cluster as a worker
echo "Connecting as Ray worker..."
if [ "${RUN_PYTHON_SCRIPT}" = "false" ]; then
    # Worker-only mode: use --block to keep ray start running

    echo "========================================================================"
    echo "Worker Mode - No Python Script"
    echo "========================================================================"
    echo "This node is now connected to the Ray cluster as a worker."
    echo "It will remain connected until the job ends or is cancelled."
    echo ""
    echo "To use this worker, run Python scripts on other nodes that connect"
    if [ "${CONNECTION_MODE}" = "job_id" ]; then
        echo "to the same Ray cluster (head job ${RAY_HEAD_JOB_ID})."
    else
        echo "to the same Ray cluster (${RAY_ADDRESS})."
    fi
    echo "========================================================================"
    if ! ray start \
        --address="${RAY_ADDRESS}" \
        --object-store-memory="${RAY_OBJECT_STORE}000000000" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${SLURM_GPUS_PER_TASK:-0}" \
        --temp-dir="${RAY_TMPDIR}" \
        --labels="hostname=$(hostname -s),head=false,slurmnode=true" \
        "${RAY_START_OPTIONS[@]}" \
        --block; then
        echo "ERROR: Failed to connect to Ray cluster"
        if [ "${CONNECTION_MODE}" = "job_id" ]; then
            echo "Check that head node (job ${RAY_HEAD_JOB_ID}) is running and accessible"
        else
            echo "Check that head node at ${RAY_ADDRESS} is running and accessible"
        fi
        exit 1
    fi
else
    # Python script mode: normal connection without blocking
    if ! ray start \
        --address="${RAY_ADDRESS}" \
        --min-worker-port="${MIN_WORKER_PORT}" \
        --max-worker-port="${MAX_WORKER_PORT}" \
        --object-store-memory="${RAY_OBJECT_STORE}000000000" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${SLURM_GPUS_PER_TASK:-0}" \
        --temp-dir="${RAY_TMPDIR}" \
        "${RAY_START_OPTIONS[@]}"; then
        echo "ERROR: Failed to connect to Ray cluster"
        if [ "${CONNECTION_MODE}" = "job_id" ]; then
            echo "Check that head node (job ${RAY_HEAD_JOB_ID}) is running and accessible"
        else
            echo "Check that head node at ${RAY_ADDRESS} is running and accessible"
        fi
        exit 1
    fi

    sleep 5

    echo "========================================================================"
    echo "Checking Ray Cluster Status"
    echo "========================================================================"
    ray status || echo "WARNING: Could not get ray status (may still be initializing)"

    # ============================================================================
    # Run Experiment or Block as Worker
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
    echo "Arguments:      ${PYTHON_ARGS[*]}"
    echo "Ray address:    ${RAY_ADDRESS} (via RAY_ADDRESS env var)"
    echo "RUN_ID:         ${RUN_ID}"
    echo "Entry point:    ${ENTRY_POINT} (ID: ${ENTRY_POINT_ID})"
    echo "========================================================================"
    echo ""
    echo "Python script should use ray.init(address='auto') to connect"
    echo "========================================================================"

    # Run Python script in background to capture PID
    python -u "${PYTHON_SCRIPT}" --log_level IMPORTANT_INFO "${PYTHON_ARGS[@]}" &
    PYTHON_PID=$!

    # Wait for Python process to complete
    wait $PYTHON_PID
    EXIT_CODE=$?

    echo "========================================================================"
    echo "Python script finished with exit code: ${EXIT_CODE}"
    echo "========================================================================"
fi

# ============================================================================
# Cleanup
# ============================================================================

echo "Disconnecting from Ray cluster..."
ray stop || echo "WARNING: Ray stop failed (node may already be disconnected)"

# Wait for any background jobs
timeout 30 wait || true

# Sync results to backup if configured
if [ "${USE_BACKUP_DUMP_DIR:-false}" = "true" ] && [ -n "${BACKUP_DUMP_DIR:-}" ]; then
    echo "Syncing results to backup: ${BACKUP_DUMP_DIR}..."
    rsync -avz "${RAY_TMPDIR}/" "${BACKUP_DUMP_DIR}/" || echo "WARNING: backup sync failed"
fi

echo "========================================================================"
echo "Job Complete"
echo "========================================================================"
echo "Exit code:      ${EXIT_CODE:-0}"
echo "Results:        ${OUTPUT_DIR}"
echo "Logs:           ${LOG_DIR}"
if [ "${CONNECTION_MODE}" = "job_id" ]; then
    echo "Head node:      ${RAY_HEAD_JOB_ID}"
else
    echo "Head address:   ${RAY_ADDRESS}"
fi
echo "Backup dump:    ${BACKUP_DUMP_DIR:-none}"
echo "========================================================================"

exit ${EXIT_CODE:-0}
