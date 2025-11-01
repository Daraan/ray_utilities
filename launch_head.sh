#!/bin/bash
#SBATCH --job-name=ray-head
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=30:00:00
#SBATCH --signal=B:SIGUSR1@600
#SBATCH --output=outputs/slurm_logs/%j-%x.out
#SBATCH --error=outputs/slurm_logs/%j-%x.err

################################################################################
# SLURM Script for Persistent Ray Head Node
#
# Launches a persistent Ray head node that can be shared by multiple jobs.
# Monitors for idle periods during off-hours and forwards timeout signals.
#
# Usage:
#   USER_PORT_OFFSET=10000 sbatch launch_head.sh
#
# Environment Variables:
#   USER_PORT_OFFSET - Port offset for this user (default: auto-calculated)
#   RAY_OBJECT_STORE - Object store memory in GB (default: 10)
#   IDLE_CHECK_ENABLED - Enable idle monitoring (default: true)
#
# Connecting Jobs:
#   Set RAY_ADDRESS="HEAD_NODE_IP:RAY_PORT" in your job script, then use:
#   ray.init(address='auto', runtime_env=runtime_env)
#
# Monitoring:
#   - Checks for idle cluster between 2:30am-9am
#   - Shuts down if no jobs are connected during this window
#   - Forwards SIGUSR1 to all connected Ray jobs on timeout
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Source common base script
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ORIGINAL_SCRIPT=$(scontrol show job "${SLURM_JOB_ID}" | grep -oP 'Command=\K[^ ]+' | head -1)
    if [[ "${ORIGINAL_SCRIPT}" =~ ^/ ]]; then
        SCRIPT_DIR="$(dirname "${ORIGINAL_SCRIPT}")"
    else
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/$(dirname "${ORIGINAL_SCRIPT}")"
    fi
else
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
fi

# Check if common_base.sh exists, otherwise use minimal setup
if [ -f "${SCRIPT_DIR}/common_base.sh" ]; then
    source "${SCRIPT_DIR}/common_base.sh"

    activate_virtualenv
else
    echo "WARNING: common_base.sh not found, using minimal setup"
    WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
    OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs}"

    # Fallback: Try to activate virtual environment manually
    if [ -f "../env/bin/activate" ]; then
        echo "Activating: ../env/bin/activate"
        source "../env/bin/activate"
    elif [ -f "./env/bin/activate" ]; then
        echo "Activating: ./env/bin/activate"
        source "./env/bin/activate"
    else
        echo "WARNING: No virtual environment found. Ray must be available in base environment."
    fi
fi

# Verify Ray is available
echo "Verifying Ray installation..."
if ! command -v ray &> /dev/null; then
    echo "ERROR: 'ray' command not found"
    echo "Python: $(which python 2>&1 || echo 'not found')"
    echo "Virtual environment active: ${VIRTUAL_ENV:-none}"
    exit 1
fi
echo "Ray found: $(which ray)"


# ============================================================================
# Signal Handler for Timeout
# ============================================================================

handle_timeout() {
    echo "========================================================================"
    echo "WARNING: Received timeout signal (10 minutes remaining)"
    echo "========================================================================"
    echo "Time: $(date)"
    echo "Stopping Ray with grace period to allow connected jobs to clean up..."

    # Get all SLURM jobs that might be using this Ray cluster
    # We look for jobs with RAY_ADDRESS matching our head node
    local head_address="${HEAD_IP}:${RAY_PORT}"
    local connected_jobs=$(squeue -u "${USER}" -h -o "%A %j" | grep -v "ray-head" | awk '{print $1}')

    # Sends SIGTERM - want to send SIGINT or SIGUSR1 but ray stop does not support that

    # 1. Get all running jobs of the user
    job_list=$(squeue -u "${USER}" -h -o "%A %j" 2>/dev/null || echo "")
    # Take only jobs that have "client" in their name (assuming they are Ray jobs)
    ray_jobs=$(echo "${job_list}" | grep "ray-client" | awk '{print $1}' || echo "")

    # 2. Need to filter out which jobs are connected to this Ray head node
    # We check outputs/slurm_logs/*-ray-client-*.out for RAY_ADDRESS in the first 64 lines
    filtered_ray_jobs=""
    if [ -n "${ray_jobs}" ]; then
        echo "DEBUG: Filtering ${ray_jobs} for connections to ${head_address}"
        echo "DEBUG: Looking in ${OUTPUT_DIR}/slurm_logs/ for log files"
        echo "DEBUG: OUTPUT_DIR=${OUTPUT_DIR}"
        echo "DEBUG: Full search path: ${OUTPUT_DIR}/slurm_logs/"

        # List all files in the directory to see what's actually there
        if [ -d "${OUTPUT_DIR}/slurm_logs" ]; then
            echo "DEBUG: Directory exists.:"
        else
            echo "DEBUG: Directory does not exist: ${OUTPUT_DIR}/slurm_logs"
        fi

        for job_id in ${ray_jobs}; do
            echo "DEBUG: Checking job ${job_id}"
            # Use shopt nullglob to handle cases where no files match
            shopt -s nullglob
            # Look for log files matching this specific job ID
            # Try multiple patterns to be more flexible
            log_files=("${OUTPUT_DIR}/slurm_logs/${job_id}-"*".out")
            shopt -u nullglob

            echo "DEBUG: Pattern used: ${OUTPUT_DIR}/slurm_logs/${job_id}-*.out"
            echo "DEBUG: Found ${#log_files[@]} log files for job ${job_id}"

            if [ ${#log_files[@]} -eq 0 ]; then
                echo "DEBUG: No files found. Trying alternative locations..."
                # Try relative path from SLURM_SUBMIT_DIR
                if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
                    shopt -s nullglob
                    log_files=("${SLURM_SUBMIT_DIR}/outputs/slurm_logs/${job_id}-"*".out")
                    shopt -u nullglob
                    echo "DEBUG: Tried ${SLURM_SUBMIT_DIR}/outputs/slurm_logs/${job_id}-*.out - found ${#log_files[@]}"
                fi
            fi

            for log_file in "${log_files[@]}"; do
                echo "DEBUG: Checking log file: ${log_file}"
                if [ -f "${log_file}" ]; then
                    echo "DEBUG: File exists, size: $(stat -c%s "${log_file}" 2>/dev/null || echo 'unknown')"
                    # Look for the head address in various formats in the log file (first 100 lines)
                    # Match patterns like:
                    # - "Head address:   10.5.85.5:6659"
                    # - "Ray address:    10.5.85.5:6659"
                    # - "RAY_ADDRESS=10.5.85.5:6659"
                    if head -n 100 "${log_file}" 2>/dev/null | grep -q "${head_address}"; then
                        echo "DEBUG: Job ${job_id} is connected to this head node (found ${head_address})"
                        filtered_ray_jobs="${filtered_ray_jobs} ${job_id}"
                        break
                    else
                        echo "DEBUG: Job ${job_id} log file exists but head address ${head_address} not found"
                        echo "DEBUG: First 10 lines of log file:"
                        head -n 10 "${log_file}" 2>/dev/null || echo "DEBUG: Failed to read file"
                    fi
                else
                    echo "DEBUG: Log file does not exist: ${log_file}"
                fi
            done
        done
    fi
    ray_jobs="${filtered_ray_jobs}"
    echo "Found connected Ray jobs: ${ray_jobs:-none} to ray head_address=${head_address}"

    if [ -n "${ray_jobs}" ]; then
        echo "Forwarding SIGUSR1 to all connected Ray jobs..."
        # Send SIGUSR1 to each the batch script and the child processs
        #for job_id in ${ray_jobs}; do
        #    echo "Sending SIGUSR2 to job batch script of ID: ${job_id}"
        #    scancel --signal=SIGUSR2 --batch "${job_id}" 2>/dev/null || echo "WARNING: Failed to send SIGUSR2 to job ${job_id}"
        #done
        #sleep 5
        for job_id in ${ray_jobs}; do
            echo "Sending SIGUSR1 to job ID: ${job_id}"
            scancel --signal=SIGUSR1 "${job_id}" 2>/dev/null || echo "WARNING: Failed to send SIGUSR1 to job ${job_id}"
        done
        sleep 5

        # wait until all jobs are done on the cluster or for 400 seconds
        local wait_time=0
        local max_wait=400
        while true; do
            local remaining_jobs=0
            for job_id in ${ray_jobs}; do
                if squeue -j "${job_id}" -h 2>/dev/null | grep -q "${job_id}"; then
                    remaining_jobs=$((remaining_jobs + 1))
                fi
            done
            if [ "${remaining_jobs}" -eq 0 ]; then
                echo "All connected Ray jobs have finished."
                sleep 5 # wait a bit before stopping ray
                break
            fi
            if [ "${wait_time}" -ge "${max_wait}" ]; then
                echo "WARNING: Timeout reached while waiting for connected Ray jobs to finish."
                echo "Proceeding with shutdown. ${remaining_jobs} job(s) still running."
                break
            fi
            echo "Waiting for connected Ray jobs to finish... (${remaining_jobs} remaining)"
            sleep 15
            wait_time=$((wait_time + 15))
        done
    else
        echo "No connected Ray jobs found."
    fi

    echo "Stopping Ray head node..."
    if [ -z "${ray_jobs}" ]; then
        echo "No connected Ray jobs found. Stopping Ray and exiting."
        ray stop --grace-period 40
        cleanup_and_exit 0
    else
        ray stop --grace-period 40
    fi
}

trap 'handle_timeout' SIGUSR1

# ============================================================================
# Cleanup Function
# ============================================================================

cleanup_and_exit() {
    local exit_code=${1:-0}
    echo "========================================================================"
    echo "Cleaning up Ray head node..."
    echo "========================================================================"

    ray stop --grace-period 30 || echo "WARNING: Ray stop failed"

    #if [ -n "${RAY_TMPDIR:-}" ] && [ -d "${RAY_TMPDIR}" ]; then
    #    echo "Cleaning up temporary directory: ${RAY_TMPDIR}"
    #    rm -rf "${RAY_TMPDIR}" || echo "WARNING: Failed to clean up temp directory"
    #fi

    echo "========================================================================"
    echo "Ray Head Node Stopped"
    echo "========================================================================"
    exit "${exit_code}"
}

trap 'cleanup_and_exit 1' EXIT ERR

# ============================================================================
# Port Configuration
# ============================================================================

if [ -z "${USER_PORT_OFFSET:-}" ]; then
    USER_PORT_OFFSET=$((((SLURM_JOB_ID % 50) * 1000) % 40000))
    echo "INFO: Auto-calculated USER_PORT_OFFSET=${USER_PORT_OFFSET} from job ID"
fi

USER_PORT_OFFSET_MINOR=$((USER_PORT_OFFSET / 40))

# Check for dws-## pattern in node list
if [[ "${SLURM_JOB_NODELIST}" =~ dws-([0-9]+) ]]; then
    DWS_NUMBER=$((10#${BASH_REMATCH[1]}))
    if [ "${DWS_NUMBER}" -ge 1 ] && [ "${DWS_NUMBER}" -le 17 ]; then
        USER_PORT_OFFSET=$((DWS_NUMBER * 2420))
        echo "INFO: Detected dws-${DWS_NUMBER}, set USER_PORT_OFFSET=${USER_PORT_OFFSET}"
        USER_PORT_OFFSET_MINOR=$((DWS_NUMBER * 40))
    fi
fi

# Ray ports
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

RAY_OBJECT_STORE="${RAY_OBJECT_STORE:-20}"  # GB
RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_head_${SLURM_JOB_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs}"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
IDLE_CHECK_ENABLED="${IDLE_CHECK_ENABLED:-true}"

# ============================================================================
# Setup
# ============================================================================

echo "========================================================================"
echo "Ray Utilities - Persistent Head Node (Job ${SLURM_JOB_ID})"
echo "========================================================================"
echo "Job name:       ${SLURM_JOB_NAME}"
echo "Node:           ${SLURM_JOB_NODELIST}"
echo "CPUs:           ${SLURM_CPUS_PER_TASK}"
echo "Memory:         ${SLURM_MEM_PER_NODE}MB"
echo "Workspace:      ${WORKSPACE_DIR:-N/A}"
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
echo "Idle monitoring:  ${IDLE_CHECK_ENABLED}"
echo "========================================================================"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}"

# Display Python/Ray info
if command -v print_python_info &> /dev/null; then
    print_python_info
else
    echo "Python: $(which python) ($(python --version 2>&1))"
    echo "Ray:    $(python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
fi

# ============================================================================
# Start Ray Head Node
# ============================================================================

HEAD_NODE=$(hostname)
HEAD_IP=$(hostname -I | awk '{print $1}')
HEAD_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "========================================================================"
echo "Ray Head Node Setup"
echo "========================================================================"
echo "Head node:      ${HEAD_NODE}"
echo "Head IP:        ${HEAD_IP}"
echo "Head address:   ${HEAD_ADDRESS}"
echo "========================================================================"
echo ""
echo "To connect from other jobs, set:"
echo "  export RAY_ADDRESS=\"${HEAD_ADDRESS}\""
echo "  ray.init(address='auto')"
echo "========================================================================"

# Start head node
echo "Starting Ray head node..."
ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-port="${DASHBOARD_PORT}" \
    --dashboard-agent-grpc-port="${DASHBOARD_AGENT_GRPC_PORT}" \
    --include-dashboard="${RAY_INCLUDE_DASHBOARD:-true}" \
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
    --disable-usage-stats

sleep 10

echo "========================================================================"
echo "Checking Ray Head Node Status"
echo "========================================================================"
if ! ray status; then
    echo "ERROR: Ray cluster not responding"
    echo "Attempting to restart Ray head node..."
    ray stop --grace-period 10 || echo "WARNING: Ray stop during restart failed"
    sleep 5

    # Retry starting Ray head node
    ray start --head \
        --node-ip-address="${HEAD_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-port="${DASHBOARD_PORT}" \
        --dashboard-agent-grpc-port="${DASHBOARD_AGENT_GRPC_PORT}" \
        --include-dashboard="${RAY_INCLUDE_DASHBOARD:-true}" \
        --node-manager-port="${NODE_MANAGER_PORT}" \
        --object-manager-port="${OBJECT_MANAGER_PORT}" \
        --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
        --redis-shard-ports="${REDIS_SHARD_PORT}" \
        --min-worker-port="${MIN_WORKER_PORT}" \
        --max-worker-port="${MAX_WORKER_PORT}" \
        --object-store-memory="${RAY_OBJECT_STORE}000000000" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${SLURM_GPUS_PER_TASK:-0}" \
        --temp-dir="${RAY_TMPDIR}"

    sleep 10

    if ! ray status; then
        echo "ERROR: Ray cluster still not responding after restart attempt"
        cleanup_and_exit 1
    fi
fi

# Save connection info
CONNECTION_FILE="${OUTPUT_DIR}/ray_head_${SLURM_JOB_ID}.info"
cat > "${CONNECTION_FILE}" << EOF
RAY_ADDRESS=${HEAD_ADDRESS}
RAY_PORT=${RAY_PORT}
HEAD_NODE=${HEAD_NODE}
HEAD_IP=${HEAD_IP}
SLURM_JOB_ID=${SLURM_JOB_ID}
USER_PORT_OFFSET=${USER_PORT_OFFSET}
STARTED=$(date -Iseconds)
EOF

echo "Connection info saved to: ${CONNECTION_FILE}"

# ============================================================================
# Idle Monitoring Loop
# ============================================================================

check_idle_and_shutdown() {
    # Only check during off-hours (2:30am - 9:00am)
    local current_hour=$(date +%H)
    local current_minute=$(date +%M)
    local current_time=$((10#${current_hour} * 60 + 10#${current_minute}))
    local start_time=$((2 * 60 + 30))  # 2:30am = 150 minutes
    local end_time=$((9 * 60))         # 9:00am = 540 minutes

    if [ "${current_time}" -lt "${start_time}" ] || [ "${current_time}" -ge "${end_time}" ]; then
        return 0  # Not in monitoring window
    fi

    echo "[$(date +%H:%M:%S)] Checking for connected jobs (idle monitoring window)..."

    # Check Ray cluster status
    local node_count=$(ray status 2>/dev/null | grep -c "node_" || echo "0")

    # If only head node (1 node), check for active jobs
    if [ "${node_count}" -le 1 ]; then
        # Check for running tasks
        local active_tasks=$(ray status 2>/dev/null | grep -oP 'RUNNING: \K[0-9]+' || echo "0")

        if [ "${active_tasks}" -eq 0 ]; then
            echo "========================================================================"
            echo "IDLE SHUTDOWN: No active jobs during monitoring window"
            echo "========================================================================"
            echo "Time:           $(date)"
            echo "Active tasks:   ${active_tasks}"
            echo "Connected nodes: ${node_count}"
            echo "========================================================================"
            cleanup_and_exit 0
        else
            echo "  Active tasks: ${active_tasks} - continuing"
        fi
    else
        echo "  Connected nodes: ${node_count} - continuing"
    fi
}

echo "========================================================================"
echo "Ray Head Node Running - Blocking Until Stopped"
echo "========================================================================"
echo "Connection:     ${HEAD_ADDRESS}"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Time limit:     ${SLURM_JOB_END_TIME:-unknown}"
echo ""
echo "Monitoring:     Every 5 minutes during 2:30am-9am"
echo "========================================================================"

# Main blocking loop with periodic idle checks
while true; do
    # Check if idle monitoring is enabled
    if [ "${IDLE_CHECK_ENABLED}" = "true" ]; then
        check_idle_and_shutdown
    fi

    # Sleep in smaller intervals to remain responsive to signals
    for _ in {1..60}; do
        sleep 10
    done
done
