#!/bin/bash
################################################################################
# Ray Cluster Client Connection Utility
#
# This script simplifies connecting client jobs to a running Ray head node.
# It validates the head node is running and submits a client job with the
# correct connection parameters.
#
# Usage:
#   ./launch_client.sh <HEAD_JOB_ID> [SBATCH_OPTIONS...] <PYTHON_FILE> [PYTHON_ARGS...]
#   ./launch_client.sh <RAY_ADDRESS> [SBATCH_OPTIONS...]
#
# Examples:
#   # Basic usage with head job ID
#   ./launch_client.sh 12345 experiments/tune.py --seed 42
#
#   # Worker-only mode with direct address
#   ./launch_client.sh 10.5.88.207:7304
#   ./launch_client.sh 10.5.88.207:7304 --time=2:00:00 --mem=50G
#
#   # With custom SLURM options (before .py file)
#   ./launch_client.sh 12345 --time=2:00:00 --mem=50G experiments/tune.py --seed 42
#
#   # With environment variables
#   RAY_OBJECT_STORE=20 ./launch_client.sh 12345 experiments/tune.py
#
# Prerequisites:
#   - For HEAD_JOB_ID mode: Ray head node must be running (launched via launch_head.sh)
#   - For RAY_ADDRESS mode: Ray head node must be accessible at the given address
#
# Notes:
#   - Uses --dependency=after:HEAD_JOB_ID+1 to wait 1 minute after head starts (job ID mode only)
#   - Connection file validation happens in the submitted job (_launch_client.sh)
#   - Client job will wait up to 60s for connection file to appear (job ID mode only)
#
# Environment Variables:
#   RAY_OBJECT_STORE - Object store memory in GB (default: 10)
#   WORKSPACE_DIR    - Workspace directory (default: auto-detected)
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Color output helpers
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

error() {
    echo -e "${RED}ERROR:${NC} $*" >&2
}

info() {
    echo -e "${BLUE}INFO:${NC} $*"
}

success() {
    echo -e "${GREEN}SUCCESS:${NC} $*"
}

# shellcheck disable=SC2329
warning() {
    echo -e "${YELLOW}WARNING:${NC} $*"
}

# ============================================================================
# Usage
# ============================================================================

usage() {
    cat << EOF
Usage: $(basename "$0") <HEAD_JOB_ID|RAY_ADDRESS> [SBATCH_OPTIONS...] [PYTHON_FILE] [PYTHON_ARGS...]

Connect a client job to a running Ray head node.

Arguments:
  HEAD_JOB_ID      SLURM job ID of the Ray head node (numeric)
  RAY_ADDRESS      Direct Ray cluster address (format: IP:PORT)
  SBATCH_OPTIONS   SLURM options (before .py file, if provided)
  PYTHON_FILE      Path to Python script to run (.py file, optional)
  PYTHON_ARGS      Arguments to pass to Python script (after .py file)

Examples:
  # Connect with Python script via job ID
  $(basename "$0") 12345 experiments/tune.py --seed 42

  # Connect as worker only via job ID
  $(basename "$0") 12345

  # Connect as worker only via direct address
  $(basename "$0") 10.5.88.207:7304
  $(basename "$0") 10.5.88.207:7304 --time=4:00:00 --mem=50G

  # With custom SLURM options
  $(basename "$0") 12345 --time=4:00:00 --mem=50G experiments/tune.py --seed 42
  RAY_OBJECT_STORE=20 $(basename "$0") 12345 experiments/tune.py

Prerequisites:
  - Ray head node must be running (sbatch launch_head.sh) or accessible

Notes:
  - Job ID mode: Uses SLURM dependency to wait 1 minute after head starts
  - Address mode: Connects immediately to the given address
  - Client job will wait up to 60s for connection file (job ID mode only)

EOF
    exit 1
}

# ============================================================================
# Parse Arguments
# ============================================================================

if [ $# -lt 1 ]; then
    error "Insufficient arguments"
    echo ""
    usage
fi

FIRST_ARG="$1"
shift

# Determine if first argument is a job ID (numeric) or RAY_ADDRESS (contains colon)
if [[ "${FIRST_ARG}" =~ ^[0-9]+$ ]]; then
    # Numeric - treat as HEAD_JOB_ID
    CONNECTION_MODE="job_id"
    HEAD_JOB_ID="${FIRST_ARG}"
    RAY_ADDRESS=""
elif [[ "${FIRST_ARG}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$ ]]; then
    # IP:PORT format - treat as RAY_ADDRESS
    CONNECTION_MODE="direct"
    HEAD_JOB_ID=""
    RAY_ADDRESS="${FIRST_ARG}"
else
    error "First argument must be either a numeric job ID or an address in IP:PORT format"
    error "Got: ${FIRST_ARG}"
    echo ""
    usage
fi

# Find -- separator (for Ray start options)
# Everything before -- = SBATCH options or Python file + args
# Everything after -- and before .py = Ray start options
# Everything after .py = Python arguments
RAY_START_OPTIONS=()
RAY_SEPARATOR_INDEX=0

for i in $(seq 1 $#); do
    if [ "${!i}" = "--" ]; then
        RAY_SEPARATOR_INDEX=$i
        break
    fi
done

# Parse based on whether -- separator exists
if [ ${RAY_SEPARATOR_INDEX} -gt 0 ]; then
    # With -- separator: parse SBATCH options, Ray options, and Python script/args
    PYTHON_FILE=""
    PYTHON_FILE_INDEX=0
    SBATCH_OPTIONS=()
    PYTHON_ARGS=()

    # Collect SBATCH options (everything before --)
    for i in $(seq 1 $((RAY_SEPARATOR_INDEX - 1))); do
        SBATCH_OPTIONS+=("${!i}")
    done

    # Collect Ray start options and find Python file (everything after --)
    for i in $(seq $((RAY_SEPARATOR_INDEX + 1)) $#); do
        arg="${!i}"
        if [[ "${arg}" == *.py ]] && [ -z "${PYTHON_FILE}" ]; then
            PYTHON_FILE="${arg}"
            PYTHON_FILE_INDEX=$i
            break
        else
            RAY_START_OPTIONS+=("${arg}")
        fi
    done

    # Collect Python args (everything after .py file)
    if [ -n "${PYTHON_FILE}" ]; then
        for i in $(seq $((PYTHON_FILE_INDEX + 1)) $#); do
            PYTHON_ARGS+=("${!i}")
        done
    fi
else
    # No -- separator: original behavior
    PYTHON_FILE=""
    PYTHON_FILE_INDEX=0
    SBATCH_OPTIONS=()
    PYTHON_ARGS=()

    for i in $(seq 1 $#); do
        arg="${!i}"
        if [[ "${arg}" == *.py ]]; then
            PYTHON_FILE="${arg}"
            PYTHON_FILE_INDEX=$i
            break
        fi
    done

    if [ -z "${PYTHON_FILE}" ]; then
        # No Python file - all remaining args are SBATCH options
        SBATCH_OPTIONS=("$@")
    else
        # Collect SBATCH options (everything before .py file)
        for i in $(seq 1 $((PYTHON_FILE_INDEX - 1))); do
            SBATCH_OPTIONS+=("${!i}")
        done

        # Collect Python args (everything after .py file)
        for i in $(seq $((PYTHON_FILE_INDEX + 1)) $#); do
            PYTHON_ARGS+=("${!i}")
        done
    fi
fi

# ============================================================================
# Detect Workspace
# ============================================================================

if [ -z "${WORKSPACE_DIR:-}" ]; then
    # Try ws_find if available
    if command -v ws_find >/dev/null 2>&1; then
        WORKSPACE_DIR=$(ws_find master_workspace 2>/dev/null || echo "")
    fi

    # Fallback to script location
    if [ -z "${WORKSPACE_DIR:-}" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        WORKSPACE_DIR="${SCRIPT_DIR}"
    fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs}"

# ============================================================================
# Display Configuration
# ============================================================================

echo "========================================================================"
echo "Ray Client Connection Utility"
echo "========================================================================"
if [ "${CONNECTION_MODE}" = "job_id" ]; then
    info "Connection mode:  Via Head Job ID"
    info "Head Job ID:      ${HEAD_JOB_ID}"
else
    info "Connection mode:  Direct Address"
    info "Ray Address:      ${RAY_ADDRESS}"
fi
if [ -n "${PYTHON_FILE}" ]; then
    info "Python script:    ${PYTHON_FILE}"
    info "Python args:      ${PYTHON_ARGS[*]:-none}"
else
    info "Mode:             Worker only (no Python script)"
fi
info "SBATCH options:   ${SBATCH_OPTIONS[*]:-none}"
info "Ray start opts:   ${RAY_START_OPTIONS[*]:-none}"
info "Workspace:        ${WORKSPACE_DIR}"
echo "========================================================================"

# ============================================================================
# Validate Head Node (Job ID mode only)
# ============================================================================

if [ "${CONNECTION_MODE}" = "job_id" ]; then
    CONNECTION_FILE="${OUTPUT_DIR}/ray_head_${HEAD_JOB_ID}.info"

    info "Checking if head node job ${HEAD_JOB_ID} is running..."
    if ! squeue -j "${HEAD_JOB_ID}" -h &>/dev/null; then
        error "Ray head node job ${HEAD_JOB_ID} is not running"
        echo ""
        echo "Check job status with: squeue -j ${HEAD_JOB_ID}"
        echo ""
        echo "To start a new head node:"
        echo "  sbatch launch_head.sh"
        exit 1
    fi
    success "Head node job ${HEAD_JOB_ID} is running"

    if [ -f "${CONNECTION_FILE}" ]; then
        info "Connection file found - reading head node details..."
        if source "${CONNECTION_FILE}" 2>/dev/null; then
            if [ -n "${RAY_ADDRESS:-}" ] && [ -n "${HEAD_NODE:-}" ]; then
                success "Connection details available"
                info "  RAY_ADDRESS:    ${RAY_ADDRESS}"
                info "  HEAD_NODE:      ${HEAD_NODE}"
                info "  RAY_PORT:       ${RAY_PORT:-unknown}"
                info "  Started:        ${STARTED:-unknown}"
            fi
        fi
    else
        info "Connection file not yet available (client job will wait for it)"
    fi
fi

# Verify Python script exists (if provided)
if [ -n "${PYTHON_FILE}" ]; then
    if [ ! -f "${PYTHON_FILE}" ]; then
        error "Python script not found: ${PYTHON_FILE}"
        echo "Current directory: $(pwd)"
        exit 1
    fi
    success "Python script found"
fi

# ============================================================================
# Submit Client Job
# ============================================================================

echo "========================================================================"
info "Submitting client job..."
echo "========================================================================"

# Build sbatch command
SBATCH_CMD=("sbatch")

# Add dependency only for job ID mode
if [ "${CONNECTION_MODE}" = "job_id" ]; then
    SBATCH_CMD+=("--dependency=after:${HEAD_JOB_ID}+1")
fi

# Add user SBATCH options
SBATCH_CMD+=("${SBATCH_OPTIONS[@]}")

# Export environment variables
if [ "${CONNECTION_MODE}" = "job_id" ]; then
    SBATCH_CMD+=("--export=ALL,RAY_HEAD_JOB_ID=${HEAD_JOB_ID}")
else
    SBATCH_CMD+=("--export=ALL,RAY_ADDRESS=${RAY_ADDRESS}")
fi

# Export Ray start options if provided
if [ ${#RAY_START_OPTIONS[@]} -gt 0 ]; then
    RAY_START_OPTS_STR="${RAY_START_OPTIONS[*]}"
    SBATCH_CMD[-1]="${SBATCH_CMD[-1]},RAY_START_OPTS=${RAY_START_OPTS_STR}"
fi

# Add script and arguments
SBATCH_CMD+=("experiments/slurm/_launch_client.sh")
if [ -n "${PYTHON_FILE}" ]; then
    SBATCH_CMD+=("${PYTHON_FILE}" "${PYTHON_ARGS[@]}")
fi

# Display command
echo "Command:"
echo "  ${SBATCH_CMD[*]}"
echo ""

# Submit job
if JOB_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1); then
    # Extract job ID from output (format: "Submitted batch job 12345")
    CLIENT_JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")

    if [ -n "${CLIENT_JOB_ID}" ]; then
        echo "========================================================================"
        success "Client job submitted successfully!"
        echo "========================================================================"
        info "Client job ID:    ${CLIENT_JOB_ID}"
        if [ "${CONNECTION_MODE}" = "job_id" ]; then
            info "Head job ID:      ${HEAD_JOB_ID}"
        else
            info "Ray address:      ${RAY_ADDRESS}"
        fi
        if [ -n "${PYTHON_FILE}" ]; then
            info "Python script:    ${PYTHON_FILE}"
        else
            info "Mode:             Worker only"
        fi
        echo ""
        echo "Monitor job with:"
        echo "  squeue -j ${CLIENT_JOB_ID}"
        echo "  tail -f ${OUTPUT_DIR}/slurm_logs/${CLIENT_JOB_ID}-ray-client-*.out"
        echo ""
        if [ "${CONNECTION_MODE}" = "job_id" ]; then
            if [ -n "${RAY_ADDRESS:-}" ]; then
                echo "Check Ray cluster status:"
                echo "  export RAY_ADDRESS=${RAY_ADDRESS}"
                echo "  ray status"
            else
                echo "Once head node is ready, check cluster status:"
                echo "  source ${CONNECTION_FILE}"
                echo "  export RAY_ADDRESS"
                echo "  ray status"
            fi
        else
            echo "Check Ray cluster status:"
            echo "  export RAY_ADDRESS=${RAY_ADDRESS}"
            echo "  ray status"
        fi
        echo "========================================================================"
    else
        error "Job submitted but could not extract job ID"
        echo "${JOB_OUTPUT}"
        exit 1
    fi
else
    error "Failed to submit job"
    echo "${JOB_OUTPUT}"
    exit 1
fi

exit 0
