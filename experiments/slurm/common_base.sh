#!/bin/bash
################################################################################
# Common Base Script for SLURM Ray Utilities
#
# This script contains shared logic used by all SLURM submission scripts.
# Source this file in your SLURM scripts: source "$(dirname "$0")/common_base.sh"
#
# Functions provided:
#   - detect_workspace_dir()
#   - parse_python_script_from_args()  # Parse Python script from mixed args
#   - setup_python_args()
#   - setup_environment_vars(entry_point)  # Requires entry point path as argument
#   - activate_virtualenv()
################################################################################

# ============================================================================
# Workspace Detection
# ============================================================================

detect_workspace_dir() {
    # Detect workspace directory (project root)
    # Priority: 1) WORKSPACE_DIR env var, 2) ws_find command, 3) auto-detect from script location
    if [ -z "${WORKSPACE_OFFLINE_DIR:-}" ]; then
        # Try ws_find if available (workspace management tool)
        if command -v ws_find >/dev/null 2>&1; then
            WORKSPACE_OFFLINE_DIR=$(ws_find master_workspace 2>/dev/null || echo "")
        fi

        # Fallback to auto-detection from script location (two levels up)
        if [ -z "${WORKSPACE_OFFLINE_DIR:-}" ]; then
            local script_dir="$( cd "$( dirname "${BASH_SOURCE[1]}" )" && pwd )"
            WORKSPACE_OFFLINE_DIR="$(cd "$script_dir/../.." && pwd)"
        fi
    fi
    export WORKSPACE_OFFLINE_DIR
    if [ -z "${WORKSPACE_DIR:-}" ]; then
        export WORKSPACE_DIR="${WORKSPACE_OFFLINE_DIR}"
    fi
}

# ============================================================================
# Python Arguments Processing
# ============================================================================

parse_python_script_from_args() {
    # Find the first .py file in arguments and separate it from other args
    # This handles cases where SLURM options are mixed with Python script path
    #
    # Args: $@ - all command line arguments (may include SLURM options)
    # Sets: PYTHON_SCRIPT - path to the Python script
    #       PYTHON_SCRIPT_INDEX - index of Python script in arguments (for caller to shift)
    #
    # Usage in calling script:
    #   parse_python_script_from_args "$@"
    #   shift "${PYTHON_SCRIPT_INDEX}"
    #   # Now PYTHON_SCRIPT is set and $@ contains only Python script arguments

    if [ $# -eq 0 ]; then
        echo "ERROR: No arguments provided"
        echo "Usage: parse_python_script_from_args <PYTHON_FILE> [PYTHON_ARGS...]"
        return 1
    fi

    # Find the first argument that ends with .py (the Python script)
    PYTHON_SCRIPT=""
    PYTHON_SCRIPT_INDEX=0
    local current_index=0

    for arg in "$@"; do
        current_index=$((current_index + 1))
        if [[ "$arg" == *.py ]]; then
            PYTHON_SCRIPT="$arg"
            PYTHON_SCRIPT_INDEX=$current_index
            break
        fi
    done

    if [ -z "${PYTHON_SCRIPT:-}" ]; then
        echo "ERROR: No Python script (.py) found in arguments"
        echo "Provided arguments: $@"
        return 1
    fi

    export PYTHON_SCRIPT
    export PYTHON_SCRIPT_INDEX
}

setup_python_args() {
    # Add --storage_path OUTPUT_DIR if not already provided
    # If "pbt" is in arguments, insert --storage_path before it
    # Otherwise, append --storage_path at the end
    #
    # Args: $@ - all Python script arguments
    # Sets: PYTHON_ARGS array

    local args=("$@")
    local has_storage_path=0
    local pbt_index=-1

    # Check if --storage_path already exists and find pbt position
    for i in "${!args[@]}"; do
        if [[ "${args[$i]}" == "--storage_path" ]] || [[ "${args[$i]}" == "--storage-path" ]]; then
            has_storage_path=1
        fi
        if [[ "${args[$i]}" == "pbt" ]]; then
            pbt_index=$i
        fi
    done

    # Build final arguments array
    if [ $has_storage_path -eq 0 ]; then
        if [ $pbt_index -ge 0 ]; then
            # Insert --storage_path before pbt
            PYTHON_ARGS=("${args[@]:0:$pbt_index}" "--storage_path" "${OUTPUT_DIR}" "${args[@]:$pbt_index}")
        else
            # Append --storage_path at the end
            PYTHON_ARGS=("${args[@]}" "--storage_path" "${OUTPUT_DIR}")
        fi
    else
        # Use arguments as-is
        PYTHON_ARGS=("${args[@]}")
    fi

    export PYTHON_ARGS
}

# ============================================================================
# Environment Setup
# ============================================================================


setup_environment_vars() {
    # Setup environment variables for Ray and experiment tracking
    # Args: $1 - entry point (Python script path)

    if [ $# -eq 0 ]; then
        echo "ERROR: setup_environment_vars requires entry point as argument"
        echo "Usage: setup_environment_vars \"\${PYTHON_SCRIPT}\""
        exit 1
    fi

    # Detect workspace directory using common function
    detect_workspace_dir

    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${SLURM_SUBMIT_DIR}"
    export RAY_UTILITIES_NO_TQDM=1

    # Set common environment variables for Ray workers
    export RAY_DEDUP_LOGS_ALLOW_REGEX="COMET|wandb"
    export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID}"

    export RAY_UTILITIES_NEW_LOG_FORMAT="1"

    # Output directories (needed for storage path)

    # add submit dir to PYTHONPATH, use empty if not set
    # NOTE: WORKSPACE_DIR could be a remote S3 path

    export OUTPUT_DIR="${WORKSPACE_DIR}/outputs/experiments"
    export USE_BACKUP_DUMP_DIR=1
    export BACKUP_DUMP_DIR="${TEMP_BACKUP_DUMP_DIR:-${WORKSPACE_DIR}/ray_job_backup}/"

    # Can be S3 path
    export RAY_UTILITIES_STORAGE_PATH="${OUTPUT_DIR}"

    # Export the starting time in seconds since epoch
    export RAY_UTILITIES_INITIALIZATION_TIMESTAMP="$(date +%s)"
    # Export the starting time in formatted string: _%Y-%m-%d_%H-%M-%S
    export RAY_UTILITIES_INITIALIZATION_TIMESTAMP_STR_LONG="$(date +"_%Y-%m-%d_%H-%M-%S")"
    # Export the starting time in short format: %y%m%d%H%M
    export RAY_UTILITIES_INITIALIZATION_TIMESTAMP_STR_SHORT="$(date +"%y%m%d%H%M")"

    # RUN_ID
    export ENTRY_POINT="$1"
    export ENTRY_POINT_ID=$(python -c "import os, hashlib; print(hashlib.blake2b(os.path.basename(os.environ['ENTRY_POINT']).encode(), digest_size=3, usedforsecurity=False).hexdigest())")
    # RUN_ID structure: [ENTRY_POINT_ID][TIMESTAMP_SHORT][RANDOM_HASH][3]
    # - ENTRY_POINT_ID: 6 hex chars, hash of entry point filename
    # - TIMESTAMP_SHORT: yymmddHHMM (10 digits)
    # - RANDOM_HASH: 4 hex chars, random + entry point hash
    # - '3': static suffix for versioning/uniqueness
    export RUN_ID=$(python -c "import os, hashlib, time; print(os.environ['ENTRY_POINT_ID'] + os.environ['RAY_UTILITIES_INITIALIZATION_TIMESTAMP_STR_SHORT'] + hashlib.blake2b(os.urandom(8) + os.environ['ENTRY_POINT_ID'].encode(), digest_size=2, usedforsecurity=False).hexdigest() + '3')")

    # Set comet logdir
    export COMET_OFFLINE_DIRECTORY_BASE="${WORKSPACE_OFFLINE_DIR}/outputs/.cometml-runs"
    local entry_point_base="$(basename "${ENTRY_POINT%.*}")"
    if [ -z "$entry_point_base" ]; then
        entry_point_base="unknown"
    fi
    export COMET_OFFLINE_DIRECTORY="${COMET_OFFLINE_DIRECTORY_BASE}/${entry_point_base}${RAY_UTILITIES_INITIALIZATION_TIMESTAMP_STR_LONG}_${RUN_ID}"
    export RAY_UTILITIES_SET_COMET_DIR=0

    # Create directories
    mkdir -p "${COMET_OFFLINE_DIRECTORY_BASE}"
}


activate_virtualenv() {
    # Activate virtual environment (tries multiple common locations)
    if [ -f "../env/bin/activate" ]; then
        echo "Activating: ../env/bin/activate"
        source "../env/bin/activate"
    elif [ -f "./venv/bin/activate" ]; then
        echo "Activating: ./venv/bin/activate"
        source "./venv/bin/activate"
    elif [ -f "${HOME}/.virtualenvs/ray_utilities/bin/activate" ]; then
        echo "Activating: ${HOME}/.virtualenvs/ray_utilities/bin/activate"
        source "${HOME}/.virtualenvs/ray_utilities/bin/activate"
    else
        echo "WARNING: No virtual environment found. Using system Python."
    fi
}

# ============================================================================
# Utility Functions
# ============================================================================

print_python_info() {
    echo "Python: $(which python) ($(python --version 2>&1))"
    local ray_version=$(python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo 'NOT INSTALLED')
    echo "Ray:    ${ray_version}"
}
