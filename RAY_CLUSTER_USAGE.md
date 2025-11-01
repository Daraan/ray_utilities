# Ray Cluster Usage Guide

This document describes how to use the persistent Ray head node architecture with SLURM.

## Overview

The Ray cluster setup consists of:
1. **Head Node** (`launch_head.sh`) - Persistent Ray cluster coordinator
2. **Client Jobs** (`launch_client.sh`) - Worker jobs that connect to the head and run experiments
3. **Connection Utility** (`connect_to_head.sh`) - Helper script to simplify client submission

## Quick Start

### 1. Start a Ray Head Node

```bash
# Start with default settings
sbatch launch_head.sh

# Or with custom settings
USER_PORT_OFFSET=10000 RAY_OBJECT_STORE=20 sbatch launch_head.sh
```

Note the job ID returned by SLURM (e.g., `Submitted batch job 12345`).

### 2. Connect Client Jobs

Use the `connect_to_head.sh` utility script:

```bash
# Basic usage
./connect_to_head.sh 12345 experiments/tune.py --seed 42

# With SLURM options (before .py file)
./connect_to_head.sh 12345 --time=4:00:00 --mem=50G experiments/tune.py --seed 42

# With environment variables
RAY_OBJECT_STORE=15 ./connect_to_head.sh 12345 experiments/tune.py --arg1 value1
```

**Argument Order:**
```
connect_to_head.sh HEAD_JOB_ID [SBATCH_OPTIONS...] PYTHON_FILE.py [PYTHON_ARGS...]
                   ^           ^                    ^              ^
                   |           |                    |              |
                   Required    Optional             Required       Optional
                               (before .py)                        (after .py)
```

### 3. Monitor Jobs

```bash
# Check running jobs
squeue -u $USER

# Monitor head node
tail -f outputs/slurm_logs/12345-ray-head-*.out

# Monitor client job
tail -f outputs/slurm_logs/67890-ray-client-*.out

# Check Ray cluster status
export RAY_ADDRESS=<IP>:<PORT>  # From connection file
ray status
```

## Architecture Details

### Head Node (`launch_head.sh`)

- Runs persistently (default: 24 hours)
- Saves connection info to `outputs/ray_head_<JOB_ID>.info`
- Monitors for idle periods (2:30am-9am) and shuts down if no clients connected
- Forwards timeout signals to connected clients for graceful shutdown

**Connection Info File:**
```bash
RAY_ADDRESS=10.0.0.1:6379
RAY_PORT=6379
HEAD_NODE=dws-01
HEAD_IP=10.0.0.1
SLURM_JOB_ID=12345
USER_PORT_OFFSET=0
STARTED=2025-10-28T10:30:00+00:00
```

### Client Jobs (`launch_client.sh`)

- Connect to existing head node as workers
- Read connection info from `outputs/ray_head_<HEAD_JOB_ID>.info`
- Validate head node is running before starting
- Disconnect cleanly on completion

**Environment Variables Required:**
- `RAY_HEAD_JOB_ID` - Set automatically by `connect_to_head.sh`

### Connection Utility (`connect_to_head.sh`)

Simplifies client job submission:
1. Validates head node is running
2. Reads and validates connection file
3. Submits client job with correct parameters

**Features:**
- Color-coded output (errors, warnings, success)
- Comprehensive validation
- Helpful error messages
- Automatic detection of workspace directory

## Error Handling Improvements

### In `launch_head.sh`:
1. **Robust log file matching** - Uses glob patterns safely to avoid errors when no files exist
2. **Validated Ray startup** - Fails fast if Ray doesn't start correctly
3. **Timeout handling** - Waits up to 400 seconds for clients to finish before force shutdown
4. **Empty job list handling** - Gracefully handles case when no client jobs are found

### In `launch_client.sh`:
1. **Connection validation** - Verifies head node is running and accessible
2. **File existence checks** - Validates Python script exists before submission
3. **Network testing** - Pings head node (with warning if ICMP blocked)
4. **Connection info validation** - Ensures all required variables are present

## Port Configuration

Ports are automatically configured to avoid conflicts:

```bash
# Port calculation (based on USER_PORT_OFFSET or dws-## node)
RAY_PORT = 6379 + (USER_PORT_OFFSET / 40)
MIN_WORKER_PORT = 11000 + USER_PORT_OFFSET
MAX_WORKER_PORT = 13411 + USER_PORT_OFFSET
...
```

**Special handling for `dws-##` nodes:**
- Detects node name pattern `dws-01` through `dws-17`
- Auto-calculates offset: `USER_PORT_OFFSET = NODE_NUM * 2420`

## Common Issues

### Connection file not found
```bash
ERROR: Connection file not found: outputs/ray_head_12345.info
```
**Solution:** Head node may still be initializing. Wait 10-20 seconds and retry.

### Head node not running
```bash
ERROR: Ray head node job 12345 is not running
```
**Solution:** Check job status with `squeue -j 12345`. If not running, start a new head node.

### Cannot connect to Ray cluster
```bash
ERROR: Failed to connect to Ray cluster
```
**Solution:**
1. Verify head node is accessible: `ping <HEAD_IP>`
2. Check head node logs for errors
3. Ensure firewall rules allow connections

### Python script not found
```bash
ERROR: Python script not found: experiments/tune.py
```
**Solution:** Use path relative to where you run `connect_to_head.sh`, or use absolute path.

## Best Practices

1. **Use `connect_to_head.sh`** instead of directly calling `sbatch launch_client.sh`
2. **Monitor head node idle time** to avoid wasting resources
3. **Set appropriate SLURM options** (time, memory) based on your experiment
4. **Check connection file** exists before submitting many client jobs
5. **Clean up** completed experiments to free cluster resources

## Examples

### Running multiple experiments on same head node
```bash
# Start head node once
sbatch launch_head.sh
# Job ID: 12345

# Submit multiple experiments
./connect_to_head.sh 12345 experiments/tune.py --seed 1
./connect_to_head.sh 12345 experiments/tune.py --seed 2
./connect_to_head.sh 12345 experiments/tune.py --seed 3

# All use the same Ray cluster
```

### Long-running experiments with custom resources
```bash
# Start head with more memory
sbatch --mem=200G launch_head.sh
# Job ID: 12345

# Submit client with long time limit and GPU
./connect_to_head.sh 12345 --time=12:00:00 --gres=gpu:1 experiments/tune.py
```

### Multiple users sharing cluster
```bash
# User A
USER_PORT_OFFSET=0 sbatch launch_head.sh

# User B (different ports)
USER_PORT_OFFSET=10000 sbatch launch_head.sh

# Each user connects to their own head node
```

## Files Created

```
outputs/
├── ray_head_<JOB_ID>.info          # Connection info for each head node
├── experiments/                     # Experiment results
│   └── <trial_dirs>/
├── slurm_logs/                      # SLURM job logs
│   ├── <JOB_ID>-ray-head-*.out
│   └── <JOB_ID>-ray-client-*.out
└── experiment_overview.csv          # Summary of all experiments
```

## Troubleshooting

Enable debug output:
```bash
# In launch_head.sh or launch_client.sh, add:
set -x  # Print commands as they execute
```

Check Ray logs:
```bash
# On head node
ls -la /tmp/ray_head_<JOB_ID>/session_latest/logs/

# On client node
ls -la /tmp/ray_client_<JOB_ID>/session_latest/logs/
```

Manual connection test:
```bash
# Export connection info
source outputs/ray_head_12345.info

# Test connection
ray status

# Check cluster state
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
```
