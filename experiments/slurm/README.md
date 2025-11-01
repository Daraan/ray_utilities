# SLURM Cluster Execution Guide

This directory contains SLURM submission scripts for running Ray Utilities experiments on HPC clusters.

## � Files Overview

- **`common_base.sh`**: Shared functions used by all SLURM scripts (workspace detection, argument processing, environment setup)
- **`tune_symmetric_run.sh`**: Recommended for Ray 2.49+ single-tenant clusters (uses `ray symmetric-run`)
- **`tune_multi_tenant.sh`**: For shared clusters with multiple concurrent users (custom port management)
- **`tune_with_scheduler.sh`**: Legacy-compatible script for manual Ray cluster setup
- **`CHANGES_NEEDED.md`**: Python script compatibility guide for SLURM execution
- **`slurm.rst`**: Detailed Sphinx documentation

## �🚀 Ray 2.49+ Features

Ray 2.49 introduces **`ray symmetric-run`**, which dramatically simplifies SLURM deployment:

- ✅ **Automatic cluster management**: No manual head/worker setup
- ✅ **Simplified scripts**: One `srun` command handles everything
- ✅ **Better error handling**: Clearer failure messages
- ✅ **Automatic cleanup**: Cluster stops when script completes
- ❌ **Limitation**: Does NOT work in multi-tenant environments (shared clusters)

## 📋 Quick Decision Guide

**Which script should I use?**

| Scenario | Script to Use | Ray Version |
|----------|---------------|-------------|
| Single-tenant cluster (you control all nodes) | `tune_symmetric_run.sh` | Ray 2.49+ |
| Multi-tenant cluster (shared with other users) | `tune_multi_tenant.sh` | Any version |
| Legacy compatibility or Ray < 2.49 | `tune_with_scheduler.sh` | Any version |


## ⚠️ Important: Python Script Compatibility

### Option 1: Use `ray symmetric-run` (RECOMMENDED for Ray 2.49+)

**No Python script changes needed!** The `tune_symmetric_run.sh` script uses `ray symmetric-run`, which means your Python scripts work as-is:

```python
# ✅ Works perfectly with tune_symmetric_run.sh
with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
    results = run_tune(setup)
```

The `ray symmetric-run` command handles cluster initialization automatically. Your script's `ray.init()` just connects to the cluster that's already running.

### Option 2: Manual cluster setup (for legacy scripts or multi-tenant)

For `tune_with_scheduler.sh` or `tune_multi_tenant.sh`, modify your script:

```python
# ❌ Local version - starts a new cluster
with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
    results = run_tune(setup)
```

**Change to:**
```python
# ✅ SLURM version - connects to existing cluster
with ray.init(address='auto', runtime_env=runtime_env):
    results = run_tune(setup)
```

**Or use conditional initialization:**
```python
import os
if 'SLURM_JOB_ID' in os.environ:
    # Running on SLURM - connect to existing cluster
    with ray.init(address='auto', runtime_env=runtime_env):
        results = run_tune(setup)
else:
    # Running locally - start new cluster
    with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
        results = run_tune(setup)
```

**Key insight:**
- With `ray symmetric-run`: Your script's `ray.init()` works normally
- With manual `ray start`: Use `address='auto'` or `address='head_ip:port'`
- The context manager disconnects on exit (doesn't shut down cluster)
- The SLURM script handles cluster lifecycle


## Quick Start

### Using ray symmetric-run (Ray 2.49+, Single-Tenant)

```bash
# Basic submission - your Python script works as-is!
sbatch experiments/slurm/tune_symmetric_run.sh experiments/tune_with_scheduler.py

# With custom arguments
sbatch experiments/slurm/tune_symmetric_run.sh experiments/tune_with_scheduler.py --seed 123 --iterations 2000

# Different experiment
sbatch experiments/slurm/tune_symmetric_run.sh experiments/default_training.py --env_type pendulum

# NEW: SLURM options can now be mixed with Python script arguments!
# The scripts automatically find the .py file and separate SLURM options from Python arguments
sbatch experiments/slurm/tune_symmetric_run.sh --mem=240GB --cpus-per-task=26 python experiments/tune.py --seed 42 --tune minibatch_size

# This works too (Python script anywhere in arguments):
sbatch experiments/slurm/tune_with_scheduler.sh --time=48:00:00 python experiments/tune.py --num_samples 100
```

**Note**: The scripts intelligently parse arguments to find the first `.py` file, so you can pass SLURM options (like `--mem`, `--cpus-per-task`) alongside your Python script path and arguments. All arguments before and including the `.py` file are handled as setup, and everything after becomes Python script arguments.

### Using Legacy Scripts (Any Ray Version)

```bash
# Traditional manual cluster setup
sbatch experiments/slurm/tune_with_scheduler.sh experiments/tune_with_scheduler.py --seed 123

# Multi-tenant environment with port isolation
USER_PORT_OFFSET=10000 sbatch experiments/slurm/tune_multi_tenant.sh experiments/tune.py --seed 42
```

### Environment Variables

Set these before `sbatch` to customize behavior:

| Variable | Default | Applies To | Description |
|----------|---------|------------|-------------|
| `WORKSPACE_DIR` | Auto-detected | All | Custom workspace directory (see below) |
| `RAY_PORT` | 6379 | All | Ray head node port |
| `RAY_TMPDIR` | `$SLURM_TMPDIR/ray_*` | symmetric-run, multi-tenant | Ray temporary directory |
| `RAY_OBJECT_STORE` | 10 (GB) | legacy scripts | Ray object store memory |
| `USER_PORT_OFFSET` | Job ID % 50000 | multi-tenant only | Port offset for multi-tenant isolation |

Example:
```bash
WORKSPACE_DIR=/scratch/$USER/my_workspace \
RAY_TMPDIR=/fast-scratch/ray \
sbatch experiments/slurm/tune_symmetric_run.sh experiments/tune.py --seed 42
```

### Workspace Directory Detection

All scripts use `common_base.sh` for workspace detection with the following priority:

1. **`WORKSPACE_DIR` environment variable** - Explicitly set before `sbatch`
2. **`ws_find` command** - If available (workspace management tool on some clusters)
3. **Auto-detection** - Two directories up from the script location

**Important**: `WORKSPACE_DIR` is used **only for output/storage** (results, logs, checkpoints). It is NOT used for locating Python scripts. Python script paths are resolved relative to the directory where you run `sbatch`.

Example:
```bash
# WORKSPACE_DIR is where results will be stored
WORKSPACE_DIR=/scratch/$USER/results sbatch experiments/slurm/tune_symmetric_run.sh experiments/tune.py --seed 42

# Python script location: resolved from current directory (experiments/tune.py)
# Results location: stored in /scratch/$USER/results/outputs/experiments/
```

### Automatic `--storage_path` Handling

All scripts **automatically add** `--storage_path <OUTPUT_DIR>` to your Python arguments if not already provided:

- **Without `pbt`**: Appends `--storage_path` at the end
- **With `pbt`**: Inserts `--storage_path` **before** the `pbt` argument
- **Already provided**: Uses your custom `--storage_path` unchanged

Example behavior:
```bash
# Input:  sbatch tune_symmetric_run.sh exp.py --seed 42
# Actual: python exp.py --seed 42 --storage_path /workspace/outputs/experiments

# Input:  sbatch tune_symmetric_run.sh exp.py --seed 42 pbt --lr 0.001
# Actual: python exp.py --seed 42 --storage_path /workspace/outputs/experiments pbt --lr 0.001

# Input:  sbatch tune_symmetric_run.sh exp.py --storage_path /custom/path
# Actual: python exp.py --storage_path /custom/path  # Uses your custom path
```


## Configuration

### Choosing SBATCH Parameters

All scripts use similar `#SBATCH` directives. Edit these at the top of each script:

```bash
#SBATCH --job-name=ray_tune          # Job name
#SBATCH --nodes=3                     # Number of nodes
#SBATCH --ntasks-per-node=1          # MUST be 1 for Ray
#SBATCH --cpus-per-task=24           # CPUs per node (Ray will use all)
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=24:00:00              # Max runtime
#SBATCH --partition=compute          # Partition (adjust for your cluster)
#SBATCH --gpus-per-task=1            # Optional: GPUs per node
```

**Important SBATCH Notes:**

1. **`--ntasks-per-node=1`**: REQUIRED for Ray. Do not change this.
2. **`--cpus-per-task`**: Ray sees this as available CPUs per node
3. **`--mem`**: Total memory per node (not per CPU)
4. **`--gpus-per-task`**: Use instead of `--gres=gpu:N` on some clusters
5. **`--exclusive`**: Optional, reserves entire nodes for your job

### Script-Specific Configuration

#### tune_symmetric_run.sh (Ray 2.49+)

**Advantages:**
- Simplest setup
- Automatic head/worker management
- Better error messages
- Synchronous startup (script waits for all nodes)

**Limitations:**
- Ray 2.49+ required
- Single-tenant only (doesn't work if other users run Ray simultaneously)

**Best for:** Dedicated cluster allocations, large experiments, newest Ray features

#### tune_multi_tenant.sh (Multi-User Clusters)

**Advantages:**
- Port isolation prevents conflicts
- Works with any Ray version
- Multiple users can run simultaneously

**Configuration:**
```bash
# Each user should pick a unique offset
USER_PORT_OFFSET=0     sbatch ...  # User 1
USER_PORT_OFFSET=10000 sbatch ...  # User 2
USER_PORT_OFFSET=20000 sbatch ...  # User 3
```

**Best for:** Shared clusters, production environments with multiple teams

#### tune_with_scheduler.sh (Legacy/Compatible)

**Advantages:**
- Works with older Ray versions
- Proven, stable approach
- No special requirements

**Best for:** Ray < 2.49, maximum compatibility, troubleshooting


## Key Features

### Automatic Workspace Detection

All scripts automatically detect the workspace directory:

1. Uses `WORKSPACE_DIR` environment variable if set
2. Tries `ws_find master_workspace` (if workspace management tool available)
3. Falls back to auto-detection from script location

### Ray Cluster Management

#### With `ray symmetric-run` (tune_symmetric_run.sh)

- **Single command**: `srun ray symmetric-run` handles everything
- **Automatic**: Head node selection, worker connection, resource detection
- **Synchronous**: Script only runs after all nodes join (controlled by `--min-nodes`)
- **Automatic cleanup**: Ray cluster stops when Python script completes
- **No manual ray.init modification needed**: Your scripts work as-is

Example of what happens:
```bash
# This one srun command:
srun ray symmetric-run --address $ip_head --min-nodes 3 -- python script.py

# Automatically does:
# 1. Starts Ray head on first node
# 2. Starts Ray workers on remaining nodes
# 3. Waits for all nodes to join
# 4. Runs script.py ONLY on head node
# 5. Stops cluster when script completes
```

#### With Manual Setup (tune_with_scheduler.sh, tune_multi_tenant.sh)

- **Manual head/worker**: Separate `ray start --head` and `ray start --address`
- **Port configuration**: Must specify ports in multi-tenant environments
- **Explicit cleanup**: Must call `ray stop` in cleanup section
- **Requires `address='auto'`**: Python scripts must be modified

### Python Script Execution Patterns

#### Pattern 1: Works with symmetric-run (RECOMMENDED)

```python
# No changes needed! The ray.init() here works perfectly with symmetric-run
import ray
from ray_utilities import run_tune, PPOSetup, DefaultArgumentParser

runtime_env = {...}

if __name__ == "__main__":
    with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
        with PPOSetup(config_files=["experiments/default.cfg"]) as setup:
            setup.config.training(num_epochs=10)
        results = run_tune(setup)
```

**How it works:**
- `ray symmetric-run` starts the cluster before your script runs
- Your `ray.init()` detects the existing cluster and connects
- Resource arguments (num_cpus, etc.) are ignored (cluster already configured)
- `runtime_env` is still used to propagate environment to workers

#### Pattern 2: Manual cluster connection

```python
# Required for tune_with_scheduler.sh and tune_multi_tenant.sh
import ray
from ray_utilities import run_tune, PPOSetup

runtime_env = {...}

if __name__ == "__main__":
    # Connect to pre-started cluster
    with ray.init(address='auto', runtime_env=runtime_env):
        with PPOSetup(config_files=["experiments/default.cfg"]) as setup:
            setup.config.training(num_epochs=10)
        results = run_tune(setup)
```

#### Pattern 3: Adaptive (works everywhere)

```python
import os
import ray
from ray_utilities import run_tune, PPOSetup

runtime_env = {...}

if __name__ == "__main__":
    # Auto-detect environment
    if 'SLURM_JOB_ID' in os.environ:
        # Running on SLURM - connect to existing cluster
        context = ray.init(address='auto', runtime_env=runtime_env)
    else:
        # Running locally - start new cluster
        context = ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env)

    with context:
        with PPOSetup(config_files=["experiments/default.cfg"]) as setup:
            setup.config.training(num_epochs=10)
        results = run_tune(setup)
```


## Directory Structure

```
WORKSPACE_DIR/
├── outputs/
│   ├── experiments/          # Tune results (shared across all scripts)
│   │   └── checkpoints/      # Model checkpoints
│   ├── comet_offline/        # Comet offline logs
│   ├── wandb/                # WandB offline logs
│   └── slurm_logs/           # SLURM-specific logs
└── ray_tune-<JOBID>.out      # Job stdout/stderr
```

## Script Comparison

| Feature | symmetric-run | multi-tenant | legacy | generic |
|---------|--------------|--------------|---------|---------|
| **Ray Version** | 2.49+ | Any | Any | Any |
| **Setup Complexity** | ⭐ Easy | ⭐⭐ Medium | ⭐⭐ Medium | ⭐ Easy |
| **Multi-Tenant** | ❌ No | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **Python Changes** | ✅ None | ⚠️ `address='auto'` | ⚠️ `address='auto'` | ⚠️ `address='auto'` |
| **Automatic Cleanup** | ✅ Yes | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| **Node Count** | Any | Any | Any | 1-2 |
| **Error Messages** | ✅ Clear | ⚠️ Complex | ⚠️ Complex | ✅ Clear |
| **Best For** | New projects | Shared clusters | Compatibility | Testing |

**Recommendation:** Start with `tune_symmetric_run.sh` if you have Ray 2.49+ and a dedicated allocation. Fall back to `tune_multi_tenant.sh` only if you experience port conflicts.

## Passing Custom Arguments

The first argument must be the Python script path (relative to workspace root). All subsequent arguments are passed to that script:

```bash
# Override seed and iterations
sbatch experiments/slurm/tune_with_scheduler.sh \
    experiments/tune_with_scheduler.py \
    --seed 999 --iterations 5000

# Change environment
sbatch experiments/slurm/tune_with_scheduler.sh \
    experiments/default_training.py \
    --env_type pendulum

# Multiple arguments
sbatch experiments/slurm/tune_with_scheduler.sh \
    experiments/tune_batch_size.py \
    --seed 123 \
    --iterations 3000 \
    --num_samples 2 \
    --env_type acrobot
```

## File Transfer

### Shared Filesystem (Recommended)

If your cluster uses shared storage (NFS, Lustre, GPFS), files are automatically accessible:

- All nodes write directly to `WORKSPACE_DIR/outputs`
- No manual transfer needed
- Results immediately available after job completion

### Local Node Storage

If using `SLURM_TMPDIR` or local storage:

1. The script automatically syncs from `$SLURM_TMPDIR/outputs` to `WORKSPACE_DIR/outputs`
2. Happens at job completion (see end of script)
3. Uses `rsync -avz` for efficient transfer

To customize transfer behavior, modify the cleanup section:

```bash
# Sync specific subdirectories
rsync -avz ${SLURM_TMPDIR}/outputs/experiments/ ${OUTPUT_DIR}/experiments/
rsync -avz ${SLURM_TMPDIR}/outputs/wandb/ ${WANDB_DIR}/
```

## Monitoring

### Check Job Status

```bash
# View queue
squeue -u $USER

# Job details
scontrol show job <JOBID>

# Cancel job
scancel <JOBID>
```

### Monitor Running Job

```bash
# Tail output log
tail -f ray_pbt_tune-<JOBID>.out

# Connect to head node (while job running)
srun --jobid=<JOBID> --pty bash

# Check Ray cluster status
ray status  # Run on head node
```

### Post-Job Analysis

```bash
# View full output
less ray_pbt_tune-<JOBID>.out

# Check results
ls -lh ${WORKSPACE_DIR}/outputs/experiments/

# View WandB offline logs
ls -lh ${WORKSPACE_DIR}/outputs/wandb/
```


## Common Issues and Solutions

### Ray 2.49+ Issues

#### "symmetric-run command not found"

**Error**: `ray: command not found: symmetric-run`

**Cause**: Ray version < 2.49

**Solution**:
```bash
# Check Ray version
python -c "import ray; print(ray.__version__)"

# If < 2.49, either:
# 1. Upgrade Ray
pip install --upgrade "ray[default]>=2.49"

# 2. Or use legacy script
sbatch experiments/slurm/tune_with_scheduler.sh ...
```

#### "symmetric-run not working with multiple users"

**Error**: Ports already in use, connection failures

**Cause**: `symmetric-run` doesn't support multi-tenant environments (documented limitation)

**Solution**: Use multi-tenant script instead
```bash
USER_PORT_OFFSET=10000 sbatch experiments/slurm/tune_multi_tenant.sh ...
```

### Cluster Setup Issues

#### Virtual Environment Not Found

**Error**: `WARNING: No virtual environment found`

**Solutions**:

1. **Create venv in workspace**:
```bash
cd ${WORKSPACE_DIR}
python -m venv venv
source venv/bin/activate
pip install -e .
```

2. **Edit script to point to your venv**:
```bash
# In tune_symmetric_run.sh, add your venv path:
if [ -f "/absolute/path/to/your/venv/bin/activate" ]; then
    source "/absolute/path/to/your/venv/bin/activate"
fi
```

3. **Use conda environment**:
```bash
# Add to script's environment section:
conda activate my-env
```

#### Ray Cluster Won't Start

**Error**: Ray workers fail to connect, timeout errors

**Solutions**:

1. **Check firewall**: Ensure port 6379 (and worker ports) are open between nodes
```bash
# Test connectivity
srun --nodes=1 -w <worker_node> nc -zv <head_node> 6379
```

2. **Verify network**: Check node IPs are reachable
```bash
srun --nodes=1 -w <node> hostname -I
```

3. **Increase sleep time**: Some clusters need more time for head node startup
```bash
# In tune_with_scheduler.sh, increase after ray start:
sleep 30  # Instead of sleep 10
```

4. **Check Ray logs**:
```bash
# On head node:
cat ${RAY_TMPDIR}/session_latest/logs/*
```

#### Out of Memory

**Error**: `Ray object store memory exceeded`, workers killed by OOM

**Solutions**:

1. **For symmetric-run**: Ray auto-detects memory, but you can limit with `--memory`
```bash
# Edit tune_symmetric_run.sh, add to ray symmetric-run:
--memory=$((SLURM_MEM_PER_NODE * 1024 * 1024))  # Convert MB to bytes
```

2. **For manual scripts**: Increase object store size
```bash
RAY_OBJECT_STORE=20 sbatch ...  # 20 GB instead of default 10 GB
```

3. **Or increase SBATCH allocation**:
```bash
#SBATCH --mem=64G  # Instead of 32G
```

4. **Check actual usage**:
```bash
# While job running, on head node:
ray status
```

#### Results Not Transferred

**Error**: Results missing after job completion

**Solutions**:

1. **Check if SLURM_TMPDIR exists**:
```bash
echo $SLURM_TMPDIR  # Should be something like /tmp/slurm.12345
```

2. **Verify rsync section is active**: Uncomment file transfer at end of script

3. **Check permissions**:
```bash
ls -ld ${WORKSPACE_DIR}/outputs
# Should be writable by your user
```

4. **Add explicit sync**:
```bash
# At end of script, before exit:
rsync -avz --progress ${SLURM_TMPDIR}/outputs/ ${OUTPUT_DIR}/
```

### Script Compatibility Issues

#### "Address already in use" (Multi-Tenant)

**Error**: `address already in use`, port binding failures

**Cause**: Multiple Ray clusters trying to use same ports

**Solution**: Use different port offsets
```bash
# Coordinate with other users:
USER_PORT_OFFSET=0     sbatch ...  # User A
USER_PORT_OFFSET=10000 sbatch ...  # User B
USER_PORT_OFFSET=20000 sbatch ...  # User C
```

#### Python Script Can't Connect

**Error**: `Failed to connect to Ray cluster`

**Cause**: Script using wrong Ray connection method

**Solution**: Match script to SLURM script type
```python
# For tune_symmetric_run.sh: NO CHANGES NEEDED
with ray.init(num_cpus=24, runtime_env=...):
    results = run_tune(setup)

# For tune_with_scheduler.sh or tune_multi_tenant.sh:
with ray.init(address='auto', runtime_env=...):
    results = run_tune(setup)
```

### Performance Issues

#### Slow Worker Startup

**Symptom**: Long delay before all workers join

**Solutions**:

1. **Use `--min-nodes`** (symmetric-run only): Ensures all nodes join before starting
```bash
# Already in tune_symmetric_run.sh:
--min-nodes "${SLURM_JOB_NUM_NODES}"
```

2. **Preload environment**: Create a shared venv accessible to all nodes
```bash
# Create venv on shared filesystem
python -m venv /shared/venvs/ray_utilities
```

3. **Use `runtime_env` carefully**: Large dependencies slow down workers
```python
# Instead of:
runtime_env = {"pip": ["large-package==1.0.0"]}

# Pre-install in venv:
# pip install large-package==1.0.0
runtime_env = {}  # Empty if everything is pre-installed
```

#### Slow Experiment Startup

**Symptom**: Long time before first iteration

**Cause**: Checkpoint restoration, dataset loading, compilation

**Solutions**:

1. **Use fast local storage** for Ray temp dir:
```bash
RAY_TMPDIR=/fast-scratch/ray sbatch ...
```

2. **Parallelize initialization**: Use Ray for data loading
```python
@ray.remote
def load_data():
    return expensive_operation()

# Parallel loading
futures = [load_data.remote() for _ in range(num_workers)]
data = ray.get(futures)
```

## Customization Examples

### Run on Single Node

```bash
# Edit SBATCH directives
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

# Script will detect and skip worker node setup
```

### Load Specific Modules

```bash
# Uncomment and edit module loading section (line ~90)
module load python/3.10
module load gcc/11.2.0
module load cuda/11.8
```

### Use Different Python Script

Create a copy and modify the embedded Python code (line ~205):

```bash
# Change to your script
cat > "${TEMP_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
import ray
from ray_utilities import run_tune
# ... your custom experiment ...
EOF
```


## Best Practices

### 1. Choose the Right Script

- **Ray 2.49+ on dedicated cluster**: Use `tune_symmetric_run.sh` (simplest)
- **Shared cluster (multi-tenant)**: Use `tune_multi_tenant.sh` (port isolation)
- **Maximum compatibility**: Use `tune_with_scheduler.sh` (works everywhere)

### 2. Resource Allocation

#### Start Small, Scale Up
```bash
# Test run: 1 node, short time, few samples
#SBATCH --nodes=1
#SBATCH --time=01:00:00
sbatch tune_symmetric_run.sh experiments/tune.py --num_samples 2 --iterations 100

# Production run: scale up after testing
#SBATCH --nodes=10
#SBATCH --time=24:00:00
sbatch tune_symmetric_run.sh experiments/tune.py --num_samples 50 --iterations 5000
```

#### Right-Size Your Allocation
```bash
# Don't over-request (wastes allocation)
#SBATCH --cpus-per-task=8   # If you only use 4 CPUs
#SBATCH --mem=16G            # If you only use 8 GB

# Don't under-request (job fails)
#SBATCH --cpus-per-task=4   # If you need 8 CPUs
#SBATCH --mem=8G             # If you need 16 GB
```

#### Monitor Resource Usage
```bash
# While job is running
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %C %m"

# After job completion, check efficiency
seff ${SLURM_JOB_ID}
```

### 3. Testing Strategy

#### Local Testing First
```bash
# Test locally before SLURM submission
cd ${WORKSPACE_DIR}
python experiments/tune_with_scheduler.py --num_samples 1 --iterations 10
```

#### Incremental SLURM Testing
```bash
# 1. Test single node
#SBATCH --nodes=1
#SBATCH --time=00:30:00
sbatch tune_symmetric_run.sh experiments/tune.py --num_samples 1

# 2. Test multi-node with short runtime
#SBATCH --nodes=3
#SBATCH --time=01:00:00
sbatch tune_symmetric_run.sh experiments/tune.py --num_samples 3 --iterations 100

# 3. Full production run
#SBATCH --nodes=10
#SBATCH --time=24:00:00
sbatch tune_symmetric_run.sh experiments/tune.py --num_samples 50 --iterations 5000
```

### 4. Logging and Monitoring

#### Use Offline Logging
```python
# In your experiment script, use offline mode to avoid network issues
with DefaultArgumentParser.patch_args("--wandb", "offline+upload"):
    with PPOSetup(config_files=["experiments/default.cfg"]) as setup:
        pass
    results = run_tune(setup)

# After job completes, upload logs
python upload_wandb.py ${WORKSPACE_DIR}/outputs/wandb
```

#### Monitor Job Progress
```bash
# Watch job queue
watch -n 30 squeue -u $USER

# Tail output log (while job running)
tail -f ray_tune-${SLURM_JOB_ID}.out

# Check Ray cluster status (on head node)
srun --jobid=${SLURM_JOB_ID} --pty bash
ray status

# Watch results directory
watch -n 60 "ls -lh ${WORKSPACE_DIR}/outputs/experiments/"
```

### 5. Error Handling

#### Capture Complete Logs
```bash
# In your SBATCH script, ensure stdout/stderr capture
#SBATCH --output=%x-%j.out    # %x = job name, %j = job ID
#SBATCH --error=%x-%j.err

# Combine stdout/stderr (if preferred)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
```

#### Set Time Limits Appropriately
```bash
# Always specify --time to prevent infinite running
#SBATCH --time=24:00:00

# Add cleanup logic before timeout (in script)
# Ray handles this automatically with symmetric-run
```

#### Handle Checkpointing
```python
# Ray Tune automatically checkpoints
# Ensure TUNE_RESULT_DIR is on shared filesystem
# Set storage_path="<WORKSPACE_DIR>/outputs/experiments" in the RunConfig of a tuner

# Resume interrupted experiments
python experiments/tune.py --from_checkpoint ${WORKSPACE_DIR}/outputs/experiments/checkpoint_000100
```

### 6. Filesystem Considerations

#### Use Shared Filesystem for Outputs
```bash
# Outputs should be on shared storage (NFS, Lustre, GPFS)
WORKSPACE_DIR=/shared/nfs/${USER}/ray_utilities
sbatch tune_symmetric_run.sh experiments/tune.py

# NOT local node storage (unless you plan to transfer)
WORKSPACE_DIR=/tmp  # ❌ Wrong: Not accessible from other nodes
```

#### Fast Local Storage for Temporary Files
```bash
# Use fast local storage for Ray's temporary files
RAY_TMPDIR=/fast-scratch/${USER}/ray sbatch tune_symmetric_run.sh ...

# Or in script:
export RAY_TMPDIR="/nvme/tmp/ray_${SLURM_JOB_ID}"
```

### 7. Multi-Tenant Clusters

#### Coordinate Port Usage
```bash
# Create a simple coordination system
# Option 1: Use job ID
USER_PORT_OFFSET=$((SLURM_JOB_ID % 50000))

# Option 2: Assign ranges per team/user
# Team A: 0-9999
# Team B: 10000-19999
# Team C: 20000-29999
USER_PORT_OFFSET=10000 sbatch tune_multi_tenant.sh ...
```

#### Avoid Peak Times
```bash
# Check cluster load before submitting large jobs
sinfo -o "%P %.5a %.10l %.6D %.6t %N"

# Submit during off-peak hours
#SBATCH --begin=20:00:00  # Start at 8 PM
#SBATCH --time=08:00:00   # Run overnight
```

### 8. Advanced: Parameter Sweeps

#### Job Arrays
```bash
# Run multiple experiments with different seeds
#SBATCH --array=1-10

sbatch tune_symmetric_run.sh experiments/tune.py --seed ${SLURM_ARRAY_TASK_ID}
```

#### Scripted Submissions
```bash
#!/bin/bash
# sweep_experiments.sh
for env_type in pendulum acrobot cartpole; do
    for seed in 42 123 456 789; do
        sbatch tune_symmetric_run.sh experiments/tune.py \
            --env_type ${env_type} \
            --seed ${seed}
    done
done
```

### 9. Debugging

#### Interactive Sessions
```bash
# Allocate interactive node
salloc --nodes=1 --ntasks=1 --cpus-per-task=24 --time=02:00:00

# Once allocated, manually run commands
source ${WORKSPACE_DIR}/venv/bin/activate
cd ${WORKSPACE_DIR}
python experiments/tune.py --num_samples 1
```

#### Verbose Logging
```bash
# Enable Ray debug logging
export RAY_BACKEND_LOG_LEVEL=debug

# Enable Python unbuffered output
python -u experiments/tune.py
```

#### Check Specific Node
```bash
# SSH to specific node (if allowed)
srun --jobid=${SLURM_JOB_ID} -w <node_name> --pty bash

# Check processes
ps aux | grep ray

# Check logs
ls -la ${RAY_TMPDIR}/session_latest/logs/
```


## Advanced Topics

### Multi-Job Parameter Sweeps

Run parameter sweeps across multiple SLURM jobs:

```bash
#!/bin/bash
# sweep_experiments.sh
for env_type in pendulum acrobot cartpole; do
    for seed in 42 123 456 789; do
        sbatch tune_symmetric_run.sh experiments/tune.py \
            --env_type ${env_type} \
            --seed ${seed}
    done
done
```

Or use SLURM job arrays:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name=sweep_ray

SEEDS=(42 123 456 789 999 111 222 333 444 555)
sbatch tune_symmetric_run.sh experiments/tune.py --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```

### Custom Runtime Environments

Propagate complex environments to all Ray workers:

```python
# In your experiment script
runtime_env = {
    "pip": ["custom-package==1.0.0"],
    "env_vars": {
        "MY_CONFIG": "value",
        "CUDA_VISIBLE_DEVICES": "0,1"
    },
    "working_dir": ".",  # Sync current directory
}

with ray.init(runtime_env=runtime_env):
    results = run_tune(setup)
```

### Network Interface Selection

Some clusters have multiple network interfaces. Specify which to use:

```bash
# In tune_symmetric_run.sh, modify head IP detection:
# Instead of:
# ip_head="${head_node}:${RAY_PORT}"

# Use specific interface (e.g., InfiniBand):
head_ip=$(srun --nodes=1 -w "${head_node}" ip addr show ib0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
ip_head="${head_ip}:${RAY_PORT}"
```

### Integration with Experiment Tracking

#### WandB Offline + Upload Pattern

```python
# experiments/tune_with_wandb.py
from ray_utilities import run_tune, PPOSetup, DefaultArgumentParser

if __name__ == "__main__":
    # Use offline mode on SLURM to avoid network issues
    with DefaultArgumentParser.patch_args("--wandb", "offline+upload"):
        with PPOSetup(config_files=["experiments/default.cfg"]) as setup:
            pass
        results = run_tune(setup)
```

Then after job completes:
```bash
# Upload offline logs
python upload_wandb.py ${WORKSPACE_DIR}/outputs/wandb
```

### Ray Dashboard Access

Access Ray dashboard from your local machine:

```bash
# While job is running, set up SSH tunnel
ssh -N -L 8265:${HEAD_NODE}:8265 ${CLUSTER_LOGIN_NODE}

# Open in browser:
# http://localhost:8265
```

### GPU Configuration

For GPU-accelerated experiments:

```bash
#SBATCH --gpus-per-task=2  # or --gres=gpu:2 on some clusters

# In Python script, Ray automatically detects GPUs
# Or explicitly set:
config.resources(num_gpus_per_learner_worker=1)
```

### Preemptible Jobs

Handle job preemption gracefully:

# TODO: restore_path and Tuner restore is NOT YET IMPLEMENTED

```bash
# Use Ray Tune's built-in checkpointing
# Experiments automatically resume from last checkpoint

# Resubmit with --restore_path
sbatch tune_symmetric_run.sh experiments/tune.py \
    --restore_path ${WORKSPACE_DIR}/outputs/experiments/PPO_*/checkpoint_*
```

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] Ray version ≥ 2.49 (for symmetric-run)
- [ ] Virtual environment is activated and has ray installed
- [ ] Python script exists at specified path
- [ ] SBATCH directives are appropriate for your cluster
- [ ] `--ntasks-per-node=1` is set (required for Ray)
- [ ] Port 6379 (or custom port) is accessible between nodes
- [ ] Shared filesystem is mounted on all nodes
- [ ] Sufficient disk space in `WORKSPACE_DIR` and `RAY_TMPDIR`
- [ ] Job hasn't exceeded time limit
- [ ] Memory allocation is sufficient

Check these logs:
1. SLURM output: `ray_tune-${SLURM_JOB_ID}.out`
2. Ray logs: `${RAY_TMPDIR}/session_latest/logs/`
3. Experiment results: `${WORKSPACE_DIR}/outputs/experiments/`

## Support

For Ray-specific issues:
- Ray documentation: https://docs.ray.io/
- Ray Tune guide: https://docs.ray.io/en/latest/tune/

For SLURM issues:
- Check your cluster documentation
- Contact cluster administrators
- SLURM docs: https://slurm.schedmd.com/
