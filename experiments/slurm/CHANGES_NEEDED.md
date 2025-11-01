# Python Scripts for SLURM Execution - Quick Guide

## Summary: Do You Need to Change Your Scripts?

| SLURM Script | Python Changes Needed? | Ray Version |
|--------------|------------------------|-------------|
| `tune_symmetric_run.sh` | ✅ **NO** - Works as-is! | 2.49+ |
| `tune_multi_tenant.sh` | ⚠️ **YES** - Use `address='auto'` | Any |
| `tune_with_scheduler.sh` | ⚠️ **YES** - Use `address='auto'` | Any |

## Option 1: Use `tune_symmetric_run.sh` (RECOMMENDED for Ray 2.49+)

**NO CHANGES NEEDED!** Your existing scripts work perfectly:

```python
# experiments/tune_with_scheduler.py - NO CHANGES REQUIRED
with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
    results = run_tune(setup)
```

**How it works:**
- `ray symmetric-run` starts the cluster before your script runs
- Your `ray.init()` automatically detects and connects to it
- Resource arguments (`num_cpus`, etc.) are ignored (cluster already configured)
- `runtime_env` still propagates to workers

**Submit with:**
```bash
sbatch experiments/slurm/tune_symmetric_run.sh experiments/tune_with_scheduler.py --seed 42
```

## Option 2: Use Multi-Tenant or Legacy Scripts

For `tune_multi_tenant.sh` or `tune_with_scheduler.sh`, you need to modify your scripts:

### Current Code (Local Only)

```python
# ❌ Won't work on SLURM with manual cluster setup
with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
    results = run_tune(setup)
```

### Option 2A: Simple SLURM-Only Version

```python
# ✅ For SLURM only
with ray.init(address='auto', runtime_env=runtime_env):
    results = run_tune(setup)
```

**Submit with:**
```bash
sbatch experiments/slurm/tune_with_scheduler.sh experiments/tune_with_scheduler.py --seed 42
# or
USER_PORT_OFFSET=10000 sbatch experiments/slurm/tune_multi_tenant.sh experiments/tune.py
```

### Option 2B: Conditional (Works Both Local and SLURM)

```python
# ✅ Works everywhere
import os

if 'SLURM_JOB_ID' in os.environ:
    # Running on SLURM cluster
    with ray.init(address='auto', runtime_env=runtime_env):
        results = run_tune(setup)
else:
    # Running locally
    with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
        results = run_tune(setup)
```

### 2. Files That Need Modification

Based on the `experiments/` directory, these files need to be updated:

- `experiments/tune_with_scheduler.py` ⚠️ **Priority**
- `experiments/default_training.py`
- `experiments/tune_batch_size.py`
- `experiments/dynamic_batch_size.py`
- `experiments/dynamic_batch_and_buffer.py`
- `experiments/dynamic_buffer.py`
- `experiments/scheduler_experiments.py`

### 3. Key Points

**What to change:**
- Replace `ray.init(num_cpus=X, object_store_memory=Y, ...)` with `ray.init(address='auto', ...)`
- Keep the `with` context manager (recommended for automatic cleanup)
- Keep `runtime_env` parameter

**What NOT to change:**
- ✅ **Keep** `with` statement - it's the recommended pattern
- ✅ **Keep** `runtime_env` - needed for environment variables
- ❌ **Remove** `num_cpus` - controlled by SLURM
- ❌ **Remove** `object_store_memory` - controlled by SLURM

**How `with ray.init()` works:**
```python
with ray.init(address='auto', runtime_env=runtime_env):
    results = run_tune(setup)
# __exit__ calls ray.shutdown(), which:
#   - Disconnects this client from the cluster
#   - Does NOT shut down the cluster (SLURM manages that)
```

## Why This Is Necessary

1. **Resource Management**: SLURM allocates resources (CPUs, memory) via SBATCH directives
2. **Cluster Lifecycle**: The bash script manages Ray cluster start/stop
3. **Multi-node Support**: Workers are started on other SLURM nodes; your script connects as a client
4. **Conflict Prevention**: Starting a new cluster when one already exists will fail

**Important:** `ray.shutdown()` behaves differently based on how Ray was initialized:
- **If you connected** to an existing cluster (`address='auto'`): Only disconnects the client
- **If you started** a new cluster (no address): Shuts down the entire cluster
- The SLURM script starts the cluster, so your script just connects and disconnects

## Testing Your Changes

### Test Locally First

Your script should still work locally with the conditional approach:

```bash
# Should work with conditional initialization
python experiments/tune_with_scheduler.py --seed 42 --iterations 10
```

### Test on SLURM

```bash
# Basic test (short run)
sbatch experiments/slurm/tune_with_scheduler.sh \
    experiments/tune_with_scheduler.py \
    --seed 42 --iterations 10

# Full experiment
sbatch experiments/slurm/tune_with_scheduler.sh \
    experiments/tune_with_scheduler.py \
    --seed 42 --iterations 2000
```

## Quick Reference: Before vs After

### Before (Local Only)
```python
if __name__ == "__main__":
    with DefaultArgumentParser.patch_args(...):
        setup = PPOMLPWithPBTSetup(...)
        with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
            results = run_tune(setup)
```

### After (SLURM - Recommended)
```python
if __name__ == "__main__":
    with DefaultArgumentParser.patch_args(...):
        setup = PPOMLPWithPBTSetup(...)
        with ray.init(address='auto', runtime_env=runtime_env):
            results = run_tune(setup)
```

### After (Both Local + SLURM - Conditional)
```python
if __name__ == "__main__":
    with DefaultArgumentParser.patch_args(...):
        setup = PPOMLPWithPBTSetup(...)

        # Detect SLURM environment
        if 'SLURM_JOB_ID' in os.environ:
            with ray.init(address='auto', runtime_env=runtime_env):
                results = run_tune(setup)
        else:
            with ray.init(num_cpus=24, object_store_memory=3 * 1024**3, runtime_env=runtime_env):
                results = run_tune(setup)
```

## Alternative: Create SLURM-Specific Versions

If you don't want to modify the original files, create SLURM-specific copies:

```bash
# Copy and modify
cp experiments/tune_with_scheduler.py experiments/tune_with_scheduler_slurm.py

# Edit tune_with_scheduler_slurm.py to use ray.init(address='auto')

# Submit SLURM version
sbatch experiments/slurm/tune_with_scheduler.sh \
    experiments/tune_with_scheduler_slurm.py \
    --seed 42
```

## Need Help?

If you encounter issues:

1. Check the SLURM output log: `ray_pbt_tune-<JOBID>.out`
2. Look for Ray connection errors
3. Verify Ray cluster is running: `ray status` (on head node)
4. Check if script is using correct `ray.init()` call
