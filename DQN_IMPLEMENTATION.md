# DQN Implementation Summary

## Overview
Successfully added DQN (Deep Q-Network) support alongside existing PPO implementation in the ray_utilities framework.

## Changes Made

### 1. Argument Parser (`config/parser/default_argument_parser.py`)
- **Added `--algorithm` CLI argument** to `_DefaultSetupArgumentParser`:
  - Choices: `'ppo'` (default) | `'dqn'`
  - Type: `AlwaysRestore` (persists in checkpoints)
  - Short form: `-algo`

- **Refactored parser classes** (Latest Change):
  - Created `_BaseRLlibArgumentParser`: Common parameters for all RLlib algorithms (`train_batch_size_per_learner`, `lr`)
  - Created `PPOArgumentParser`: PPO-specific parameters (`minibatch_size`, `num_epochs`) with validation logic
  - Created `DQNArgumentParser`: DQN-specific parameters (`target_network_update_freq`, `tau`, `epsilon`, `double_q`, `dueling`)
  - Refactored `RLlibArgumentParser`: Now inherits from both `PPOArgumentParser` and `DQNArgumentParser` using multiple inheritance
    - Eliminates code duplication
    - Automatically includes all parameters from both algorithm parsers
    - Maintains backward compatibility

- **Algorithm-specific validations**:
  - PPO: minibatch_size warnings and adjustments
  - DQN: No specific validations currently (replay buffer handled by RLlib)

### 2. Algorithm Setup (`setup/algorithm_setup.py`)
- **Added imports** for DQN classes
- **Created `_get_algorithm_classes()` classmethod**:
  - Returns `(DQNConfig, DQN)` when `args.algorithm == "dqn"`
  - Returns `(PPOConfig, PPO)` otherwise
  
- **Updated `_config_from_args()`**:
  - Dynamically selects config class based on algorithm
  - Only applies gradient accumulation learner for PPO
  
- **Created `DQNSetup` class**:
  - Similar structure to `PPOSetup`
  - Type-safe with `DQNConfig` and `DQN` classes
  - Full docstring with examples

### 3. Config Creation (`config/create_algorithm.py`)
- **Added algorithm detection** in training configuration
- **Conditional training params**:
  - PPO: `minibatch_size`, `num_epochs`, `use_critic`, `clip_param`, `entropy_coeff`, `use_gae`
  - DQN: `target_network_update_freq`, `num_steps_sampled_before_learning_starts`, `tau`, `epsilon`, `double_q`, `dueling`
- Both algorithms share: `gamma`, `lr`, `train_batch_size_per_learner`, `grad_clip`

## Architecture

### Parser Class Hierarchy

```text
_EnvRunnerParser
└── _BaseRLlibArgumentParser (common: train_batch_size_per_learner, lr)
    ├── PPOArgumentParser (adds: minibatch_size, num_epochs)
    ├── DQNArgumentParser (adds: target_network_update_freq, tau, epsilon, etc.)
    └── RLlibArgumentParser (multiple inheritance from PPO + DQN)
```

**Key Design**:
- `RLlibArgumentParser` uses multiple inheritance from both `PPOArgumentParser` and `DQNArgumentParser`
- This eliminates code duplication - all parameter definitions and CLI argument setup are inherited
- Only the conditional validation logic in `process_args()` is implemented
- Backward compatible - existing code using `RLlibArgumentParser` continues to work unchanged
- New specialized parsers (`PPOArgumentParser`, `DQNArgumentParser`) available for algorithm-specific code

## Usage

### Running with PPO (default)
```bash
python experiments/default_training.py --env CartPole-v1
# or explicitly:
python experiments/default_training.py --env CartPole-v1 --algorithm ppo
```

### Running with DQN
```bash
python experiments/default_training.py --env CartPole-v1 --algorithm dqn
```

### DQN-Specific Arguments
```bash
python experiments/default_training.py \
    --algorithm dqn \
    --env CartPole-v1 \
    --target_network_update_freq 1000 \
    --num_steps_sampled_before_learning_starts 5000 \
    --tau 0.001 \
    --epsilon "[(0, 1.0), (50000, 0.01)]" \
    --double_q \
    --dueling
```

### Using Setup Classes
```python
# PPO Setup
from ray_utilities.setup import PPOSetup
setup = PPOSetup()

# DQN Setup
from ray_utilities.setup import DQNSetup
setup = DQNSetup()

# Dynamic Setup (detects from args)
from ray_utilities.setup import AlgorithmSetup
setup = AlgorithmSetup()  # Uses args.algorithm to choose
```

## Remaining Work

### High Priority
1. **Testing**: Create comprehensive tests for DQN setup and training
2. **Dynamic batch/buffer sizing**: Verify compatibility with off-policy DQN
3. **Callbacks**: Ensure all callbacks work with both algorithms

### Medium Priority
4. **Gradient accumulation for DQN**: Implement if needed
5. **Type hints**: Update `algorithm_return.py` and `metrics.py` for DQN-specific metrics (replay buffer stats)
6. **Documentation**: Update user guides and examples

### Low Priority
7. **Additional algorithms**: Consider SAC, TD3, or other algorithms following this pattern
8. **Warnings**: Make batch size warnings conditional on algorithm

## Key Design Decisions

1. **Single parser with both parameter sets**: Rather than creating separate parsers, we include all parameters in `RLlibArgumentParser` and only use relevant ones per algorithm.
   
2. **Dynamic algorithm selection**: Config and algorithm classes are determined at runtime based on `args.algorithm`, not at import time.

3. **Backward compatibility**: Default behavior (PPO) unchanged, DQN opt-in via `--algorithm dqn`.

4. **Checkpoint compatibility**: Algorithm choice is `AlwaysRestore`, ensuring restored experiments use the same algorithm.

## Algorithm Comparison

| Feature | PPO | DQN |
|---------|-----|-----|
| Type | On-policy | Off-policy |
| Experience | Immediate sampling | Replay buffer |
| Update frequency | After each batch | Configurable |
| Key params | minibatch_size, num_epochs | epsilon, replay_buffer_config |
| Target network | No | Yes (soft update via tau) |
| Exploration | Policy entropy | Epsilon-greedy |
