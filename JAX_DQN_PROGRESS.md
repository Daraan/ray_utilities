# JAX DQN Implementation Progress

## Completed ✅

### 1. JaxDQNModule (`ray_utilities/jax/dqn/jax_dqn_module.py`)
✅ **Status: Complete and lint-error-free** (356 lines)

**Features Implemented:**
- Extends both `DefaultDQNRLModule` (for DQN interface) and `JaxModule` (for JAX functionality)
- Support for dueling architecture (separate advantage and value streams)
- Support for distributional Q-learning (C51/Rainbow style with atoms)
- Proper JAX state management via `JaxDQNStateDict`
- Q-value computation for both current and target networks
- Configurable encoder and head networks

**Key Methods:**
- `setup()`: Initializes encoder, advantage head, value head (if dueling), and network states
- `_init_qf_state()`: Creates initial Q-function state
- `_forward()` / `_forward_train()`: Forward passes for inference and training
- `compute_q_values()`: Main Q-value computation using current network
- `compute_target_q_values()`: Target Q-value computation for TD-error
- `_qf_forward_helper()`: Core helper for Q-value computation through encoder and heads

**Architecture Support:**
1. Standard Q-Network: `encoder → af (advantage) → Q-values`
2. Dueling Q-Network: `encoder → af + vf → Q = V + (A - mean(A))`
3. Distributional (C51): Adds logits, probs, and atoms for value distribution

**State Structure:**
```python
JaxDQNStateDict = {
    "qf": {...},           # Current Q-function state
    "qf_target": {...},    # Target Q-function state  
    "module_key": ...,     # JAX PRNG key
}
```

### 2. JaxDQNCatalog (`ray_utilities/jax/dqn/jax_dqn_catalog.py`)
✅ **Status: Complete and lint-error-free** (162 lines)

Extends both `DQNCatalog` and `JaxCatalog` to provide JAX-specific model creation for DQN.

**Features:**
- Builds JAX/Flax encoders using parent Catalog logic
- Builds advantage/Q-function heads (AF) with configurable output dimensions
- Builds value function heads (VF) for dueling architecture
- Supports distributional Q-learning via `num_atoms` configuration
- Framework validation (enforces "jax" framework)

**Key Methods:**
- `build_encoder()`: Creates JAX encoder from observation space
- `build_af_head()`: Creates advantage/Q-function head (action_space.n * num_atoms outputs)
- `build_vf_head()`: Creates value head for dueling (1 output)
- `_get_head_config()`: Inherited from DQNCatalog for MLP head configuration

### 3. compute_dqn_loss (`ray_utilities/jax/dqn/compute_dqn_loss.py`)
✅ **Status: Complete and lint-error-free** (173 lines)

Factory function for creating JIT-compiled DQN loss computation functions.

**Features:**
- JIT-compiled TD-error loss computation
- Support for MSE and Huber loss
- Double DQN support (use current network to select, target to evaluate)
- Handles terminal states correctly (no bootstrap on done=True)
- Loss masking support for handling variable-length episodes
- Comprehensive metrics tracking

**Key Function:**
- `make_jax_compute_dqn_loss_function()`: Factory that creates specialized loss function
  - Returns `jax_compute_loss_for_module()` JIT-compiled function
  - Signature: `(qf_state_params, batch, fwd_out, gamma, *, double_q) -> (loss, metrics)`
  - Metrics: td_error_mean, q_mean, q_max, q_min

**Loss Computation:**
1. Extract Q-values for actions taken: `Q(s, a)`
2. Compute target Q-values: `r + gamma * max_a' Q_target(s', a')`
3. TD-error: `delta = target - Q(s, a)`
4. Loss: MSE or Huber loss on TD-error
5. Return scalar loss and metrics tuple

### 4. JaxDQNLearner (`ray_utilities/jax/dqn/jax_dqn_learner.py`)
✅ **Status: Complete and lint-error-free** (383 lines)

JAX-based learner extending both `DQNLearner` and `JaxLearner` for efficient JIT-compiled DQN training.

**Features:**
- JIT-compiled forward pass with automatic differentiation
- JIT-compiled parameter updates
- Target network updates (soft or hard) based on timestep frequency
- Gradient accumulation support (parameter available but not yet used)
- Q-function state management with target network tracking
- Comprehensive metrics logging

**Key Methods:**
- `build()`: Initialize JAX-specific state, create JIT-compiled functions
- `compute_loss_for_module()`: Compute TD-error loss for single module
- `_jax_compute_losses()`: Compute losses for all modules (JIT-compatible)
- `_forward_train_call()`: Execute forward pass for training
- `_jax_forward_pass()`: JIT-compilable forward + loss computation
- `_update_jax()`: JIT-compiled parameter update with gradient computation
- `_update()`: Main update method called during training
- `after_gradient_based_update()`: Update target networks after gradient update
- `_get_state_parameters()`: Extract parameters from Q-function states

**State Management:**
- `_states`: Mapping[ModuleID, JaxDQNStateDict] with qf, qf_target, module_key
- `_compute_loss_for_modules`: Dict of JIT-compiled loss functions per module
- `_forward_with_grads`: JIT-compiled forward pass with gradient computation
- `_update_jax`: JIT-compiled update function

### 5. Module Exports (`ray_utilities/jax/dqn/__init__.py`)
✅ **Status: Complete**

Exports all JAX DQN components with proper documentation:
- `JaxDQNModule`, `JaxDQNStateDict`
- `JaxDQNCatalog`
- `JaxDQNLearner`
- `make_jax_compute_dqn_loss_function`

---

## Next Steps

### 6. Integration with Setup Classes
**Status: Not started**

**Purpose:** Add JAX DQN learner configuration when using `--jax` flag with DQN algorithm

**Location:** `ray_utilities/setup/algorithm_setup.py`

**Required changes:**
```python
# In _get_algorithm_classes() or similar setup method
if args.jax and args.algorithm == "dqn":
    from ray_utilities.jax.dqn import JaxDQNLearner, JaxDQNCatalog
    config.learners(learner_class=JaxDQNLearner)
    config.rl_module(
        rl_module_class=JaxDQNModule,
        model_config={"catalog_class": JaxDQNCatalog},
    )
```

### 7. Update Parent __init__.py
**Status: Not started**

### 4. JaxDQNLearner
**Status: Not started**

**Purpose:** JAX-based learner for DQN training with TD-error computation

**Pattern to follow:** `ray_utilities/jax/ppo/jax_ppo_learner.py` (JaxPPOLearner)

**Key differences from PPO:**
- DQN uses TD-error loss (not policy gradient)
- Needs target network updates (soft or hard)
- Uses replay buffer (not on-policy rollouts)
- May need double DQN support (max over current, evaluate with target)

**Required methods:**
- `compute_loss_for_module()`: Compute DQN TD-error loss
  - Extract Q-values from module
  - Compute target Q-values
  - Calculate TD-error: `td_error = (r + γ * max_a' Q_target(s', a')) - Q(s, a)`
  - Apply loss function (MSE or Huber)
- `update()` or `after_gradient_based_update()`: Sync target network
- Handle distributional loss if `num_atoms > 1` (cross-entropy with projected distribution)

**Dependencies:**
- Needs `compute_dqn_loss()` function (see next item)
- Target network update logic (tau parameter for soft updates)

### 5. compute_dqn_loss.py
**Status: Not started**

**Purpose:** Jittable loss computation function for DQN

**Pattern to follow:** `ray_utilities/jax/ppo/compute_ppo_loss.py`

**Required functionality:**
```python
@jax.jit
def compute_dqn_loss(
    module: JaxDQNModule,
    batch: dict[str, Any],
    params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    gamma: float = 0.99,
    double_q: bool = True,
    use_huber: bool = False,
    ...
) -> tuple[jax.Array, dict[str, Any]]:
    """
    Compute DQN TD-error loss.
    
    Returns:
        loss: Scalar loss value
        metrics: Dictionary with td_error, q_values, target_q_values, etc.
    """
```

**Key computations:**
1. Forward pass for current Q-values: `Q(s, a)`
2. Forward pass for next-state Q-values with target network
3. Compute target: `y = r + γ * max_a' Q_target(s', a')`
4. TD-error: `δ = y - Q(s, a)`
5. Loss: MSE or Huber loss on TD-error
6. Handle terminal states (no bootstrap if done)

**Special cases:**
- Double DQN: Use current network to select action, target network to evaluate
- Distributional: Project target distribution onto support and use cross-entropy loss

### 6. Update Parent __init__.py Files
**Status: Not started**

**Files to update:**
- `ray_utilities/jax/__init__.py`: Export JaxDQNModule, JaxDQNCatalog, JaxDQNLearner

### 7. Testing
**Status: Not started**

**Test files to create:**
- `test/jax/test_jax_dqn_module.py`: Unit tests for JaxDQNModule
  - Test standard and dueling architectures
  - Test distributional Q-learning
  - Test state initialization
  - Test forward passes
- `test/jax/test_jax_dqn_learner.py`: Integration tests for training
  - Test loss computation
  - Test target network updates
  - Test with simple environment (CartPole)

**Pattern to follow:** Existing JAX PPO tests (if any)

---

## Integration Points

### With Existing Setup Classes
Current setup in `ray_utilities/setup/algorithm_setup.py`:
```python
def _get_algorithm_classes(...):
    if args.algorithm == "dqn":
        return DQNConfig, DQN
    ...
```

**JAX learner configuration** (to be added):
```python
if args.jax and args.algorithm == "dqn":
    from ray_utilities.jax.dqn import JaxDQNLearner
    config.learners(learner_class=JaxDQNLearner)
    config.rl_module(rl_module_class=JaxDQNModule)
```

### With Existing Argument Parser
May need to add JAX-specific DQN arguments to `DQNArgumentParser`:
- `--num-atoms`: Number of atoms for distributional Q-learning
- `--v-min`, `--v-max`: Value range for distributional Q-learning
- `--double-q`: Enable double DQN
- `--target-update-tau`: Soft target network update coefficient

---

## Implementation Priority

1. **JaxDQNCatalog** (highest priority)
   - Required for model creation
   - Relatively straightforward port from JaxCatalog

2. **compute_dqn_loss.py** (high priority)
   - Core training logic
   - Can be developed and tested independently

3. **JaxDQNLearner** (high priority)
   - Ties everything together
   - Depends on catalog and loss function

4. **Testing** (medium priority)
   - Validate implementation
   - Can be done incrementally with each component

5. **Documentation** (low priority)
   - Update docs with JAX DQN examples
   - Add to existing experiment files

---

## Notes

- The `JaxDQNModule` uses a different parameter handling approach than the PyTorch `DefaultDQNRLModule`:
  - JAX models (encoder, af, vf) are called directly without explicit parameter passing
  - Parameters are managed internally by the Flax models
  - The `_qf_forward_helper` takes model objects, not parameter dicts

- Target network updates in DQN:
  - Hard update: Copy parameters from current to target network every N steps
  - Soft update: `target = tau * current + (1 - tau) * target` every step
  - Ray's DQN may have built-in support for this

- Distributional Q-learning requires careful handling:
  - Forward pass returns logits over value support
  - Target distribution must be projected onto support atoms
  - Loss is cross-entropy, not MSE

## References

- JAX PPO implementation: `ray_utilities/jax/ppo/`
- PyTorch DQN implementation: Ray RLlib's `DefaultDQNRLModule`, `DQNTorchLearner`
- DQN paper: https://www.nature.com/articles/nature14236
- Rainbow paper (distributional): https://arxiv.org/abs/1710.02298
