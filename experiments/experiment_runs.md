# For each env
# 2x2 different seeds.
# - DQN
# -evaluation_num_env_runners 1
# ALWAYS USE TMUX or SLURM

# Activate venv before running
tmux new-window -t session -n window_name 'source ../env/bin/activate && bash'


--total_steps ?
--comet offline+upload --wandb offline+upload
--seed 42
--seed 128  --tag:seed=128
--checkpoint_frequency 65.536  # 8192*8
--tag:core

TODO:
- do with num_envs_per_env_runner = 1

# possibly superseeded by tune
python experiments/default_training.py --num_samples 3
python experiments/default_training.py --num_samples 3 --seed 128

# Static runs

## ------------ Batch size (Rollout buffer) ------------
### grid_search

### Tune Normal

    python experiments/tune_batch_size.py --tune batch_size --num_samples 16 --comet offline+upload --wandb offline+upload --tag:core --comment "Core: Tune Batch Size exhaustive"
    [x] ( 23d5ab25102720579eba3)

    // Seed 128

    python experiments/tune_batch_size.py
        --tune batch_size --num_samples 16 \
        --seed 128 --tag:seed=128 \
        --comet offline+upload --wandb offline+upload \
        --tag:core --comment "Core: Tune Batch Size exhaustive"

### PBT
// num_samples is multiplied by number of population members, i.e. 2x8 = 16
// Keep 3 top trials. Guarantees top quantile is not pure and also at least one resample once into the top trial.
// 3 / 16 = 0.1875  (0.15 enough with ceiling)

    python experiments/tune_with_scheduler.py \
        --tune batch_size --num_samples 2 \
        --tag:core --comment "Core: Tune Batch Size (PBT)" \
        --comet offline+upload --wandb offline+upload \
        pbt --quantile_fraction 0.1875 --perturbation_interval 0.125

    // Seed 128
    python experiments/tune_with_scheduler.py --tune batch_size --num_samples 2 --tag:core --comment "Core: Tune Batch Size (PBT)" --seed 128 --tag:seed=128 pbt --quantile_fraction 0.1875 --perturbation_interval 0.125


#### PBT with just 1 sample range, do twice. Guarantees full exploration range for trial.

    python experiments/tune_with_scheduler.py \
         --env_type CartPole-v1 \
         --tune batch_size \
         --num_samples 1 \
         --tag:core --log_stats learners \
         --comment "Core: PBT Batch Size exhaustive" \
         --comet  offline+upload@end  --wandb  offline+upload@end    --tag:pbt  \
         pbt  --perturbation_interval 0.125 --quantile_fraction 0.1

    [ ]  0549cd2511071358c3033


    // Seed 128
    python experiments/tune_with_scheduler.py \
         --env_type CartPole-v1 \
         --tune batch_size \
         --num_samples 1 \
         --tag:core --log_stats learners \
         --seed 128 --tag:seed=128 \
         --comment "Core: PBT Batch Size exhaustive" \
         --comet  offline+upload@end  --wandb  offline+upload@end    --tag:pbt  \
         pbt  --perturbation_interval 0.125 --quantile_fraction 0.1

    // Seed 256


## ------------ Minibatch size (SGD batch size) ---------

### Question: Minibach size vs Gradient Accumulation ? Do both?

-- A --

- Only minibatch size:
- constant batch_size (rollout) of 2048 / 8192

### Tune grid_search
// cont have minibatch_size > batch_size -> less runs
// 2048 variants: 64, 128, 256, 512, 1024, 2048 = 6 variants
// TODO: 64 is NOT in the variant currently.
NOTE: for optuna should duplicate the entries in the grid search instead of increasing num_samples

#### Batch size 2048

    // General tune file
    python experiments/tune.py --batch_size 2048 --tune minibatch_size --tag:tune:minibatch_size --num_samples 12 --tag:core  --comment "Core: Tune minibatch_size Size exhaustive"

    // Seed 128
    python experiments/tune.py --batch_size 2048 --tune minibatch_size --tag:tune:minibatch_size --num_samples 12 --tag:core --evaluation_num_env_runners 1 --seed 128 --tag:seed=128 --comment "Core: Tune minibatch_size Size exhaustive"
    [x] (259cfd25102509323cf03, 259cfd2510251623b46f3)

    // Seed 128
    python experiments/tune_batch_size.py --batch_size 2048 --tune minibatch_size --num_samples 12 --tag:core --seed 128 --tag:seed=128 --comment "Core: Tune minibatch_size Size exhaustive"
    [x] (259cfd25102501165267)


#### Batch size 8196

    // 8192 variants: +2 = 8 variants

    experiments/tune.py --batch_size 8192 --tune minibatch_size --tag:tune:minibatch_size --num_samples 16 --tag:core --comment "Core: Tune minibatch_size Size exhaustive"
    [ ] (259cfd2510270101ee6f3 DWS (completed, needs upload!)) - missing
    [ ] (259cfd2510271107333d3 61252 (likely duplicate))

    // Seed 128
    experiments/tune.py --batch_size 8192 --tune minibatch_size --tag:tune:minibatch_size --num_samples 16 --tag:core --seed 128 --tag:seed=128 --comment "Core: Tune minibatch_size Size exhaustive"

// use 16000?

### Tune as gradient accumulation

    experiments/tune.py \
        --batch_size 128 --tune accumulate_gradients_every --num_samples 2 \
        --tag:tune:accumulate_gradients_every --tag:core --comment "Core: Tune accumulate_gradients_every base 128" \
        --comet offline+upload --wandb offline+upload --log_stats timers+learners \
        --log_level INFO

### PBT

// Use BasicVariantGenerator so that minibatch_size <= train_batch_size

#### Batch Size 2048

    // Variants: 128, 256, 512, 1024, 2048 * 2 samples = 10
    python experiments/tune_with_scheduler.py --batch_size 2048 --tune minibatch_size --num_samples 2 --tag:core --comment "Core: Tune Minibatch Size (PBT)" --wandb offline+upload@end --comet offline+upload --log_stats timers+learners pbt --quantile_fraction 0.1875 --perturbation_interval 0.125
    [ ] (61260 0549cd2510271138ffd13, 0549cd2510280059d2cf3)

#### Batch Size 8192
    // +  2*2 samples = 14 variants

    python experiments/tune_with_scheduler.py --batch_size 8192 --tune minibatch_size --num_samples 2 --tag:core --comment "Core: Tune Minibatch Size (PBT)" --wandb offline+upload@end --comet offline+upload --log_stats timers+learners pbt --quantile_fraction 0.1875 --perturbation_interval 0.125
    [ ] (0549cd251027004189c43 - needs upload and verification)

    python experiments/tune_with_scheduler.py \
        --env_type CartPole-v1 \
        --tune minibatch_size \
        --num_samples 1 \
        --batch_size 8192 \
        --tag:core --log_stats learners \
        --comment "Core: Tune Minibatch Size (PBT)" \
        --comet  offline+upload@end  --wandb  offline+upload@end  --tag:pbt  \
        pbt  --perturbation_interval 0.125 --quantile_fraction 0.125
    [ ] (0549cd2511071329baf53)


#### PBT 2 ?

    // OPEN

## -- B --  Batch size AND Minibatch size -------

- minibatch_scale: full, half, quarter, 1/8, 128
- TODO: need minimum value (else skip) - grid search with batch size
- 128 included in Normal

### Tune Normal
// 8*8 = 64 variants, but minibatch_size > batch_size skipped, so 8 + 7 + 6 + 5 + 4 +3 +2 +1 = 36 variants
// Skip double samples as too many

    python experiments/tune.py --tune batch_size minibatch_size --num_samples 36 -J 6 --tag:core --comet offline+upload --wandb offline+upload --log_level IMPORTANT_INFO --log_stats timers+learners --comment "Core: Tune batch_size + minibatch_size Size exhaustive" --num_env_runners 1
    [x] ( 259cfd2510310125f2493 )


    // Seed 128
    python experiments/tune.py --tune minibatch_size batch_size --num_samples 36 --tag:core --seed 128 --tag:seed=128 --comment "Core: Tune batch_size + minibatch_size Size exhaustive"

    #### Acrobot
    python experiments/tune.py \
        --env_type Acrobot-v1 \
        --tune batch_size minibatch_size \
        --num_env_runners 1 \
        --num_samples 36 -J 8 \
        --tag:core --comment "Core: Tune batch_size + minibatch_size Size \
        --comet offline+upload --wandb offline+upload \
        --log_level IMPORTANT_INFO --log_stats timers+learners exhaustive"
    [x] (259cfd25110202005af03)

    #### Lunar Lander
    # new usage of num_samples
    python experiments/tune.py \
        --env_type LunarLander-v3 \
        --tune batch_size minibatch_size \
        --num_samples 2 \
        --tag:core --comment "Core: Tune batch_size + minibatch_size Size \
        --comet offline+upload --wandb offline+upload \
        --log_level IMPORTANT_INFO --log_stats timers+learners exhaustive"
    [ ] (259cfd25110414551cbf3) some errored, resume?
    [ ]  0549cd251106103384aa3 some missing




### 1 env only:

    python experiments/tune.py --tune batch_size minibatch_size --num_samples 36 -J 6 --tag:core --comet offline+upload --wandb offline+upload --log_level IMPORTANT_INFO --log_stats timers+learners --comment "Core: Tune batch_size + minibatch_size Size exhaustive" --num_env_runners 0 --num_envs_per_env_runner 1
    [x] (259cfd2510311124ceaf3 - some duplicates might be missing)

### PBT

    python experiments/tune_with_scheduler.py --tune batch_size minibatch_size --num_samples 2 --tag:core --tag:pbt --comet offline+upload --wandb offline --offline_loggers json --log_level IMPORTANT_INFO  --comment "Core: Tune batch_size + minibatch_size Size exhaustive" --num_env_runners 1 --evaluation_num_env_runners 1 --log_stats timers+learners pbt --quantile_fraction 0.1 --perturbation_interval 0.125
    [💥] Default-mlp-CartPole-v1-0549cd2510310950a3bb3 (incomplete)
    [x] (0549cd2510312359a666, unfinished)
    [x] (0549cd251101162659a73)  (some errors on last steps, uploaded)

    // only 1 env (also use buffer_length="auto")
    python experiments/tune_with_scheduler.py \
        --tune batch_size minibatch_size --num_samples 1 \
        --tag:core --tag:pbt --comment "Core: PBT Tune batch_size + minibatch_size Size exhaustive, single environment" \
        --comet offline+upload@end --wandb offline+upload@end --offline_loggers json --log_level IMPORTANT_INFO --log_stats timers+learners \
        --num_envs_per_env_runner 1 --buffer_length auto \
        pbt --quantile_fraction 0.1 --perturbation_interval 0.125
    [ ] (0549cd2511011846a6733, canceled slow)
    [ ] (0549cd25110201490ef43)  errored at mid run, no priority to continue

    #### Acrobot
    // 1 env only
    python experiments/tune_with_scheduler.py \
        --env_type Acrobot-v1 \
        --tune batch_size minibatch_size --num_samples 2 \
        --tag:core --tag:pbt --comment "Core: PBT Tune batch_size + minibatch_size Size exhaustive, single environment" \
        --comet offline+upload@end --wandb offline+upload@end --offline_loggers json --log_level IMPORTANT_INFO --log_stats timers+learners \
        pbt --quantile_fraction 0.1 --perturbation_interval 0.125
    [ ] ( 0549cd2511021624bba43) errored

    #### Lunar Lander
    python experiments/tune_with_scheduler.py \
        --env_type LunarLander-v3 \
        --tune batch_size minibatch_size --num_samples 2 \
        --tag:core --tag:pbt --comment "Core: PBT Tune batch_size + minibatch_size Size exhaustive" \
        --comet offline+upload@end --wandb offline+upload@end --offline_loggers json --log_level IMPORTANT_INFO --log_stats timers+learners \
        --buffer_length auto \
        pbt --quantile_fraction 0.1 --perturbation_interval 0.125
    [ ] ( ) - needs replay wandb crashed
    [ ] (0549cd2511041948f7293 ) many not uploaded



### PBT 2

    // OPEN

## B 2 --  Batch size AND Minibatch size SCALE -------


# Num envs

## Tune Normal

// Not that long


## PBT

    python experiments/tune_with_scheduler.py \
        --tune num_envs_per_env_runner --num_samples 1 \
        --tag:core --tag:pbt --comment "Core: PBT num environments" \
        --comet offline+upload --wandb offline --offline_loggers json --log_level IMPORTANT_INFO \
        --minibatch_size 256 \
        --num_env_runners 1 --evaluation_num_env_runners 0 \
        --total_steps 524288 --use_exact_total_steps \
        --log_stats timers+learners pbt --quantile_fraction 0.1 --perturbation_interval 0.25

    [ ] (0549cd2511042048d8bd3) - needs wandb upload, batch_size 256 variant (less perturbations)
        - need replay:
            - 0549cd2511042048d8bd3X32d97C00F32d97C01S0cMK (40% missing of train metrics)

    [ ] (0549cd2511042239764e3) minibatch_size 128

    python experiments/tune_with_scheduler.py \
        --tune num_envs_per_env_runner batch_size --num_samples 2 \
        --tag:core --tag:pbt --comment "Core: PBT num_environments to batch_size & minibatch_scale" \
        --comet offline+upload --wandb offline+upload@end --offline_loggers json --log_level IMPORTANT_INFO \
        --minibatch_scale 0.25 \
        --total_steps 524288 --use_exact_total_steps \
        --log_stats timers+learners pbt --quantile_fraction 0.1 --perturbation_interval 0.125
    [ ] (0549cd25110422496a333) - canceled error; not uploaded to Wandb
    [ ] (0549cd25110520296c5b3)

    python experiments/tune_with_scheduler.py \
        --tune num_envs_per_env_runner batch_size --num_samples 2 \
        --tag:core --tag:pbt --comment "Core: PBT num_environments to batch_size & minibatch_scale" \
        --comet offline+upload --wandb offline+upload@end --offline_loggers json --log_level IMPORTANT_INFO \
        --minibatch_scale 0.125 \
        --total_steps 524288 --use_exact_total_steps \
        --log_stats timers+learners pbt --quantile_fraction 0.1 --perturbation_interval 0.125
    [ ] (0549cd251104225098923) - canceled error; not uploaded to Wandb
        - many need replay
        - 0549cd251104225098923X52a89C06F52a89C62S1NFA
        - 0549cd251104225098923X52a89C13F52a89C62S1NFA
        - 0549cd251104225098923X52a89C12F52a89C61S16C8
    [x] (0549cd2511052028cb8e3) - failed, only interesting for start (1st perturbation)
    [ ] Redo low prio


    // Acrobot

     python experiments/tune_with_scheduler.py \
        --env_type Acrobot-v1 \
        --tune num_envs_per_env_runner --num_samples 1 \
        --tag:core --tag:pbt --comment "Core: PBT num environments" \
        --comet offline+upload --wandb offline --offline_loggers json --log_level IMPORTANT_INFO \
        --minibatch_size 256 \
        --total_steps 524288 --use_exact_total_steps \
        --log_stats timers+learners pbt --quantile_fraction 0.1 --perturbation_interval 0.25

#

TODO: Add check implementation and add minimum value (skip some combinations)
experiments/tune.py --tune minibatch_scale batch_size --num_samples 36 --tag:core --seed 128 --tag:seed=128 --comment "Core: Tune minibatch_size Size exhaustive"

# Dynamic Scaling

- dynamic increase batch_size (rollout)
- dynamic increase minibatch_size (SGD batch)
- increase number of envs sampled from
- other hyperparameters

# Dynamic increase batch_size (rollout)
# Minibatch size: Use full, half, quarter, (512), 128 batch
