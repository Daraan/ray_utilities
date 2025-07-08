# simulate Increase
if __name__ == "__main__X":
    from ray_utilities import nice_logger
    from ray_utilities.dynamic_config.dynamic_buffer_update import split_timestep_budget

    nice_logger(logger, "DEBUG")
    global_step = 0
    dynamic_buffer = True
    dynamic_batch = True
    iterations = 340
    total_steps = 1_000_000
    # Test
    budget = split_timestep_budget(
        total_steps=total_steps,
        min_size=32,
        max_size=2**13,
        assure_even=True,
    )
    total_steps = budget["total_steps"]
    n_envs = 1
    n_steps = 2048 // 8
    initial_steps = budget["step_sizes"][0]  # 128
    # batch_size = n_envs * n_steps
    n_steps_old = None

    while global_step < total_steps:
        global_step += n_steps
        batch_size, _, n_steps = update_buffer_and_rollout_size(
            total_steps=total_steps,
            dynamic_buffer=dynamic_buffer,
            dynamic_batch=dynamic_batch,
            global_step=global_step,
            accumulate_gradients_every_initial=1,
            initial_steps=initial_steps,
            num_increase_factors=len(budget["step_sizes"]),
            n_envs=1,
        )
        if n_steps_old != n_steps:
            n_steps_old = n_steps
            logger.debug(
                "Updating at step %d / %s (%f%%) to '%s x %s=%s' from initially (%s), batch size=%s (initial: %s)",
                global_step,
                total_steps,
                round((global_step / total_steps) * 100, 0),
                n_steps,
                1,
                n_steps * 1,
                initial_steps,
                batch_size,
                int(n_envs * initial_steps),
            )
    logger.debug(
        "Finished at step %d / %s (%f%%) to '%s x %s=%s' from initially (%s), batch size=%s (initial: %s)",
        global_step,
        total_steps,
        round((global_step / total_steps) * 100, 0),
        n_steps,
        1,
        n_steps * 1,
        initial_steps,
        batch_size,  # pyright: ignore[reportPossiblyUnboundVariable]
        int(n_envs * initial_steps),
    )
