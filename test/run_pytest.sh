#!/bin/bash
# runs pytest locally in parallel
# file needs to be kept in sync manually with the workflow file

export RAY_UTILITIES_KEEP_TESTING_STORAGE=1 # when run in parallel do not clear output dir that are still used
# Note each process takes up to 4 cpus
(trap 'kill 0' SIGINT; 
pytest -k "not test_trainable" -v -x --fast --capture=sys &
pytest -k "test_validate_save_restore" -v --fast --timeout_method=thread --capture=sys &
pytest -k "test_trainable and not test_tuner_checkpointing and not test_validate_save_restore"  -v --fast --capture=sys &
pytest -k "test_tuner_checkpointing" -v --fast --timeout_method=thread --capture=sys &
pytest -k "test_trainable and not test_tuner_checkpointing and not test_validate_save_restore"  -v --timeout_method=thread --mp-only --capture=sys &
pytest -k "TestMetricsRestored and test_with_tuner" --mp-only --capture=sys &
pytest -k "test_tuner_checkpointing" -v --mp-only --timeout_method=thread --capture=sys &
wait
)
rm -rf ../outputs/experiments/TESTING