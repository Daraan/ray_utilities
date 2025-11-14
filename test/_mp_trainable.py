from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ray_utilities.testing_utils import TestHelpers

if TYPE_CHECKING:
    from multiprocessing import connection


def remote_process(
    path_conn: dict[str, Any] | str,
    conn: connection.Connection | None = None,
    num_env_runners: int = 0,
    env_seed: int | None = None,
):
    if isinstance(path_conn, dict):
        path = path_conn["dir"]
        # conn = path_conn["connection"]
        num_env_runners = path_conn["num_env_runners"]
        env_seed = path_conn["env_seed"]
    else:
        path = path_conn
    print("Creating trainable in remote process")
    helper = TestHelpers()
    trainable, _ = helper.get_trainable(num_env_runners=num_env_runners, env_seed=env_seed, eval_interval=None)
    print("Saving trainable to", path)
    trainable.save(path)
    # Cannot pickle
    # return cloudpickle.dumps(trainable)
