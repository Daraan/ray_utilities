from __future__ import annotations

from typing import TYPE_CHECKING

from ray_utilities.testing_utils import TestHelpers

if TYPE_CHECKING:
    from multiprocessing import connection


def remote_process(path_conn, conn: connection.Connection | None = None):
    if isinstance(path_conn, tuple):
        path, conn = path_conn
    else:
        path = path_conn
    print("Creating trainable in remote process")
    helper = TestHelpers()
    trainable, _ = helper.get_trainable(num_env_runners=0)
    print("Saving trainable to", path)
    trainable.save(path)


if __name__ == "__main__":
    ...
