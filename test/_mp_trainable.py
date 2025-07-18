from __future__ import annotations
from ray_utilities.testing_utils import TestHelpers
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing import connection


def remote_process(path_conn, conn: connection.Connection | None = None):
    if isinstance(path_conn, tuple):
        path, conn = path_conn
    else:
        path = path_conn
    assert conn is not None, "Connection must be provided when path_conn is a string"
    print("Creating trainable in remote process")
    helper = TestHelpers()
    trainable, _ = helper.get_trainable(num_env_runners=0)
    print("Saving trainable to", path)
    trainable.save(path)
    return None
    pickled_trainable = pickle.dumps(trainable)
    # conn.send(pickled_trainable)
    conn.close()
    return pickled_trainable


if __name__ == "__main__":
    ...
