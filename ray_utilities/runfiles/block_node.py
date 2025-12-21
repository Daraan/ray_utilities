"""Script that will block a node by creating long-running dummy actors on it.

This can be used to prevent scheduling on certain nodes.
Usage:
    ray_utilities/runfiles/block_nody.py <label> <value> [--number N] [--timeout T]

Arguments:
    <label>        The node label to select nodes (e.g., 'gpu_type' or 'ray.io/node-id').
    <value>       The value of the label to select nodes (e.g., 'A100').
    --number N    Number of dummy actors to create (default: 50).
    --timeout T   Time in seconds for which each actor will run (default: 86400 seconds = 24 hours).
"""

if __name__ == "__main__":
    import ray
    import argparse
    import atexit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "label",
    )
    parser.add_argument(
        "value",
    )
    parser.add_argument("--number", "-n", type=int, default=None)
    parser.add_argument("--timeout", "-t", type=int, default=24 * 60 * 60)
    args = parser.parse_args()
    if args.label.replace("-", "_") in ("node_id",):
        args.label = "ray.io/node-id"

    @ray.remote(label_selector={args.label: args.value}, num_cpus=1)
    class BlockNode:
        timeout = args.timeout

        def wait(self):
            import time  # noqa: PLC0415

            time.sleep(self.timeout)

    blockers = [BlockNode.remote() for _ in range(args.number or 50)]
    pings = [b.wait.remote() for b in blockers]  # pyright: ignore[reportAttributeAccessIssue]
    print(f"Created {len(blockers)} blocking actors on nodes with {args.label}={args.value}")

    def cleanup():
        for block in blockers:
            ray.kill(block, no_restart=True)  # pyright: ignore[reportArgumentType]

    atexit.register(cleanup)
    ray.wait(pings, num_returns=len(pings), fetch_local=False)  # Block until all timed out

    ray.shutdown()
