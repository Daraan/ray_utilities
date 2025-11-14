import ray

# Connect to existing cluster
ray.init(address="auto")  # Auto-discovers running cluster

# Get cluster info
print(ray.get_runtime_context().gcs_address)
# Example output: 192.168.1.100:6379

# Get node IP
import socket  # noqa: E402

print(socket.gethostbyname(socket.gethostname()))
