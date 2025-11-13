from ray.rllib.utils.metrics.metrics_logger import *

# pip install https://s3-us-west-2.amazonaws.com/ray-wheels/master/5dc15b43912f58651f99e7b1d015949921ff0e1e/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl

# Simulate first training
root_logger = MetricsLogger()
sub_logger = MetricsLogger()

sub_logger.log_value("test", 100, reduce="sum", clear_on_reduce=False)

# Combine
root_logger.aggregate([sub_logger.reduce()])
root_logger.reduce()

# Checkpoint
logger_state = root_logger.get_state()
sub_logger_state = sub_logger.get_state()

# Recreate
restored_root_logger = MetricsLogger()
restored_sub_logger = MetricsLogger()
restored_root_logger.set_state(logger_state)
restored_sub_logger.set_state(sub_logger_state)

# Continue training
sub_logger.log_value("test", 14, reduce="sum", clear_on_reduce=False)
restored_sub_logger.log_value("test", 14, reduce="sum", clear_on_reduce=False)

# new_logger2.stats["test"].id_ = sub_logger.stats["test"].id_
# Aggregate
sub_reduced = sub_logger.reduce()
restored_sub_reduced = restored_sub_logger.reduce()

root_logger.aggregate([sub_reduced])

# Should log only 14, but logs 114
# aggregate on restored logger, does not work neither sub_reduced nor restored_sub_reduced
restored_root_logger.aggregate([restored_sub_reduced])

# Out
log_reduced = root_logger.reduce()
new_log_reduced = restored_root_logger.reduce()

assert sub_reduced == restored_sub_reduced  # OK
assert log_reduced == new_log_reduced, f"{log_reduced} != {new_log_reduced}"  # FAIL
# 114 (correct) vs 214 (wrong, double counted, last reduce not subtracted)
print("success")
