import logging
import os

os.environ["RAY_DEDUP_LOGS_ALLOW_REGEX"] = "COMET|wandb"

import ray

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.handlers = [handler]

wandb_logger = logging.getLogger("wandb")
wandb_logger2 = logging.getLogger("wandb")
wandb_formatter = logging.Formatter("WANDB: %(levelname)s:%(name)s:%(message)s")
wandb_formatter2 = logging.Formatter("wandb: %(levelname)s:%(name)s:%(message)s")


@ray.remote
def f():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # wandb_handler = logging.StreamHandler()
    # wandb_handler.setLevel(logging.INFO)
    # wandb_handler.setFormatter(wandb_formatter)

    wandb_handler2 = logging.StreamHandler()
    wandb_handler2.setLevel(logging.INFO)
    wandb_handler2.setFormatter(wandb_formatter2)

    logger.handlers = [
        # handler,
        # wandb_handler,
        wandb_handler2,
    ]
    logger.info("wandb log")
    logger.info("COMET log")
    logger.info("comet log")
    logger.info("some other log")


ray.init()
ray.get([f.remote() for _ in range(5)])
