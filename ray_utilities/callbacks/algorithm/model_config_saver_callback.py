from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.rl_module.rl_module import RLModule

logger = logging.getLogger(__name__)


def save_model_config_and_architecture(*, algorithm: "Algorithm", **kwargs) -> None:
    """on_algorithm_init callback that saves the model config and architecture as json dict."""
    module = _get_module(algorithm)
    config = _get_module_config(module)
    for k, v in config.items():
        config[k] = repr(v).replace("\\n", "\n")
    arch = _get_model_architecture(module)
    output = {
        "config": config,
        "architecture": arch,
    }
    out_path = "./model_architecture.json"
    try:
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Saved model architecture/config to: %s", out_path)
    except OSError as e:
        logger.error("Failed to save model architecture/config: %s", str(e))


def _get_module(algorithm: "Algorithm") -> TorchRLModule:
    module = getattr(algorithm, "rl_module", None)
    if module is None:
        try:
            module = algorithm.learner_group._learner.module  # pyright: ignore[reportOptionalMemberAccess]
            assert module
        except AttributeError:
            module = algorithm.config.rl_module_spec.build()  # pyright: ignore[reportOptionalMemberAccess]
    if isinstance(module, MultiRLModule):
        module = module["default_policy"]
    if isinstance(module, TorchRLModule):
        return module
    if module is not None and hasattr(module, "_modules"):
        modules = getattr(module, "_modules", {})
        for m in modules.values():
            if isinstance(m, TorchRLModule):
                return m
    raise RuntimeError("No TorchRLModule found in algorithm.rl_module")


def _get_module_config(module: TorchRLModule | RLModule) -> dict:
    # config of RLModule is deprecated
    args, kwargs = module.get_ctor_args_and_kwargs()
    return {"args": args, **kwargs}


def _get_model_architecture(module: TorchRLModule) -> dict:
    arch = {}
    torch_model = getattr(module, "model", module)
    arch["summary"] = str(torch_model)
    arch["layers"] = _extract_layers(torch_model)
    return arch


def _extract_layers(torch_model) -> list:
    layers = []
    for name, layer in getattr(torch_model, "named_modules", list)():
        if name == "":
            continue
        layer_info = {
            "name": name,
            "type": layer.__class__.__name__,
            "params": sum(p.numel() for p in getattr(layer, "parameters", lambda recurse=False: [])(recurse=False)),
        }
        layers.append(layer_info)
    return layers
