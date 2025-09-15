from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, TypeAlias

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.rl_module.rl_module import RLModule

logger = logging.getLogger(__name__)

_Node: TypeAlias = dict[str, str | dict[str, "_Node"] | list["_Node"]]


def _jsonify_summary(summary: str) -> dict:
    lines = [line.rstrip() for line in summary.strip().splitlines() if line.strip()]
    stack = []
    root: _Node = {}
    current: _Node = root
    indent_stack = [-1]
    name_stack = []

    def add_child(parent, name, child):
        if "children" not in parent:
            parent["children"] = {}
        parent["children"][name] = child

    for line in lines:
        indent = len(line) - len(line.lstrip())
        content = line.strip()

        # Match module start: e.g. (encoder): TorchActorCriticEncoder(
        m = re.match(r"\(([^)]+)\): ([^(]+)\($", content)
        if m:
            key, module = m.groups()
            node = {"type": module, "children": {}}
            add_child(current, key, node)
            stack.append(current)
            current = node
            indent_stack.append(indent)
            name_stack.append(key)
            continue

        # Match top-level module: e.g. DefaultPPOTorchRLModule(
        m = re.match(r"([^(]+)\($", content)
        if m:
            module = m.group(1)
            node = {"type": module}  # "children": {}
            root = node
            current = node
            stack = []
            indent_stack = [-1]
            name_stack = [module]
            continue

        # Match layer: e.g. (0): Linear(in_features=4, out_features=1, bias=True)
        m = re.match(r"\(([^)]+)\): ([^(]+)\((.*)\)", content)
        if m:
            key, layer, params = m.groups()
            if "layers" not in current:
                current["layers"] = []
            layer_info = {"name": key, "type": layer}
            if params:
                layer_info["params"] = params
            current["layers"].append(layer_info)  # pyright: ignore[reportAttributeAccessIssue]
            continue

        # Match simple layer: e.g. (1): Tanh()
        m = re.match(r"\(([^)]+)\): ([^(]+)\(\)", content)
        if m:
            key, layer = m.groups()
            if "layers" not in current:
                current["layers"] = []
            current["layers"].append({"name": key, "type": layer})  # pyright: ignore[reportAttributeAccessIssue]
            continue

        # Match block end: )
        if content == ")":
            while indent_stack and indent_stack[-1] >= indent:
                current = stack.pop()
                indent_stack.pop()
                name_stack.pop()
            continue

    return root


def save_model_config_and_architecture(*, algorithm: "Algorithm", **kwargs) -> None:
    """on_algorithm_init callback that saves the model config and architecture as json dict."""
    module = _get_module(algorithm)
    config = _get_module_config(module)
    for k, v in config.items():
        # Step 1: Convert value to string and replace escaped newlines
        value_str = repr(v).replace("\\n", "\n")
        # Step 2: Replace multiple spaces with ', '
        value_with_commas = re.sub(r"\ {2,}", ", ", value_str)
        # Step 3: Replace 'inf ' with 'inf,'
        cleaned_value = re.sub(r"inf ", "inf,", value_with_commas)
        config[k] = cleaned_value

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
    constructor_params = module.get_ctor_args_and_kwargs()
    return {"args": constructor_params[0], **constructor_params[1]}


def _get_model_architecture(module: TorchRLModule) -> dict:
    arch = {}
    torch_model = getattr(module, "model", module)
    arch["summary_str"] = str(torch_model)
    arch["summary_json"] = _jsonify_summary(arch["summary_str"])
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
