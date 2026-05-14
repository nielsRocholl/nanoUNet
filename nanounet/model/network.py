"""Instantiate ResidualEncoderUNet (optional extra input channels for prompts) from plans arch block."""

from __future__ import annotations

import importlib
import pydoc

from copy import deepcopy

import torch

from nanounet.plan.labels import Labels
from nanounet.plan.plans import Config3d, determine_num_input_channels


def _build_class(network_class: str, arch_kwargs: dict, req: list | tuple):
    kwargs = dict(arch_kwargs)
    for ri in req:
        if kwargs.get(ri) is not None:
            kwargs[ri] = pydoc.locate(kwargs[ri])
    mod_name, _, cls_name = network_class.rpartition(".")
    try:
        nw = getattr(importlib.import_module(mod_name), cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(network_class) from e
    return nw, kwargs


def build_net(
    cm: Config3d,
    lm: Labels,
    dataset_json: dict,
    enable_deep_supervision: bool,
    n_extra_in: int = 2,
    num_classes_override: int | None = None,
):
    n_in = determine_num_input_channels(cm, dataset_json)
    nw, kwargs = _build_class(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
    )
    kwargs = dict(kwargs)
    kwargs["deep_supervision"] = enable_deep_supervision
    nc = lm.num_segmentation_heads if num_classes_override is None else num_classes_override
    net = nw(input_channels=n_in + n_extra_in, num_classes=nc, **kwargs)
    if hasattr(net, "initialize"):
        net.apply(net.initialize)
    return net


def estimate_conv_feature_map_size(
    patch_size: tuple,
    n_in: int,
    n_out: int,
    network_class: str,
    arch_kwargs: dict,
    req: list | tuple,
    deep_supervision: bool = True,
) -> float:
    import os

    a = torch.get_num_threads()
    t = int(os.environ.get("nnUNet_n_proc_DA", "12"))
    t = max(1, min(t, os.cpu_count() or 1))
    torch.set_num_threads(t)
    kw = deepcopy(arch_kwargs)
    nw, k2 = _build_class(network_class, kw, req)
    k2 = dict(k2)
    k2["deep_supervision"] = deep_supervision
    net = nw(input_channels=n_in, num_classes=n_out, **k2)
    r = float(net.compute_conv_feature_map_size(patch_size))
    torch.set_num_threads(a)
    return r
