"""Load plans.json: inheritance, 3d_fullres Config3d, resampling partials, old-UNet format migration."""

from __future__ import annotations

from copy import deepcopy
from functools import cached_property, partial
from typing import Callable, Union

from batchgenerators.utilities.file_and_folder_operations import load_json
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nanounet.data.resampling import resample_data_or_seg_to_shape
from nanounet.plan.labels import Labels, labels_from_dataset_json


def _bind_resample(cfg: dict, key: str) -> Callable:
    name = cfg[key]
    if isinstance(name, str) and "." in name:
        name = name.split(".")[-1]
    if name != "resample_data_or_seg_to_shape":
        raise NotImplementedError(name)
    return partial(resample_data_or_seg_to_shape, **dict(cfg[key + "_kwargs"]))


class Config3d:
    """One nnU-Net configuration block after `inherits_from` merge."""

    def __init__(self, configuration_dict: dict):
        d = deepcopy(configuration_dict)
        if "architecture" not in d:
            d = _migrate_old_architecture(d)
        self.configuration = d

    @property
    def data_identifier(self) -> str:
        return self.configuration["data_identifier"]

    @property
    def batch_size(self) -> int:
        return self.configuration["batch_size"]

    @property
    def patch_size(self) -> list[int]:
        return self.configuration["patch_size"]

    @property
    def spacing(self) -> list[float]:
        return self.configuration["spacing"]

    @property
    def normalization_schemes(self):
        return self.configuration["normalization_schemes"]

    @property
    def use_mask_for_norm(self):
        return self.configuration["use_mask_for_norm"]

    @property
    def network_arch_class_name(self) -> str:
        return self.configuration["architecture"]["network_class_name"]

    @property
    def network_arch_init_kwargs(self) -> dict:
        return self.configuration["architecture"]["arch_kwargs"]

    @property
    def network_arch_init_kwargs_req_import(self):
        return self.configuration["architecture"]["_kw_requires_import"]

    @property
    def batch_dice(self) -> bool:
        return self.configuration["batch_dice"]

    @property
    def pool_op_kernel_sizes(self):
        return tuple(tuple(x) for x in self.configuration["architecture"]["arch_kwargs"]["strides"])

    @property
    def previous_stage_name(self):
        return self.configuration.get("previous_stage")

    @cached_property
    def resampling_fn_data(self) -> Callable:
        return _bind_resample(self.configuration, "resampling_fn_data")

    @cached_property
    def resampling_fn_probabilities(self) -> Callable:
        return _bind_resample(self.configuration, "resampling_fn_probabilities")

    @cached_property
    def resampling_fn_seg(self) -> Callable:
        return _bind_resample(self.configuration, "resampling_fn_seg")


class Plans:
    def __init__(self, plans_file_or_dict: Union[str, dict]):
        self.plans = plans_file_or_dict if isinstance(plans_file_or_dict, dict) else load_json(plans_file_or_dict)

    def _resolve(self, configuration_name: str, visited: tuple[str, ...] | None = None) -> dict:
        cfgs = self.plans["configurations"]
        if configuration_name not in cfgs:
            raise KeyError(f"{configuration_name!r} not in plans")
        configuration = deepcopy(cfgs[configuration_name])
        if "inherits_from" not in configuration:
            return configuration
        parent = configuration.pop("inherits_from")
        if visited is None:
            visited = (configuration_name,)
        else:
            if parent in visited:
                raise RuntimeError("circular inherits_from")
            visited = (*visited, configuration_name)
        base = self._resolve(parent, visited)
        base.update(configuration)
        return base

    def get_configuration(self, name: str = "3d_fullres") -> Config3d:
        return Config3d(self._resolve(name))

    @property
    def transpose_forward(self) -> list[int]:
        return self.plans["transpose_forward"]

    @property
    def transpose_backward(self) -> list[int]:
        return self.plans["transpose_backward"]

    @property
    def image_reader_writer(self) -> str:
        return self.plans["image_reader_writer"]

    def get_label_manager(self, dataset_json: dict) -> Labels:
        return labels_from_dataset_json(dataset_json)

    @property
    def foreground_intensity_properties_per_channel(self) -> dict:
        p = self.plans
        if "foreground_intensity_properties_per_channel" in p:
            return p["foreground_intensity_properties_per_channel"]
        return p["foreground_intensity_properties_by_modality"]


def _migrate_old_architecture(configuration: dict) -> dict:
    u = configuration["UNet_class_name"]
    if u == "PlainConvUNet":
        net_cls = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        key_blk = "n_conv_per_stage"
    elif u == "ResidualEncoderUNet":
        net_cls = "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet"
        key_blk = "n_blocks_per_stage"
    else:
        raise RuntimeError(u)
    n_stg = len(configuration["n_conv_per_stage_encoder"])
    dim = len(configuration["patch_size"])
    cop = convert_dim_to_conv_op(dim)
    inst = get_matching_instancenorm(dimension=dim)
    arch = {
        "network_class_name": net_cls,
        "arch_kwargs": {
            "n_stages": n_stg,
            "features_per_stage": [
                min(configuration["UNet_base_num_features"] * 2**i, configuration["unet_max_num_features"])
                for i in range(n_stg)
            ],
            "conv_op": cop.__module__ + "." + cop.__name__,
            "kernel_sizes": deepcopy(configuration["conv_kernel_sizes"]),
            "strides": deepcopy(configuration["pool_op_kernel_sizes"]),
            key_blk: deepcopy(configuration["n_conv_per_stage_encoder"]),
            "n_conv_per_stage_decoder": deepcopy(configuration["n_conv_per_stage_decoder"]),
            "conv_bias": True,
            "norm_op": inst.__module__ + "." + inst.__name__,
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }
    for k in (
        "UNet_class_name",
        "UNet_base_num_features",
        "n_conv_per_stage_encoder",
        "n_conv_per_stage_decoder",
        "num_pool_per_axis",
        "pool_op_kernel_sizes",
        "conv_kernel_sizes",
        "unet_max_num_features",
    ):
        del configuration[k]
    configuration["architecture"] = arch
    return configuration


def determine_num_input_channels(cm: Config3d, dataset_json: dict) -> int:
    n_mod = len(dataset_json["modality"]) if "modality" in dataset_json else len(dataset_json["channel_names"])
    if cm.previous_stage_name is not None:
        raise NotImplementedError("cascade input channels")
    return n_mod
