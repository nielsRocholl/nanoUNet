"""Label dict → training heads, ignore label, foreground sets. Regions unsupported."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image


def _softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def filter_background(classes_or_regions: List):
    return [
        i
        for i in classes_or_regions
        if ((not isinstance(i, (tuple, list))) and i != 0)
        or (isinstance(i, (tuple, list)) and not (len(np.unique(i)) == 1 and np.unique(i)[0] == 0))
    ]


class Labels:
    """Single-task segmentation (no region heads). Mirrors nnU-Net LabelManager subset."""

    def __init__(self, label_dict: dict):
        if "background" not in label_dict:
            raise RuntimeError("labels need 'background' = 0")
        if isinstance(label_dict["background"], (tuple, list)):
            raise RuntimeError("background must be scalar 0")
        assert int(label_dict["background"]) == 0
        for _k, r in label_dict.items():
            if _k == "ignore":
                continue
            if isinstance(r, (tuple, list)) and len(r) > 1:
                raise NotImplementedError("region labels not supported in nanoUNet")
        self.label_dict = label_dict
        ig = label_dict.get("ignore")
        self._ignore_label: int | None = int(ig) if ig is not None else None
        self._all_labels = _collect_labels(label_dict)
        if self._ignore_label is not None:
            assert self._ignore_label == max(self._all_labels) + 1

    @property
    def has_ignore_label(self) -> bool:
        return self._ignore_label is not None

    @property
    def ignore_label(self) -> int | None:
        return self._ignore_label

    @property
    def all_labels(self) -> List[int]:
        return self._all_labels

    @property
    def has_regions(self) -> bool:
        return False

    @property
    def foreground_labels(self):
        return filter_background(self._all_labels)

    @property
    def foreground_regions(self):
        return None

    @property
    def num_segmentation_heads(self) -> int:
        return len(self._all_labels)

    @property
    def annotated_classes_key(self):
        return tuple([-1] + list(self._all_labels))

    def _infer_nonlin(self, logits: torch.Tensor) -> torch.Tensor:
        return _softmax_dim1(logits.float())

    @torch.inference_mode()
    def convert_logits_to_segmentation(self, predicted_logits: np.ndarray | torch.Tensor):
        if isinstance(predicted_logits, np.ndarray):
            logits_t = torch.from_numpy(predicted_logits)
        else:
            logits_t = predicted_logits
        probs = self._infer_nonlin(logits_t)
        if isinstance(predicted_logits, np.ndarray):
            return probs.argmax(0).cpu().numpy()
        return probs.argmax(0)

    @torch.inference_mode()
    def convert_probabilities_to_segmentation(self, predicted_probabilities: np.ndarray | torch.Tensor):
        if isinstance(predicted_probabilities, torch.Tensor):
            pp = predicted_probabilities.float()
            if pp.device.type != "cpu":
                return pp.argmax(0).cpu().numpy()
            return pp.argmax(0).numpy()
        return predicted_probabilities.argmax(0)

    def revert_cropping_on_probabilities(self, predicted_probabilities, bbox: List[List[int]], original_shape):
        if isinstance(predicted_probabilities, torch.Tensor):
            pp = predicted_probabilities.cpu().numpy()
        else:
            pp = predicted_probabilities
        out = np.zeros((pp.shape[0], *original_shape), dtype=pp.dtype)
        out[0] = 1
        return insert_crop_into_image(out, pp, bbox)


def _collect_labels(label_dict: dict) -> List[int]:
    xs = []
    for k, r in label_dict.items():
        if k == "ignore":
            continue
        xs.append(int(r))
    xs = list(np.unique(xs))
    xs.sort()
    return xs


def labels_from_dataset_json(dataset_json: dict) -> Labels:
    return Labels(dataset_json["labels"])
