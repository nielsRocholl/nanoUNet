"""Blosc2 preprocessed cases: identifiers, load_case, optional centroids JSON merge."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from typing import Tuple, Union

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_pickle, write_pickle

class Blosc2Folder:
    def __init__(self, folder: str, identifiers: list[str] | None = None, folder_with_segs_from_previous_stage: str | None = None):
        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.identifiers = sorted(identifiers) if identifiers is not None else self.get_identifiers(folder)
        blosc2.set_nthreads(1)

    def load_case(self, identifier: str):
        dparams = {"nthreads": 1}
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
        data = blosc2.open(urlpath=join(self.source_folder, identifier + ".b2nd"), mode="r", dparams=dparams, **mmap_kwargs)
        seg = blosc2.open(urlpath=join(self.source_folder, identifier + "_seg.b2nd"), mode="r", dparams=dparams, **mmap_kwargs)
        if self.folder_with_segs_from_previous_stage is not None:
            seg_prev = blosc2.open(
                urlpath=join(self.folder_with_segs_from_previous_stage, identifier + ".b2nd"),
                mode="r",
                dparams=dparams,
                **mmap_kwargs,
            )
        else:
            seg_prev = None
        properties = load_pickle(join(self.source_folder, identifier + ".pkl"))
        cj = join(self.source_folder, identifier + "_centroids.json")
        if isfile(cj):
            properties = {**properties, **json.load(open(cj, encoding="utf-8"))}
        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(data: np.ndarray, seg: np.ndarray, properties: dict, output_filename_truncated: str, chunks=None, blocks=None, chunks_seg=None, blocks_seg=None, clevel: int = 8, codec=blosc2.Codec.ZSTD):
        blosc2.set_nthreads(1)
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks
        # blosc2.asarray refuses an existing urlpath; drop any partial outputs from a killed prior worker.
        for s in (".b2nd", "_seg.b2nd", ".pkl"):
            p = output_filename_truncated + s
            if os.path.isfile(p):
                os.remove(p)
        cparams = {"codec": codec, "clevel": clevel}
        blosc2.asarray(np.ascontiguousarray(data), urlpath=output_filename_truncated + ".b2nd", chunks=chunks, blocks=blocks, cparams=cparams)
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated + "_seg.b2nd", chunks=chunks_seg, blocks=blocks_seg, cparams=cparams)
        write_pickle(properties, output_filename_truncated + ".pkl")

    @staticmethod
    def comp_blosc2_params(
        image_size: Tuple[int, int, int, int],
        patch_size: Union[Tuple[int, int, int], Tuple[int, int, int]],
        bytes_per_pixel: int = 4,
        l1_cache_size_per_core_in_bytes: int = 32768,
        l3_cache_size_per_core_in_bytes: int = 1441792,
        safety_factor: float = 0.8,
    ):
        num_channels = image_size[0]
        assert len(patch_size) == 3
        ps = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** max(0, math.ceil(math.log2(i))) for i in ps]))
        est_blk = np.prod(block_size) * bytes_per_pixel
        while est_blk > l1_cache_size_per_core_in_bytes * safety_factor:
            axis_order = np.argsort(block_size[1:] / ps)[::-1]
            idx, picked = 0, int(axis_order[0])
            while block_size[picked + 1] == 1:
                idx += 1
                picked = int(axis_order[idx])
            block_size[picked + 1] = 2 ** max(0, math.floor(math.log2(block_size[picked + 1] - 1)))
            block_size[picked + 1] = min(block_size[picked + 1], image_size[picked + 1])
            est_blk = np.prod(block_size) * bytes_per_pixel
        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])
        chunk_size = deepcopy(block_size)
        est_chunk = np.prod(chunk_size) * bytes_per_pixel
        while est_chunk < l3_cache_size_per_core_in_bytes * safety_factor:
            if ps[0] == 1 and all(i == j for i, j in zip(chunk_size[2:], image_size[2:])):
                break
            if all(i == j for i, j in zip(chunk_size, image_size)):
                break
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx, picked = 0, int(axis_order[0])
            while chunk_size[picked + 1] == image_size[picked + 1] or ps[picked] == 1:
                idx += 1
                picked = int(axis_order[idx])
            chunk_size[picked + 1] += block_size[picked + 1]
            chunk_size[picked + 1] = min(chunk_size[picked + 1], image_size[picked + 1])
            est_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean([i / j for i, j in zip(chunk_size[1:], ps)]) > 1.5:
                chunk_size[picked + 1] -= block_size[picked + 1]
                break
        chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]
        return tuple(block_size), tuple(chunk_size)

    @staticmethod
    def get_identifiers(folder: str) -> list[str]:
        return sorted([i[:-5] for i in os.listdir(folder) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")])
