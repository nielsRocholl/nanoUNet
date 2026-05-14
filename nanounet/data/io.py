"""SimpleITK reader/writer: nnU-Net channel stacking, ``sitk_stuff`` in properties, plus I/O class resolution."""

from __future__ import annotations

import traceback
from typing import List, Sequence, Tuple, Type

import numpy as np
import pydoc
import SimpleITK as sitk

from nanounet.common import cprint

_SUPPORTED = (".nii.gz", ".nrrd", ".mha", ".gipl")


def _same_shape(shapes: list) -> bool:
    return all(s == shapes[0] for s in shapes[1:])


class SimpleITKIO:
    supported_file_endings = list(_SUPPORTED)

    def read_images(self, image_fnames: Sequence[str]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []
        sp_nn = []
        for f in image_fnames:
            itk = sitk.ReadImage(f)
            spacings.append(itk.GetSpacing())
            origins.append(itk.GetOrigin())
            directions.append(itk.GetDirection())
            npy = sitk.GetArrayFromImage(itk)
            if npy.ndim == 2:
                npy = npy[None, None]
                sp_nn.append((max(spacings[-1]) * 999, *list(spacings[-1])[::-1]))
            elif npy.ndim == 3:
                npy = npy[None]
                sp_nn.append(list(spacings[-1])[::-1])
            elif npy.ndim == 4:
                sp_nn.append(list(spacings[-1])[::-1][1:])
            else:
                raise RuntimeError(npy.ndim)
            images.append(npy)
            sp_nn[-1] = list(np.abs(sp_nn[-1]))
        shapes = [i.shape for i in images]
        if not _same_shape(shapes):
            raise RuntimeError(f"shape mismatch {shapes} {image_fnames}")
        for i in range(1, len(spacings)):
            if not np.allclose(spacings[0], spacings[i]):
                raise RuntimeError(f"spacing mismatch {image_fnames}")
        props = {
            "sitk_stuff": {"spacing": spacings[0], "origin": origins[0], "direction": directions[0]},
            "spacing": sp_nn[0],
        }
        return np.vstack(images, dtype=np.float32, casting="unsafe"), props

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3
        od = len(properties["sitk_stuff"]["spacing"])
        assert 1 < od < 4
        if od == 2:
            seg = seg[0]
        itk = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.int16, copy=False))
        itk.SetSpacing(properties["sitk_stuff"]["spacing"])
        itk.SetOrigin(properties["sitk_stuff"]["origin"])
        itk.SetDirection(properties["sitk_stuff"]["direction"])
        sitk.WriteImage(itk, output_fname, True)


def read_images(rw_class: Type[SimpleITKIO], files: Sequence[str]) -> Tuple[np.ndarray, dict]:
    return rw_class().read_images([*files])


def read_seg(rw_class: Type[SimpleITKIO], seg_file: str):
    return rw_class().read_seg(seg_file)


def write_seg(rw_class: Type[SimpleITKIO], seg: np.ndarray, out_trunc: str, props: dict) -> None:
    rw_class().write_seg(seg, out_trunc, props)


def reader_writer_class_from_dataset(dataset_json: dict, example_file: str | None, verbose: bool = True) -> Type[SimpleITKIO]:
    ow = dataset_json.get("overwrite_image_reader_writer")
    if ow and str(ow) != "None":
        o = str(ow)
        cls = pydoc.locate(o)
        # dataset.json often has bare "SimpleITKIO" or nnU-Net's dotted path; pydoc only resolves fully qualified names.
        if cls is None and (
            o == "SimpleITKIO"
            or o.endswith(".SimpleITKIO")
            or "simpleitk_reader_writer.SimpleITKIO" in o
        ):
            cls = SimpleITKIO
        if cls is None:
            raise RuntimeError(o)
        if verbose:
            cprint(f"[dim]Using {cls} reader/writer[/dim]")
        return cls
    fe = dataset_json["file_ending"].lower()
    for rw in (SimpleITKIO,):
        if fe in [e.lower() for e in rw.supported_file_endings]:
            if example_file is not None:
                try:
                    rw().read_images((example_file,))
                except Exception:
                    if verbose:
                        traceback.print_exc()
                    raise RuntimeError(example_file)
            if verbose:
                cprint(f"[dim]Using {rw} as reader/writer[/dim]")
            return rw
    for rw in (SimpleITKIO,):
        if example_file is None:
            continue
        try:
            rw().read_images((example_file,))
            if verbose:
                cprint(f"[dim]Using {rw} as reader/writer[/dim]")
            return rw
        except Exception:
            if verbose:
                traceback.print_exc()
    raise RuntimeError(f"no reader for {fe} {example_file}")
