# SPDX-License-Identifier: MIT
"""Streaming RGB PNG-to-NPZ conversion for ADM evaluation."""
import os
from pathlib import Path
import zipfile


def pack_samples(sample_dir, num_samples=50000, output_file=None):
    import numpy as np
    from PIL import Image
    if num_samples < 1:
        raise ValueError('num_samples must be positive')
    folder = Path(sample_dir)
    target = Path(output_file) if output_file is not None else Path(str(folder) + '.npz')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(target) + '.partial')
    shape = None
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        with archive.open('arr_0.npy', 'w', force_zip64=True) as stream:
            for index in range(num_samples):
                with Image.open(folder / f'{index:06d}.png') as image:
                    frame = np.asarray(image, dtype=np.uint8)
                if shape is None:
                    shape = frame.shape
                    if len(shape) != 3 or shape[-1] != 3:
                        raise ValueError('Expected RGB PNG samples')
                    np.lib.format.write_array_header_1_0(stream, {
                        'descr': '|u1', 'fortran_order': False, 'shape': (num_samples, *shape),
                    })
                if frame.shape != shape:
                    raise ValueError('Inconsistent PNG shape at image ' + str(index))
                stream.write(frame.tobytes())
    os.replace(partial, target)
    return target
