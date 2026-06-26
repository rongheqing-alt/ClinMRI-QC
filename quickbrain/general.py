import json
import os
import tempfile
from pathlib import Path

import nibabel
import numpy as np

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "default.json"


def load_config(path: str | Path | None = None) -> dict:
    """Load QC thresholds from a JSON config file.

    Args:
        path: Path to a JSON config file. Defaults to config/default.json.

    Returns:
        Dictionary of thresholds and parameters.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    with open(config_path) as f:
        return json.load(f)

def load_nifti (path:str) -> np.ndarray: 
    img = nibabel.load(path)
    img = nibabel.as_closest_canonical(img)  # force RAS orientation
    data = np.asanyarray(img.dataobj, dtype = np.float32)
    return data, img


import nibabel as nib
from scipy.ndimage import zoom

def get_brain_mask(path: str, outfile=None) -> np.ndarray:
    import brainchop as bc
    from brainchop.cli import _save_inverse_conform

    vol = bc.load(path)
    brain_mask = bc.segment(vol, "mindgrab")
 
    if outfile is None:
        outfile = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    
    _save_inverse_conform(brain_mask, path, outfile.name)
    
    mask, _ = load_nifti(outfile.name)

    return (mask>0).astype(bool)

