"""
Image Quality Metrics (IQMs) for structural MRI.

EFC and SNR are re-implemented from the MRIQC formulae
(Esteban et al. 2017, PLOS ONE) using only numpy/scipy so that the
package has no dependency on the full MRIQC / nipype stack.

Outputs:
    motion_blur_score : EFC (Entropy Focus Criterion) — lower = more motion/blur
    snr               : Signal-to-Noise Ratio — lower = noisier image

A scipy-based Otsu fallback is provided when no brain mask is supplied.
"""

import numpy as np
from math import sqrt
from scipy import ndimage


# ---------------------------------------------------------------------------
# Brain mask fallback
# ---------------------------------------------------------------------------

def compute_head_mask(img: np.ndarray) -> np.ndarray:
    """
    Estimate a foreground (head/brain) mask via Otsu thresholding and
    morphological clean-up. Used when no brain mask is supplied.

    Parameters
    ----------
    img : np.ndarray
        3-D image array (H, W, D), float or int.

    Returns
    -------
    np.ndarray
        Boolean mask, same shape as img.
    """
    positive = img[img > 0]
    if positive.size == 0:
        return np.zeros_like(img, dtype=bool)

    threshold = np.percentile(positive, 15)
    mask = img > threshold
    mask = ndimage.binary_closing(mask, iterations=5)
    mask = ndimage.binary_fill_holes(mask)

    # Keep only the largest connected component
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    mask = labeled == sizes.argmax()

    return mask.astype(bool)


# ---------------------------------------------------------------------------
# IQM implementations
# ---------------------------------------------------------------------------

def efc(img: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Entropy Focus Criterion (Atkinson 1997).

    Measures ghosting and blurring caused by head motion via Shannon entropy
    of voxel intensities. Lower values indicate better quality.

    Parameters
    ----------
    img  : 3-D image array
    mask : optional boolean mask — if provided, only masked voxels are used
    """
    data = np.abs(img[mask] if mask is not None else img.ravel()).astype(np.float64)

    n_vox = data.size
    efc_max = n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))

    b_max = np.sqrt((data ** 2).sum())
    if b_max < 1e-10:
        return 0.0

    return float(
        (1.0 / efc_max)
        * np.sum((data / b_max) * np.log((data + 1e-16) / b_max))
    )



def snr(img: np.ndarray, head_mask: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio estimated within the foreground mask.

    SNR = μ_fg / (σ_fg * sqrt(n / (n-1)))

    Parameters
    ----------
    img       : 3-D image array
    head_mask : boolean mask of the head/brain region
    """
    fg = img[head_mask].astype(np.float64)
    n = fg.size
    if n < 2:
        return 0.0
    mu = fg.mean()
    sigma = fg.std()
    if sigma < 1e-10:
        return 0.0
    return float(mu / (sigma * sqrt(n / (n - 1))))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_iqms(img: np.ndarray, brain_mask: np.ndarray = None) -> dict:
    """
    Compute image quality metrics for a 3-D MRI volume.

    Parameters
    ----------
    img        : np.ndarray, shape (H, W, D)
    brain_mask : optional boolean array, same shape as img.
                 If None, a head mask is estimated via Otsu thresholding.

    Returns
    -------
    dict with keys:
        'motion_blur_score' : float — EFC; lower values indicate more motion/blurring
        'snr'               : float — signal-to-noise ratio; lower values indicate more noise
    """
    img = img.astype(np.float64)

    if brain_mask is None:
        brain_mask = compute_head_mask(img)

    return {
        'motion_blur_score': round(efc(img, mask=brain_mask), 4),
        'snr':               round(snr(img, brain_mask),      4),
    }
