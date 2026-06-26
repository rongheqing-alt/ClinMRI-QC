"""Functions to detect gadolinium contrast enhancement from a single T1w image.

Post-gadolinium T1w images show bright vessels due to T1 shortening.
Detection relies on the high-intensity tail of the brain voxel distribution:
vessels appear as a cluster of voxels well above the white matter peak.
"""

# %%
import numpy as np
import argparse
import json
from quickbrain.general import load_nifti

def compute_vessel_intensity_ratio(image: np.ndarray, brain_mask: np.ndarray) -> float:
    """Ratio of the extreme-bright tail (99.9th percentile) to the bulk tissue (50th percentile) intensity.
    Args:
        image: 3D T1w image array.
        brain_mask: Boolean mask of the brain.

    Returns:
        Ratio of the 99.9th percentile to the 50th percentile within the brain.
    """
    voxels = image[brain_mask]
    p50 = np.percentile(voxels, 50)
    p999 = np.percentile(voxels, 99.9)
    if p50 == 0:
        return np.inf
    return float(p999 / p50)


def compute_bright_voxel_fraction(
    image: np.ndarray,
    brain_mask: np.ndarray,
    sigma_factor: float = 3.0,
) -> float:
    """Fraction of brain voxels that are abnormally bright (likely vessels).
    Threshold is set at mean + sigma_factor * std of the brain intensity.

    Args:
        image: 3D T1w image array.
        brain_mask: Boolean mask of the brain.
        sigma_factor: Number of standard deviations above the mean to use as
            the vessel brightness threshold.

    Returns:
        Fraction of brain voxels above the threshold (between 0 and 1).
    """
    voxels = image[brain_mask]
    threshold = voxels.mean() + sigma_factor * voxels.std()
    return float((voxels > threshold).sum() / len(voxels))


def detect_contrast_enhancement(
    image: np.ndarray,
    brain_mask: np.ndarray,
    vessel_ratio_threshold: float = 1.6,
    bright_fraction_threshold: float = 0.002,
) -> dict:
    """Detect gadolinium contrast enhancement from a single T1w image.

    Combines two markers of vessel brightness:
      - vessel_ratio: 99th/50th percentile ratio (high → bright vessels).
      - bright_fraction: fraction of very-bright brain voxels (well above tissue mean).

    Args:
        image: 3D T1w image array.
        brain_mask: Boolean mask of the brain (excludes skull and background).
        vessel_ratio_threshold: Minimum p99/p50 ratio to flag bright vessels.
            Typical native T1w ≈ 1.2–1.4; post-gad ≈ 1.6–2.0.
        bright_fraction_threshold: Minimum fraction of very-bright voxels.
            Vessels occupy ~0.3–1 % of the brain volume after contrast.

    Returns:
        Dictionary with:
            - "enhanced": bool, True if both markers exceed their thresholds.
            - "vessel_ratio": float, p99/p50 intensity ratio.
            - "bright_voxel_fraction": float, fraction of very-bright voxels.
    """
    vessel_ratio = compute_vessel_intensity_ratio(image, brain_mask)
    bright_fraction = compute_bright_voxel_fraction(image, brain_mask)

    enhanced = (vessel_ratio >= vessel_ratio_threshold) and (
        bright_fraction >= bright_fraction_threshold
    )

    return {
        "enhanced": enhanced,
        "vessel_ratio": vessel_ratio,
        "bright_voxel_fraction": bright_fraction,
    }

# %%
def main():
# %%
    parser = argparse.ArgumentParser(description="ClinMRI-QC: run full QC pipeline")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--brain_mask", required=True, help="Path to brain mask NIfTI image")
    parser.add_argument("--outfile", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()
    
    # load nifti file
    image_arr, _= load_nifti(args.image)
    mask_arr, _= load_nifti(args.brain_mask).astype(bool)

    results = detect_contrast_enhancement(image_arr, mask_arr)
    output = json.dumps(results, indent=2)
    print(output)

    
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(output)
        print(f"\nResults saved to {args.outfile}")


# %%
