"""Scan-level MRI artifact severity detection via a pretrained ResNet50 regression model.

Six independent sigmoid heads predict artifact severity [0, 1] per class.
Severity semantics: 0.0 = absent, 0.33 = mild, 0.67 = moderate, 1.0 = severe.

Example
-------
    from clinmriqc.artifacts import detect_artifacts
    from clinmriqc.general import load_nifti

    image = load_nifti("T1w.nii.gz")
    mask  = load_nifti("brain_mask.nii.gz").astype(bool)
    result = detect_artifacts(image, mask)

Output schema
-------------
{
    "quality_passed":     bool,
    "artifacts_detected": ["noise", "bias_field"],   # empty if all severities < threshold
    "artifact_severity":  {
        "motion":     0.02,   # 0.0=absent  0.33=mild  0.67=moderate  1.0=severe
        "noise":      0.67,
        "ghosting":   0.01,
        "bias_field": 0.38,
        "gibbs":      0.00,
        "zipper":     0.01,
    },
    "iqms": {
        "motion_blur_score": 0.62,   # EFC — lower = more blur/motion
        "snr":               18.3,   # signal-to-noise ratio
    },
}
"""

import argparse
import json
import numpy as np
from pathlib import Path

from clinmriqc.general import load_nifti
try:
    from clinmriqc.general import get_brain_mask
except ImportError:
    get_brain_mask = None
from clinmriqc.iqm.metrics import compute_iqms
from clinmriqc.classifier.model import load_regression_model, predict_volume

_DEFAULT_MODEL = Path(__file__).parent / "classifier" / "best_regression_model.pt"

# Raw severity thresholds (in model output space [0,1]).
# Calibrated at the 99th percentile across 937 scans (MS patients n=610,
# Mindboggle healthy n=101, AOMIC healthy n=226); bias_field lowered to P97
# because the model systematically scores all brain MRI high on inhomogeneity.
DEFAULT_THRESHOLDS = {
    'motion':     0.1500,   # raised from 0.0956 (P99) — reduces false positives on clean scans
    'noise':      0.3800,   # raised from 0.3269
    'ghosting':   0.4500,   # raised from 0.3456
    'bias_field': 0.8818,   # P97 — P99 (0.9304) flags too few genuine outliers
    'gibbs':      0.1000,   # raised from 0.0568 — previous value flagged borderline clean scans
    'zipper':     0.0305,
}

# Scale ceilings: max(dataset_max, severe_synthetic_max) across 937 real scans
# and 5-image × 6-corruption synthetic benchmark.  Dividing raw scores by these
# gives a [0, 1] display value where 1.0 = worst observed/simulated artifact.
SEVERITY_SCALE = {
    'motion':     0.3339,
    'noise':      0.5238,
    'ghosting':   0.7841,
    'bias_field': 0.9552,
    'gibbs':      0.9950,
    'zipper':     0.0660,
}

# Thresholds expressed in the same scaled [0,1] space (for display in reports).
SCALED_THRESHOLDS = {
    cls: round(DEFAULT_THRESHOLDS[cls] / SEVERITY_SCALE[cls], 3)
    for cls in DEFAULT_THRESHOLDS
}


def detect_artifacts(
    image: np.ndarray,
    brain_mask: np.ndarray = None,
    model_path: str = None,
    model=None,
    class_thresholds: dict = None,
    device: str = None,
) -> dict:
    """Detect MRI artifacts and compute image quality metrics.

    Parameters
    ----------
    image            : 3-D float array (H, W, D) — raw voxel intensities.
    brain_mask       : optional boolean array, same shape as image — brain region only.
                       If None, IQMs fall back to Otsu-based head masking.
    model_path       : path to a trained regression checkpoint (.pt).
                       Defaults to the bundled best_regression_model.pt.
                       Ignored if `model` is provided.
    model            : pre-loaded regression model (nn.Module). Pass this in batch
                       pipelines to avoid reloading the 90 MB checkpoint per patient.
    class_thresholds : optional per-class overrides, e.g. {'bias_field': 0.90}.
                       Merged on top of DEFAULT_THRESHOLDS — only specified classes change.
    device           : 'cpu' or 'cuda'. Auto-detected if not specified.

    Returns
    -------
    dict — see module docstring for full output schema.
    """
    if model is None:
        model_path = model_path or str(_DEFAULT_MODEL)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Train the regression model first (train_regression.py)."
            )
        model = load_regression_model(model_path, device=device)

    image = np.asarray(image, dtype=np.float32)
    if brain_mask is not None:
        brain_mask = np.asarray(brain_mask, dtype=bool)

    iqms = compute_iqms(image, brain_mask=brain_mask)

    # Merge caller overrides on top of calibrated defaults
    thresholds = dict(DEFAULT_THRESHOLDS)
    if class_thresholds:
        thresholds.update(class_thresholds)

    result = predict_volume(model, image, class_thresholds=thresholds, device=device)

    raw = result["artifact_severity"]
    scaled = {
        cls: round(min(float(v) / SEVERITY_SCALE[cls], 1.0), 4)
        for cls, v in raw.items()
    }

    return {
        "quality_passed":          result["quality_passed"],
        "artifacts_detected":      result["artifacts_detected"],
        "artifact_severity":       raw,
        "artifact_severity_scaled": scaled,
        "iqms":                    iqms,
    }


def main():
    parser = argparse.ArgumentParser(description="ClinMRI-QC: detect MRI artifact severity")
    parser.add_argument("--image",      required=True, help="Path to T1w NIfTI image")
    parser.add_argument("--brain_mask", required=False, default=None,
                        help="Path to brain mask NIfTI (optional — auto-generated via brainchop if omitted)")
    parser.add_argument("--model",     default=None,
                        help="Path to regression checkpoint (optional)")
    parser.add_argument("--threshold_motion",     type=float, default=None,
                        help=f"Override motion threshold  (default {DEFAULT_THRESHOLDS['motion']})")
    parser.add_argument("--threshold_noise",      type=float, default=None,
                        help=f"Override noise threshold  (default {DEFAULT_THRESHOLDS['noise']})")
    parser.add_argument("--threshold_ghosting",   type=float, default=None,
                        help=f"Override ghosting threshold  (default {DEFAULT_THRESHOLDS['ghosting']})")
    parser.add_argument("--threshold_bias_field", type=float, default=None,
                        help=f"Override bias_field threshold  (default {DEFAULT_THRESHOLDS['bias_field']})")
    parser.add_argument("--threshold_gibbs",      type=float, default=None,
                        help=f"Override gibbs threshold  (default {DEFAULT_THRESHOLDS['gibbs']})")
    parser.add_argument("--threshold_zipper",     type=float, default=None,
                        help=f"Override zipper threshold  (default {DEFAULT_THRESHOLDS['zipper']})")
    parser.add_argument("--outfile", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    class_thresholds = {k: v for k, v in {
        'motion':     args.threshold_motion,
        'noise':      args.threshold_noise,
        'ghosting':   args.threshold_ghosting,
        'bias_field': args.threshold_bias_field,
        'gibbs':      args.threshold_gibbs,
        'zipper':     args.threshold_zipper,
    }.items() if v is not None}

    image_arr = load_nifti(args.image)
    if args.brain_mask:
        mask_arr = load_nifti(args.brain_mask).astype(bool)
    else:
        try:
            print("No brain mask supplied — generating via brainchop...")
            mask_arr = get_brain_mask(args.image)
            print("Brain mask generated.")
        except Exception as e:
            print(f"Warning: brainchop unavailable ({e}), falling back to Otsu masking.")
            mask_arr = None

    results = detect_artifacts(image_arr, mask_arr, model_path=args.model,
                               class_thresholds=class_thresholds or None)
    output = json.dumps(results, indent=2)
    print(output)

    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(output)
        print(f"\nResults saved to {args.outfile}")
