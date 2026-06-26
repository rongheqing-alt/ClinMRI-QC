"""Metadata and per-sample image-feature QC for clinical MRI.

Part of ClinMRI-QC. One flat module, matching the per-member layout
(artifacts.py, coreg.py, contrast.py). Per-sample only: each function takes one
image (optionally a brain mask) and returns a dict of named numbers plus
pass/warning/fail flags. No cohort or cross-subject logic lives here.

How to use
----------
    from quickbrain import metaqc

    # One image, no mask.
    metaqc.run_qc("study1_T1W.nii.gz")

    # One image with a brain mask (intensity stats restricted to the brain).
    metaqc.run_qc("study1_T1W.nii.gz", brain_mask_path="brainmask.nii.gz")

    # One image with brain and lesion masks (each mask QC'd separately).
    metaqc.run_qc("T1WKS.nii.gz",
                  brain_mask_path="brainmask.nii.gz",
                  lesion_mask_path="gt.nii.gz")

    # Are a subject's expected modalities/masks/timepoints present?
    files = glob.glob("patient01/*.nii.gz")
    metaqc.check_completeness(files, {"required_modalities": ["T1w", "FLAIR"]})

    # Command line (single image):
    #   python -m quickbrain.metaqc --image T1W.nii.gz --brain_mask brainmask.nii.gz

What the numbers mean
---------------------
Intensity statistics are computed over a foreground voxel set, not the whole
volume, because the MRI background is a large near-zero region that would skew
any whole-volume statistic.

  foreground:   with a brain_mask, foreground is the voxels inside it, and the
                stats are of the image within that mask. With no mask, foreground
                is voxels above the 10th percentile of the volume's own range.
                foreground_method records which rule was used.
  intensity_*:  statistics of the IMAGE over the foreground. These are image
                intensities, never the mask's own label values (a binary mask is
                all 1s, so its mean would just be 1.0). That is why the image and
                mask are passed separately.
  foreground_fraction: foreground voxels / total voxels. A small value (under
                ~0.05) usually means a near-empty or failed volume.
  centroid_offset_mm: distance from the intensity-weighted centre of the
                foreground to the geometric centre. A soft descriptor, not a hard
                verdict; a large value can indicate an off-centre or cropped brain.

Thresholds and config
----------------------
check_features takes a thresholds dict; defaults are in DEFAULT_THRESHOLDS.
Completeness and role detection take a config dict (DEFAULT_CONFIG) so the tool
works on datasets with different modality naming, mask roles, or timepoint
schemes. Pass overrides for any subset; everything is optional.

Every check returns {status, message} with status in {pass, warning, fail}, so
results compose into a merged report.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel

try:  # Prefer the shared loader if present (team convention).
    from quickbrain.general import load_nifti
except Exception:  # pragma: no cover - fallback so the file is standalone
    def load_nifti(path: str) -> np.ndarray:
        return np.asarray(nibabel.load(path).dataobj, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Configurable thresholds. Callers override any subset via the `thresholds` arg.
# --------------------------------------------------------------------------- #
DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Geometry (metadata QC)
    "min_voxel_size_mm": 0.1,       # below this, spacing is implausible
    "max_voxel_size_mm": 6.0,       # above this, spacing is implausibly coarse
    "max_anisotropy_ratio": 8.0,    # max(spacing)/min(spacing) above -> warn
    # Image features (per-sample QC)
    "min_foreground_fraction": 0.05,   # below -> likely empty/failed volume (fail)
    "warn_foreground_fraction": 0.10,  # below -> suspicious (warning)
    "min_intensity_std": 1e-6,         # ~0 -> constant image (fail)
    "max_centroid_offset_mm": 30.0,    # above -> brain far off-centre (warning)
}

_FOREGROUND_PERCENTILE = 10.0  # background-removal percentile when no mask given


# --------------------------------------------------------------------------- #
# Dataset configuration (user-defined, structure-agnostic)
# --------------------------------------------------------------------------- #
# The user declares what to expect; nothing is hard-coded to one dataset. Example:
#
#   config = {
#       "modalities": {                       # role -> filename patterns (any match)
#           "T1w":   ["t1w", "t1.nii", "_t1"],
#           "FLAIR": ["flair"],
#           "T1CE":  ["t1wks", "t1ce", "post"],
#           "T2w":   ["t2w", "t2."],
#       },
#       "masks": {                            # mask role -> patterns; any roles allowed
#           "brain":  ["brainmask", "brain_mask"],
#           "lesion": ["consensus", "lesion", "_gt"],
#       },
#       "required_modalities": ["T1w", "FLAIR"],   # subset that MUST be present
#       "required_masks": [],                       # mask roles that MUST be present
#       "timepoint_patterns": ["ses-", "study", "tp", "week", "month"],  # longitudinal
#       "expected_timepoints": None,           # int, or None to not enforce a count
#   }
#
# All fields optional; defaults below cover the common brain-MRI case.
DEFAULT_CONFIG: dict = {
    "modalities": {
        "T1w":   ["t1w", "t1.nii", "_t1.", "t1_"],
        "FLAIR": ["flair"],
        "T1CE":  ["t1wks", "t1ce", "t1post", "t1_post", "_post", "postgad", "gad", "ce.nii"],
        "T2w":   ["t2w", "t2.nii", "_t2.", "t2_"],
        "PD":    ["pd.nii", "_pd", "proton"],
        "DWI":   ["dwi", "diffusion"],
    },
    "masks": {
        "brain":  ["brainmask", "brain_mask", "brain-mask"],
        "lesion": ["consensus", "lesion", "_gt", "gt.nii", "ground_truth", "seg"],
    },
    "required_modalities": [],   # empty = report presence but never fail on absence
    "required_masks": [],
    "timepoint_patterns": ["ses-", "study", "_tp", "week", "month", "visit", "baseline", "followup"],
    "expected_timepoints": None,
}


def classify_file(filename: str, config: Optional[dict] = None) -> Tuple[str, Optional[str]]:
    """Classify a filename into (kind, role) using the config's patterns.

    kind is "modality", "mask", or "unknown"; role is the matched role name
    (e.g. "T1w", "brain") or None. Masks are checked before modalities so a
    'brainmask' is not mis-read as a T1.

    Modality matching is specificity-ordered: more specific roles (e.g. T1CE,
    whose patterns like 't1wks'/'t1ce' CONTAIN the generic 't1') are tested
    before generic ones (T1w), so 'T1WKS.nii.gz' is correctly read as T1CE, not
    T1w. Within the configured modalities, roles are tried in order of their
    longest matching pattern (longer = more specific) to avoid a short generic
    substring claiming a more specific filename.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    name = os.path.basename(filename).lower()

    # Masks first (a brain mask must not be read as an image).
    for role, patterns in cfg.get("masks", {}).items():
        if any(p in name for p in patterns):
            return "mask", role

    # Modalities, specificity-ordered: for each role, find its longest pattern
    # that matches; then pick the role whose matched pattern is longest. This
    # makes 't1wks' (T1CE) beat 't1w' (T1w) on a 'T1WKS' filename.
    best_role = None
    best_len = -1
    for role, patterns in cfg.get("modalities", {}).items():
        matched = [p for p in patterns if p in name]
        if matched:
            longest = max(len(p) for p in matched)
            if longest > best_len:
                best_len = longest
                best_role = role
    if best_role is not None:
        return "modality", best_role
    return "unknown", None


def detect_timepoint(path: str, config: Optional[dict] = None) -> Optional[str]:
    """Timepoint label from a path, or None if no longitudinal pattern is found.

    Timepoints may be a folder (``.../ses-01/...``) or a token inside the
    filename (``study1_FLAIR.nii.gz``). For each configured pattern, return the
    token it belongs to: the pattern plus any trailing digits, so
    ``study1_FLAIR`` gives ``study1``, not the whole filename.
    """
    import re
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    low = path.replace("\\", "/").lower()
    for pat in cfg.get("timepoint_patterns", []):
        idx = low.find(pat)
        if idx >= 0:
            # Capture the pattern plus trailing digits (study1, ses-01, week2...).
            m = re.match(rf"{re.escape(pat)}[-_]?\d*", low[idx:])
            token = m.group(0) if m else pat
            # Normalise: strip a trailing separator if no digits followed.
            return token.rstrip("-_")
    return None


def check_completeness(
    files: List[str],
    config: Optional[dict] = None,
) -> dict:
    """Check whether a subject's expected modalities/masks/timepoints are present.

    Parameters
    ----------
    files : list of file paths belonging to ONE subject (any structure).
    config : the expectation config (see DEFAULT_CONFIG). The relevant keys are
        required_modalities, required_masks, timepoint_patterns,
        expected_timepoints.

    Returns
    -------
    dict with {status, checks, reasons} plus:
        present_modalities, present_masks, timepoints (sorted lists),
        missing_modalities, missing_masks.

    Works for any scale:
      * one file        -> reports that single modality; "completeness" of a
                           one-image input passes unless required items are set.
      * one subject     -> checks all required modalities/masks are present.
      * longitudinal    -> groups files by detected timepoint and (if
                           expected_timepoints is set) checks the count.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    checks: Dict[str, dict] = {}

    present_modalities, present_masks, timepoints = set(), set(), set()
    # Track which modalities appear in each detected timepoint, so longitudinal
    # completeness can require every timepoint to have the full set.
    by_timepoint: Dict[str, set] = {}
    for f in files:
        kind, role = classify_file(f, cfg)
        tp = detect_timepoint(f, cfg)
        if kind == "modality":
            present_modalities.add(role)
            if tp:
                by_timepoint.setdefault(tp, set()).add(role)
        elif kind == "mask":
            present_masks.add(role)
        if tp:
            timepoints.add(tp)

    # Modality completeness.
    req_mod = list(cfg.get("required_modalities", []))
    missing_mod = [m for m in req_mod if m not in present_modalities]
    if not req_mod:
        checks["modalities"] = {"status": "pass",
                                "message": f"Present: {sorted(present_modalities) or 'none'} "
                                           "(no required set; presence only)."}
    elif missing_mod:
        checks["modalities"] = {"status": "fail",
                                "message": f"Missing required modalities: {missing_mod} "
                                           f"(present: {sorted(present_modalities)})."}
    else:
        checks["modalities"] = {"status": "pass",
                                "message": f"All required modalities present: {req_mod}."}

    # Mask completeness.
    req_mask = list(cfg.get("required_masks", []))
    missing_mask = [m for m in req_mask if m not in present_masks]
    if not req_mask:
        checks["masks"] = {"status": "pass",
                           "message": f"Masks present: {sorted(present_masks) or 'none'} "
                                      "(no required set)."}
    elif missing_mask:
        checks["masks"] = {"status": "fail",
                           "message": f"Missing required masks: {missing_mask} "
                                      f"(present: {sorted(present_masks)})."}
    else:
        checks["masks"] = {"status": "pass",
                           "message": f"All required masks present: {req_mask}."}

    # Longitudinal completeness.
    exp_tp = cfg.get("expected_timepoints")
    if exp_tp is not None or timepoints:
        n = len(timepoints)
        # Count check (only when an expected number is configured).
        count_ok = (exp_tp is None) or (n >= exp_tp)
        # Per-timepoint modality completeness: each timepoint should carry the
        # full required set (a timepoint missing a modality is incomplete).
        per_tp_missing = {}
        if req_mod:
            for tp in sorted(timepoints):
                miss = [m for m in req_mod if m not in by_timepoint.get(tp, set())]
                if miss:
                    per_tp_missing[tp] = miss
        if not count_ok:
            checks["timepoints"] = {"status": "fail",
                                    "message": f"Found {n} timepoint(s) {sorted(timepoints)}, "
                                               f"expected {exp_tp}."}
        elif per_tp_missing:
            checks["timepoints"] = {"status": "fail",
                                    "message": f"Timepoint(s) missing modalities: {per_tp_missing}."}
        else:
            checks["timepoints"] = {"status": "pass",
                                    "message": f"{n} timepoint(s) present, each complete: "
                                               f"{sorted(timepoints)}."}

    return {
        "status": _worst([c["status"] for c in checks.values()]),
        "checks": checks,
        "reasons": _reasons(checks),
        "present_modalities": sorted(present_modalities),
        "present_masks": sorted(present_masks),
        "timepoints": sorted(timepoints),
        "missing_modalities": missing_mod,
        "missing_masks": missing_mask,
    }


# --------------------------------------------------------------------------- #
# Metadata (header-only) QC
# --------------------------------------------------------------------------- #
def extract_metadata(path: str) -> dict:
    """Read geometry from a NIfTI header without loading the full voxel array.

    Returns a dict with: shape, n_dims, voxel_spacing (mm, per axis),
    n_volumes (4th dim or 1), orientation (e.g. 'RAS'), affine (4x4 list),
    dtype, and read_ok/error. Never raises.
    """
    try:
        img = nibabel.load(path)
        hdr = img.header
        shape = tuple(int(s) for s in img.shape)
        zooms = tuple(float(z) for z in hdr.get_zooms())
        affine = np.asarray(img.affine, dtype=float)
        try:
            orientation = "".join(nibabel.aff2axcodes(affine))
        except Exception:
            orientation = None
        return {
            "read_ok": True,
            "shape": list(shape),
            "n_dims": len(shape),
            "voxel_spacing": [round(z, 4) for z in zooms[:3]],
            "n_volumes": int(shape[3]) if len(shape) > 3 else 1,
            "orientation": orientation,
            "affine": affine.tolist(),
            "dtype": str(hdr.get_data_dtype()),
            "error": None,
        }
    except Exception as exc:
        return {"read_ok": False, "shape": None, "n_dims": None,
                "voxel_spacing": None, "n_volumes": None, "orientation": None,
                "affine": None, "dtype": None, "error": f"{exc!r}"}


def check_metadata(path: str, thresholds: Optional[dict] = None) -> dict:
    """Header-only QC for a single NIfTI file.

    Returns a dict of named sub-checks, each {"status","message"}, plus an
    overall "status" (worst of the sub-checks) and the raw "metadata".
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    meta = extract_metadata(path)
    checks: Dict[str, dict] = {}

    if not meta["read_ok"]:
        return {"status": "fail", "metadata": meta,
                "checks": {"readable": {"status": "fail",
                                        "message": f"Could not read header: {meta['error']}"}}}

    checks["readable"] = {"status": "pass", "message": "Header read successfully."}

    # Dimensionality: expect 3D (or 4D with a volume axis).
    nd = meta["n_dims"]
    if nd in (3, 4):
        checks["dimensionality"] = {"status": "pass", "message": f"{nd}D image."}
    else:
        checks["dimensionality"] = {"status": "fail",
                                    "message": f"Unexpected dimensionality: {nd}D."}

    # Voxel spacing: present, positive, plausible, not too anisotropic.
    sp = meta["voxel_spacing"] or []
    if not sp or any(s is None for s in sp):
        checks["voxel_spacing"] = {"status": "fail", "message": "Missing voxel spacing."}
    elif any(s <= 0 for s in sp):
        checks["voxel_spacing"] = {"status": "fail",
                                   "message": f"Non-positive voxel spacing: {sp}."}
    elif any(s < t["min_voxel_size_mm"] or s > t["max_voxel_size_mm"] for s in sp):
        checks["voxel_spacing"] = {"status": "warning",
                                   "message": f"Implausible voxel spacing {sp} mm "
                                              f"(expected {t['min_voxel_size_mm']}–{t['max_voxel_size_mm']} mm)."}
    else:
        ratio = max(sp) / min(sp) if min(sp) > 0 else float("inf")
        if ratio > t["max_anisotropy_ratio"]:
            checks["voxel_spacing"] = {"status": "warning",
                                       "message": f"Highly anisotropic voxels (ratio {ratio:.1f}): {sp} mm."}
        else:
            checks["voxel_spacing"] = {"status": "pass",
                                       "message": f"Voxel spacing {sp} mm."}

    # Affine validity: present and non-singular.
    aff = meta["affine"]
    if aff is None:
        checks["affine"] = {"status": "fail", "message": "Missing affine."}
    else:
        det = float(np.linalg.det(np.asarray(aff)[:3, :3]))
        if abs(det) < 1e-8:
            checks["affine"] = {"status": "fail",
                                "message": f"Singular affine (det={det:.2e}); geometry undefined."}
        else:
            checks["affine"] = {"status": "pass",
                                "message": f"Affine valid (det={det:.3g}), orientation {meta['orientation']}."}

    overall = _worst([c["status"] for c in checks.values()])
    return {"status": overall, "metadata": meta, "checks": checks}


# --------------------------------------------------------------------------- #
# Per-sample image-feature QC
# --------------------------------------------------------------------------- #
def _foreground_from_image(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Estimate foreground when no brain mask is supplied.

    Rule: voxels above the 10th percentile of the volume's own intensity range.
    MRI background is a large near-zero peak, so this removes it without
    assuming intensity units. The method string is returned for auditability.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=bool), "none(no finite voxels)"
    thr = float(np.percentile(finite, _FOREGROUND_PERCENTILE))
    if thr <= float(finite.min()):
        thr = float(finite.min())
        return image > thr, f"greater-than-min({thr:.4g})"
    return image > thr, f"percentile>{_FOREGROUND_PERCENTILE:g}"


def compute_features(
    image: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
) -> dict:
    """Compute per-sample image features over a foreground voxel set.

    Parameters
    ----------
    image : 3-D float array of voxel intensities.
    brain_mask : optional boolean array, same shape as image. If given,
        statistics are computed over the IMAGE intensities INSIDE this mask
        (the correct option for brain MRI). If omitted, foreground is estimated
        from the image itself (see ``_foreground_from_image``).
    affine : optional 4x4 array. If given, the centroid offset is reported in
        millimetres; otherwise it is reported in voxels.

    Returns
    -------
    dict with (all over the foreground voxel set):
        foreground_method      : str, how foreground was chosen
        foreground_voxels      : int, number of foreground voxels
        total_voxels           : int, total voxels in the volume
        foreground_fraction    : float, foreground_voxels / total_voxels
        intensity_mean/std/min/max/p50/p99 : float, of the IMAGE over foreground
        centroid_offset_mm     : float, distance from foreground centre of mass
                                 to the volume's geometric centre (mm if affine
                                 given, else voxels; key name kept for stability)
        centroid_units         : "mm" or "voxels"
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim > 3:
        image = image[..., 0]
    total = int(image.size)

    if brain_mask is not None:
        mask = np.asarray(brain_mask, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError(f"brain_mask shape {mask.shape} != image shape {image.shape}")
        fg = mask
        method = "brain_mask"
    else:
        fg, method = _foreground_from_image(image)

    fg_values = image[fg]
    fg_count = int(np.count_nonzero(fg))

    if fg_values.size > 0:
        i_mean = float(np.mean(fg_values))
        i_std = float(np.std(fg_values))
        i_min = float(np.min(fg_values))
        i_max = float(np.max(fg_values))
        i_p50 = float(np.percentile(fg_values, 50))
        i_p99 = float(np.percentile(fg_values, 99))
    else:
        i_mean = i_std = i_min = i_max = i_p50 = i_p99 = None

    # Intensity-weighted centroid of the foreground, then offset from the
    # volume's geometric centre.
    centroid_offset = None
    units = "voxels"
    if fg_count > 0:
        weighted = np.where(fg, image, 0.0)
        total_w = float(weighted.sum())
        if total_w > 0:
            grids = np.meshgrid(*[np.arange(s) for s in image.shape], indexing="ij")
            com_vox = np.array([float((g * weighted).sum() / total_w) for g in grids])
            geom_vox = np.array([(s - 1) / 2.0 for s in image.shape])
            if affine is not None:
                aff = np.asarray(affine, dtype=float)
                com_world = aff @ np.array([*com_vox, 1.0])
                geom_world = aff @ np.array([*geom_vox, 1.0])
                centroid_offset = float(np.linalg.norm(com_world[:3] - geom_world[:3]))
                units = "mm"
            else:
                centroid_offset = float(np.linalg.norm(com_vox - geom_vox))
                units = "voxels"

    return {
        "foreground_method": method,
        "foreground_voxels": fg_count,
        "total_voxels": total,
        "foreground_fraction": (fg_count / total) if total else None,
        "intensity_mean": _round(i_mean),
        "intensity_std": _round(i_std),
        "intensity_min": _round(i_min),
        "intensity_max": _round(i_max),
        "intensity_p50": _round(i_p50),
        "intensity_p99": _round(i_p99),
        "centroid_offset_mm": _round(centroid_offset),
        "centroid_units": units,
    }


def check_features(features: dict, thresholds: Optional[dict] = None) -> dict:
    """Grade per-sample features against configurable thresholds.

    Parameters
    ----------
    features : output of ``compute_features``.
    thresholds : optional overrides for ``DEFAULT_THRESHOLDS``. Relevant keys:
        min_foreground_fraction, warn_foreground_fraction, min_intensity_std,
        max_centroid_offset_mm.

    Returns
    -------
    dict of named sub-checks ({"status","message"}) plus overall "status".
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks: Dict[str, dict] = {}

    # Foreground fraction.
    ff = features.get("foreground_fraction")
    if ff is None:
        checks["foreground"] = {"status": "fail", "message": "No foreground computed."}
    elif ff < t["min_foreground_fraction"]:
        checks["foreground"] = {"status": "fail",
                                "message": f"Foreground fraction {ff:.3f} below "
                                           f"{t['min_foreground_fraction']:.2f}: likely empty/failed volume."}
    elif ff < t["warn_foreground_fraction"]:
        checks["foreground"] = {"status": "warning",
                                "message": f"Low foreground fraction {ff:.3f} "
                                           f"(< {t['warn_foreground_fraction']:.2f})."}
    else:
        checks["foreground"] = {"status": "pass",
                                "message": f"Foreground fraction {ff:.3f}."}

    # Intensity dynamic range (constant image is a failure).
    istd = features.get("intensity_std")
    if istd is None:
        checks["intensity"] = {"status": "fail", "message": "No intensity statistics."}
    elif istd < t["min_intensity_std"]:
        checks["intensity"] = {"status": "fail",
                               "message": f"Near-constant intensity (std={istd:.2e}): no contrast."}
    else:
        checks["intensity"] = {"status": "pass",
                               "message": f"Intensity mean={features.get('intensity_mean')}, "
                                          f"std={istd}."}

    # Centroid offset (soft warning only).
    off = features.get("centroid_offset_mm")
    if off is None:
        checks["centroid"] = {"status": "warning", "message": "Centroid not computed."}
    elif features.get("centroid_units") == "mm" and off > t["max_centroid_offset_mm"]:
        checks["centroid"] = {"status": "warning",
                              "message": f"Centroid {off:.1f} mm from centre "
                                         f"(> {t['max_centroid_offset_mm']:.0f} mm): brain off-centre?"}
    else:
        checks["centroid"] = {"status": "pass",
                              "message": f"Centroid offset {off} {features.get('centroid_units')}."}

    return {"status": _worst([c["status"] for c in checks.values()]),
            "checks": checks, "reasons": _reasons(checks)}


def _reasons(checks: dict) -> List[str]:
    """List the checks that did NOT pass, as short 'name: message' strings.

    Lets a caller (or a CSV column) see *which* checks fired without scanning
    every per-check field. A passing result yields an empty list.
    """
    out = []
    for name, info in checks.items():
        if info.get("status") != "pass":
            out.append(f"{name} [{info.get('status')}]: {info.get('message')}")
    return out


# --------------------------------------------------------------------------- #
# Mask / label QC (different checks than images; no intensity statistics)
# --------------------------------------------------------------------------- #
def check_mask(
    mask_path: str,
    reference_image_path: Optional[str] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """QC for a mask or label volume (brain mask, lesion mask, segmentation).

    Masks are NOT images: computing 'intensity mean' on a binary label is
    meaningless (it would be ~1). So a mask gets its own checks:

    * non-empty: at least some non-zero voxels (an all-zero mask is a failure).
    * label-like values: values are integer-like (a mask should not be a
      continuous-intensity image mislabelled as a mask).
    * grid match (if a reference image is given): the mask shares the image's
      shape, so it can actually overlay it.

    Returns the usual {status, checks, reasons} plus a small 'mask_stats' dict
    (non-zero voxel count and fraction, number of distinct labels).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks: Dict[str, dict] = {}
    stats: Dict[str, object] = {}
    try:
        data = np.asarray(load_nifti(mask_path)[0], dtype=np.float32)
        if data.ndim > 3:
            data = data[..., 0]
    except Exception as exc:
        return {"status": "fail",
                "checks": {"readable": {"status": "fail",
                                        "message": f"Could not load mask: {exc!r}"}},
                "reasons": [f"readable [fail]: {exc!r}"], "mask_stats": {}}

    nonzero = int(np.count_nonzero(data))
    total = int(data.size)
    stats["nonzero_voxels"] = nonzero
    stats["nonzero_fraction"] = round(nonzero / total, 6) if total else None
    distinct = np.unique(data[data != 0])
    stats["n_labels"] = int(distinct.size)

    # Non-empty.
    if nonzero == 0:
        checks["non_empty"] = {"status": "fail", "message": "Mask is entirely zero (empty)."}
    else:
        checks["non_empty"] = {"status": "pass",
                               "message": f"{nonzero} non-zero voxels "
                                          f"({stats['nonzero_fraction']:.4f} of volume)."}

    # Label-like (values are (near-)integers).
    if distinct.size > 0:
        non_integer = np.any(np.abs(distinct - np.round(distinct)) > 1e-4)
        if non_integer:
            checks["label_values"] = {"status": "warning",
                                      "message": "Mask has non-integer values; is this "
                                                 "actually an intensity image, not a mask?"}
        else:
            checks["label_values"] = {"status": "pass",
                                      "message": f"{distinct.size} integer label value(s)."}

    # Grid match against a reference image, if given.
    if reference_image_path:
        try:
            ref_shape = tuple(int(s) for s in nibabel.load(reference_image_path).shape[:3])
            if tuple(data.shape[:3]) == ref_shape:
                checks["grid_match"] = {"status": "pass",
                                        "message": f"Mask matches image grid {ref_shape}."}
            else:
                checks["grid_match"] = {"status": "fail",
                                        "message": f"Mask shape {tuple(data.shape[:3])} != "
                                                   f"image shape {ref_shape}; will not overlay."}
        except Exception as exc:
            checks["grid_match"] = {"status": "warning",
                                    "message": f"Could not compare to image grid: {exc!r}"}

    return {"status": _worst([c["status"] for c in checks.values()]),
            "checks": checks, "reasons": _reasons(checks), "mask_stats": stats}


# --------------------------------------------------------------------------- #
# Top-level per-sample entry point
# --------------------------------------------------------------------------- #
def run_qc(
    image_path: str,
    brain_mask_path: Optional[str] = None,
    lesion_mask_path: Optional[str] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """Full per-sample QC for one image, with optional brain and lesion masks.

    Each input is QC'd with the checks appropriate to its role:

    * ``image_path``       -> metadata (header) QC + image-feature QC. If a
      brain mask is given, intensity statistics are restricted to it.
    * ``brain_mask_path``  -> mask QC (non-empty, label-like, grid-matches image).
      Optional; pass it both to restrict image stats AND to QC the mask itself.
    * ``lesion_mask_path`` -> mask QC (same checks). Optional. A lesion mask is
      not treated as an image, so no intensity stats.

    Returns one combined dict::

        {
          "image": <path>,
          "status": "pass"/"warning"/"fail",   # worst across everything run
          "reasons": [ ... ],                   # which checks fired, plainly
          "metadata_qc": {...},
          "feature_qc":  {...},
          "features":    {...},
          "brain_mask_qc":  {...} or None,
          "lesion_mask_qc": {...} or None,
        }

    Reads only the header for metadata QC; reads voxels for the rest.
    """
    meta_qc = check_metadata(image_path, thresholds)

    feature_qc: dict = {}
    features: dict = {}
    try:
        image, _ = load_nifti(image_path)
        affine = np.asarray(nibabel.load(image_path).affine, dtype=float)
        mask = None
        if brain_mask_path:
            mask, _ = load_nifti(brain_mask_path).astype(bool)
        features = compute_features(image, brain_mask=mask, affine=affine)
        feature_qc = check_features(features, thresholds)
    except Exception as exc:
        feature_qc = {"status": "fail",
                      "checks": {"load": {"status": "fail",
                                          "message": f"Feature QC failed: {exc!r}"}},
                      "reasons": [f"load [fail]: {exc!r}"]}

    # Optional mask QC (different checks; never intensity stats).
    brain_mask_qc = (check_mask(brain_mask_path, image_path, thresholds)
                     if brain_mask_path else None)
    lesion_mask_qc = (check_mask(lesion_mask_path, image_path, thresholds)
                      if lesion_mask_path else None)

    layers = {
        "metadata": meta_qc,
        "feature": feature_qc,
        "brain_mask": brain_mask_qc,
        "lesion_mask": lesion_mask_qc,
    }
    statuses = [v.get("status", "fail") for v in layers.values() if v]
    overall = _worst(statuses)

    # Aggregate reasons across every layer, prefixed by layer so it's clear
    # WHICH part flagged (this is what makes a 'warning' self-explanatory).
    reasons: List[str] = []
    for layer_name, v in layers.items():
        if v:
            for r in v.get("reasons", []):
                reasons.append(f"{layer_name}.{r}")

    return {
        "image": image_path,
        "status": overall,
        "reasons": reasons,
        "metadata_qc": meta_qc,
        "feature_qc": feature_qc,
        "features": features,
        "brain_mask_qc": brain_mask_qc,
        "lesion_mask_qc": lesion_mask_qc,
    }


def run_qc_arrays(
    image_path: str,
    image: "np.ndarray",
    brain_mask: Optional["np.ndarray"] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """Per-sample QC when the image (and mask) are already loaded in memory.

    Same result shape as run_qc, but reuses arrays the caller already has
    instead of reading the volume from disk again. Header QC still uses the
    path (it reads only the header). Intended for master.py, where the pipeline
    has already loaded the image and computed a brain mask array.

    image      : the loaded image array.
    brain_mask : optional boolean array on the image grid; if given, intensity
                 stats are restricted to it.
    """
    meta_qc = check_metadata(image_path, thresholds)
    try:
        affine = np.asarray(nibabel.load(image_path).affine, dtype=float)
    except Exception:
        affine = None
    mask = None if brain_mask is None else np.asarray(brain_mask).astype(bool)
    features = compute_features(image, brain_mask=mask, affine=affine)
    feature_qc = check_features(features, thresholds)

    statuses = [meta_qc.get("status", "fail"), feature_qc.get("status", "fail")]
    meta_reasons = meta_qc.get("reasons") or _reasons(meta_qc.get("checks", {}))
    reasons = [f"metadata.{r}" for r in meta_reasons] \
        + [f"feature.{r}" for r in feature_qc.get("reasons", [])]
    return {
        "image": image_path,
        "status": _worst(statuses),
        "reasons": reasons,
        "metadata_qc": meta_qc,
        "feature_qc": feature_qc,
        "features": features,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_STATUS_RANK = {"pass": 0, "warning": 1, "fail": 2, "unknown": 1}


def _worst(statuses: List[str]) -> str:
    """Worst status among a list, by rank fail > warning > pass."""
    if not statuses:
        return "unknown"
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 1))


def _round(x: Optional[float], n: int = 4) -> Optional[float]:
    """Round, mapping non-finite to None (JSON-safe)."""
    if x is None:
        return None
    xf = float(x)
    if not np.isfinite(xf):
        return None
    return round(xf, n)


# --------------------------------------------------------------------------- #
# CLI (matches the team's argparse + JSON-output convention)
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="ClinMRI-QC: per-sample metadata + image-feature QC")
    parser.add_argument("--image", required=True, help="Path to a NIfTI image")
    parser.add_argument("--brain_mask", default=None,
                        help="Optional brain mask NIfTI (enables mask-restricted stats + mask QC)")
    parser.add_argument("--lesion_mask", default=None,
                        help="Optional lesion/label mask NIfTI (gets mask QC, not image QC)")
    parser.add_argument("--min_foreground_fraction", type=float, default=None,
                        help="Override: flag fail below this foreground fraction")
    parser.add_argument("--max_centroid_offset_mm", type=float, default=None,
                        help="Override: warn above this centroid offset (mm)")
    parser.add_argument("--outfile", default=None, help="Optional path to save JSON")
    args = parser.parse_args()

    thresholds = {}
    if args.min_foreground_fraction is not None:
        thresholds["min_foreground_fraction"] = args.min_foreground_fraction
    if args.max_centroid_offset_mm is not None:
        thresholds["max_centroid_offset_mm"] = args.max_centroid_offset_mm

    result = run_qc(args.image, brain_mask_path=args.brain_mask,
                    lesion_mask_path=args.lesion_mask, thresholds=thresholds)
    output = json.dumps(result, indent=2)
    print(output)
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(output)
        print(f"\nResults saved to {args.outfile}")


if __name__ == "__main__":
    main()
