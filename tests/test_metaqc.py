"""Tests for metaqc.py - per-sample metadata + image-feature QC.

These mirror validate_metaqc.py but as assertions, so `pytest` enforces that
the computed numbers match the known synthetic ground truth.
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import nibabel as nib
from quickbrain import metaqc


def _save(path, array, spacing=(1.0, 1.0, 1.0)):
    img = nib.Nifti1Image(array.astype(np.float32), np.diag([*spacing, 1.0]))
    img.header.set_zooms(spacing)
    nib.save(img, str(path))


def test_mask_restricted_mean_is_image_not_mask(tmp_path):
    # Image 50/150 split inside mask -> mean 100, std 50 (NOT the mask's 1s).
    shape = (40, 40, 40)
    img = np.zeros(shape, dtype=np.float32)
    img[10:30, 10:30, 10:20] = 50.0
    img[10:30, 10:30, 20:30] = 150.0
    mask = np.zeros(shape, dtype=np.float32); mask[10:30, 10:30, 10:30] = 1.0
    feats = metaqc.compute_features(img, brain_mask=mask.astype(bool), affine=np.eye(4))
    assert abs(feats["intensity_mean"] - 100.0) < 1e-3
    assert abs(feats["intensity_std"] - 50.0) < 1e-3
    assert feats["foreground_method"] == "brain_mask"


def test_foreground_fraction_exact(tmp_path):
    shape = (40, 40, 40)
    img = np.zeros(shape, dtype=np.float32); img[10:30, 10:30, 10:30] = 100.0
    mask = np.zeros(shape, dtype=np.float32); mask[10:30, 10:30, 10:30] = 1.0
    feats = metaqc.compute_features(img, brain_mask=mask.astype(bool))
    assert feats["foreground_voxels"] == 8000
    assert abs(feats["foreground_fraction"] - 0.125) < 1e-6


def test_uniform_region_zero_std(tmp_path):
    img = np.zeros((30, 30, 30), dtype=np.float32); img[5:25, 5:25, 5:25] = 100.0
    mask = (img > 0)
    feats = metaqc.compute_features(img, brain_mask=mask)
    assert abs(feats["intensity_std"]) < 1e-6


def test_metadata_reads_spacing(tmp_path):
    img = np.zeros((20, 20, 20), dtype=np.float32); img[5:15, 5:15, 5:15] = 100
    p = tmp_path / "s.nii.gz"; _save(p, img, spacing=(0.8, 0.8, 3.0))
    meta = metaqc.check_metadata(str(p))
    assert meta["metadata"]["voxel_spacing"] == [0.8, 0.8, 3.0]
    assert meta["status"] in ("pass", "warning")


def test_empty_volume_fails_foreground(tmp_path):
    img = np.zeros((40, 40, 40), dtype=np.float32); img[0:5, 0:5, 0:5] = 100
    p = tmp_path / "e.nii.gz"; _save(p, img)
    res = metaqc.run_qc(str(p))
    assert res["feature_qc"]["checks"]["foreground"]["status"] == "fail"


def test_threshold_override(tmp_path):
    img = np.zeros((40, 40, 40), dtype=np.float32); img[0:5, 0:5, 0:5] = 100
    p = tmp_path / "e.nii.gz"; _save(p, img)
    res = metaqc.run_qc(str(p), thresholds={"min_foreground_fraction": 0.001,
                                            "warn_foreground_fraction": 0.001})
    assert res["feature_qc"]["checks"]["foreground"]["status"] == "pass"


def test_constant_image_fails_intensity(tmp_path):
    img = np.full((30, 30, 30), 50.0, dtype=np.float32)
    mask = np.ones((30, 30, 30), dtype=bool)
    feats = metaqc.compute_features(img, brain_mask=mask)
    chk = metaqc.check_features(feats)
    assert chk["checks"]["intensity"]["status"] == "fail"


def test_mask_shape_mismatch_raises():
    img = np.zeros((40, 40, 40), dtype=np.float32)
    mask = np.zeros((30, 30, 30), dtype=bool)
    try:
        metaqc.compute_features(img, brain_mask=mask)
        assert False, "should have raised"
    except ValueError:
        pass


# ---- mask QC and reasons (added for per-role config) ----

def test_lesion_mask_gets_mask_qc_not_image_qc(tmp_path):
    sh = (40, 40, 40)
    img = np.zeros(sh, dtype=np.float32); img[10:30, 10:30, 10:30] = 100
    les = np.zeros(sh, dtype=np.float32); les[15:18, 15:18, 15:18] = 1
    _save(tmp_path / "img.nii.gz", img)
    _save(tmp_path / "lesion.nii.gz", les)
    res = metaqc.run_qc(str(tmp_path / "img.nii.gz"),
                        lesion_mask_path=str(tmp_path / "lesion.nii.gz"))
    # Lesion mask is QC'd as a mask (non-empty, label-like, grid match) ...
    assert res["lesion_mask_qc"] is not None
    assert res["lesion_mask_qc"]["mask_stats"]["nonzero_voxels"] == 27
    # ... and has NO intensity stats (those would be meaningless for a label).
    assert "intensity_mean" not in res["lesion_mask_qc"]


def test_empty_mask_fails(tmp_path):
    sh = (30, 30, 30)
    img = np.zeros(sh, dtype=np.float32); img[5:25, 5:25, 5:25] = 100
    empty = np.zeros(sh, dtype=np.float32)
    _save(tmp_path / "img.nii.gz", img)
    _save(tmp_path / "empty_mask.nii.gz", empty)
    qc = metaqc.check_mask(str(tmp_path / "empty_mask.nii.gz"),
                           str(tmp_path / "img.nii.gz"))
    assert qc["status"] == "fail"
    assert qc["checks"]["non_empty"]["status"] == "fail"


def test_mask_grid_mismatch_fails(tmp_path):
    img = np.zeros((40, 40, 40), dtype=np.float32); img[10:30, 10:30, 10:30] = 100
    mask = np.zeros((30, 30, 30), dtype=np.float32); mask[5:25, 5:25, 5:25] = 1
    _save(tmp_path / "img.nii.gz", img)
    _save(tmp_path / "mask.nii.gz", mask)
    qc = metaqc.check_mask(str(tmp_path / "mask.nii.gz"), str(tmp_path / "img.nii.gz"))
    assert qc["checks"]["grid_match"]["status"] == "fail"


def test_reasons_names_the_failing_check(tmp_path):
    img = np.zeros((40, 40, 40), dtype=np.float32); img[0:5, 0:5, 0:5] = 100
    _save(tmp_path / "e.nii.gz", img)
    res = metaqc.run_qc(str(tmp_path / "e.nii.gz"))
    # reasons should mention the foreground check explicitly
    assert any("foreground" in r for r in res["reasons"])


def test_single_image_no_mask_sections(tmp_path):
    img = np.zeros((40, 40, 40), dtype=np.float32); img[10:30, 10:30, 10:30] = 100
    _save(tmp_path / "img.nii.gz", img)
    res = metaqc.run_qc(str(tmp_path / "img.nii.gz"))
    assert res["brain_mask_qc"] is None
    assert res["lesion_mask_qc"] is None


# ---- completeness (configurable modalities/masks/timepoints) ----

def test_completeness_single_image_passes():
    r = metaqc.check_completeness(["patient01/T1.nii.gz"])
    assert r["status"] == "pass"
    assert "T1w" in r["present_modalities"]


def test_completeness_missing_required_modality_fails():
    files = ["p/T1W.nii.gz", "p/T2W.nii.gz"]
    r = metaqc.check_completeness(files, {"required_modalities": ["T1w", "FLAIR"]})
    assert r["status"] == "fail"
    assert "FLAIR" in r["missing_modalities"]


def test_completeness_custom_mask_role():
    cfg = {"masks": {"tumor": ["tumor_seg"]}, "required_masks": ["tumor"]}
    r = metaqc.check_completeness(["c/img.nii.gz", "c/tumor_seg.nii.gz"], cfg)
    assert r["status"] == "pass"
    assert "tumor" in r["present_masks"]


def test_completeness_longitudinal_timepoints():
    files = ["s/ses-01/T1.nii.gz", "s/ses-02/T1.nii.gz"]
    r = metaqc.check_completeness(files, {"expected_timepoints": 3})
    assert r["status"] == "fail"
    assert len(r["timepoints"]) == 2


def test_classify_file_mask_before_modality():
    # 'brainmask' contains 't1'? no, but ensure mask patterns win over modality
    kind, role = metaqc.classify_file("brainmask.nii.gz")
    assert kind == "mask" and role == "brain"


def test_t1wks_classified_as_t1ce_not_t1w():
    # Regression: 't1wks' contains 't1w', so plain-T1 patterns must not claim it.
    assert metaqc.classify_file("T1WKS.nii.gz") == ("modality", "T1CE")
    assert metaqc.classify_file("T1W.nii.gz") == ("modality", "T1w")


def test_completeness_t1wks_counts_as_t1ce():
    files = ["p/T1W.nii.gz", "p/FLAIR.nii.gz", "p/T1WKS.nii.gz", "p/T2W.nii.gz"]
    r = metaqc.check_completeness(files, {"required_modalities": ["T1w", "FLAIR", "T1CE", "T2w"]})
    assert r["status"] == "pass"
    assert "T1CE" in r["present_modalities"]


# ---- longitudinal, using the REAL open_ms_data longitudinal naming ----

_LONG_FILES = [
    "patient01/brainmask.nii.gz", "patient01/gt.nii.gz",
    "patient01/study1_FLAIR.nii.gz", "patient01/study1_T1W.nii.gz", "patient01/study1_T2W.nii.gz",
    "patient01/study2_FLAIR.nii.gz", "patient01/study2_T1W.nii.gz", "patient01/study2_T2W.nii.gz",
]
_LONG_CFG = {"required_modalities": ["T1w", "FLAIR", "T2w"],
             "required_masks": ["brain", "lesion"], "expected_timepoints": 2}


def test_longitudinal_timepoint_token_not_whole_filename():
    assert metaqc.detect_timepoint("patient01/study1_FLAIR.nii.gz") == "study1"
    assert metaqc.detect_timepoint("patient01/study2_T1W.nii.gz") == "study2"
    assert metaqc.detect_timepoint("patient01/brainmask.nii.gz") is None


def test_longitudinal_bare_gt_is_lesion():
    assert metaqc.classify_file("gt.nii.gz") == ("mask", "lesion")


def test_longitudinal_complete_subject_passes():
    r = metaqc.check_completeness(_LONG_FILES, _LONG_CFG)
    assert r["status"] == "pass"
    assert r["timepoints"] == ["study1", "study2"]


def test_longitudinal_missing_modality_in_one_timepoint_fails():
    files = [f for f in _LONG_FILES if f != "patient01/study2_T2W.nii.gz"]
    r = metaqc.check_completeness(files, _LONG_CFG)
    assert r["status"] == "fail"
    assert any("study2" in reason and "T2w" in reason for reason in r["reasons"])


def test_longitudinal_missing_whole_timepoint_fails_count():
    files = [f for f in _LONG_FILES if not f.split("/")[-1].startswith("study2")]
    r = metaqc.check_completeness(files, _LONG_CFG)
    assert r["status"] == "fail"  # only 1 timepoint, expected 2
