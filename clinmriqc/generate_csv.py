"""Build a flat CSV-ready record from any combination of ClinMRI-QC module outputs.

Usage
-----
    from clinmriqc.artifacts import detect_artifacts
    from clinmriqc.contrast  import detect_contrast_enhancement
    from clinmriqc.generate_csv import build_qc_record

    art = detect_artifacts(image, brain_mask)
    con = detect_contrast_enhancement(image, brain_mask)

    record = build_qc_record(
        image_path='T1w.nii.gz',
        patient_id='sub-001',
        artifacts=art,
        contrast=con,
    )
"""

from datetime import datetime
from pathlib import Path

from clinmriqc.schema import ALL_COLUMNS, ARTIFACT_CLASSES


def build_qc_record(
    image_path: str,
    patient_id: str = None,
    artifacts: dict = None,
    contrast: dict = None,
    coreg: dict = None,
    meta: dict = None,
    timestamp: str = None,
) -> dict:
    """Flatten module outputs into a flat dict matching the canonical CSV schema.

    Parameters
    ----------
    image_path  : path to the NIfTI scan — stored as scan identifier.
    patient_id  : optional label; defaults to the NIfTI filename stem.
    artifacts   : dict returned by detect_artifacts(), or None.
    contrast    : dict returned by detect_contrast_enhancement(), or None.
    coreg       : dict returned by registration_qc(), or None.
    meta        : dict returned by metaqc.run_qc(), or None.
    timestamp   : ISO datetime string; defaults to now.

    Returns
    -------
    dict with ALL_COLUMNS keys.  Columns for modules not supplied are ''.
    """
    record = dict.fromkeys(ALL_COLUMNS, '')
    record['timestamp']  = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M')
    record['scan_path']  = str(image_path)
    record['patient_id'] = patient_id or Path(image_path).stem

    if artifacts is not None:
        # Prefer scaled [0,1] severity (regression model output after calibration).
        # Falls back to raw regression scores or old softmax probabilities.
        scores = (
            artifacts.get('artifact_severity_scaled')
            or artifacts.get('artifact_severity')
            or artifacts.get('artifact_probabilities', {})
        )
        record['artifacts_quality_passed'] = artifacts.get('quality_passed', '')
        detected = artifacts.get('artifacts_detected', [])
        record['artifacts_detected'] = '|'.join(detected) if detected else ''
        # prob_clean always empty in regression mode; kept for schema compatibility.
        for cls in ARTIFACT_CLASSES:
            record[f'prob_{cls}'] = scores.get(cls, '')
        iqms = artifacts.get('iqms', {})
        record['iqm_motion_blur_score'] = iqms.get('motion_blur_score', '')
        record['iqm_snr']               = iqms.get('snr', '')

    if contrast is not None:
        record['contrast_enhanced']              = contrast.get('enhanced', '')
        record['contrast_vessel_ratio']          = contrast.get('vessel_ratio', '')
        record['contrast_bright_voxel_fraction'] = contrast.get('bright_voxel_fraction', '')

    if coreg is not None:
        record['coreg_flag'] = coreg.get('flag', '')
        record['coreg_ssim'] = coreg.get('ssim', '')
        record['coreg_ncc']  = coreg.get('ncc', '')
        passed = coreg.get('passed', {})
        record['coreg_ssim_passed'] = passed.get('ssim', '')
        record['coreg_ncc_passed']  = passed.get('ncc', '')

    if meta is not None:
        record['metaqc_status']   = meta.get('status', '')
        reasons = meta.get('reasons', [])
        record['metaqc_reasons']  = '|'.join(reasons) if reasons else ''
        features = meta.get('features', {})
        record['metaqc_foreground_fraction'] = features.get('foreground_fraction', '')
        record['metaqc_intensity_mean']      = features.get('intensity_mean', '')
        record['metaqc_intensity_std']       = features.get('intensity_std', '')
        record['metaqc_centroid_offset_mm']  = features.get('centroid_offset_mm', '')
        meta_qc = meta.get('metadata_qc', {})
        record['metaqc_metadata_status'] = meta_qc.get('status', '')

    return record
