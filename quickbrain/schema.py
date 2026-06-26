"""Shared CSV schema and display constants for ClinMRI-QC.

Imported by generate_csv, append_csv, and generate_report so the column
definition lives in exactly one place.
"""

# Canonical ordered columns for every QC CSV.
# Columns for modules not run are written as empty strings.
ALL_COLUMNS = [
    'timestamp', 'scan_path', 'patient_id',
    # detect_artifacts() — scaled [0,1] severity scores
    'artifacts_quality_passed', 'artifacts_detected',
    'prob_clean', 'prob_motion', 'prob_noise', 'prob_ghosting',
    'prob_bias_field', 'prob_gibbs', 'prob_zipper',
    'iqm_motion_blur_score', 'iqm_snr',
    # detect_contrast_enhancement()
    'contrast_enhanced', 'contrast_vessel_ratio', 'contrast_bright_voxel_fraction',
    # registration_qc()
    'coreg_flag', 'coreg_ssim', 'coreg_ncc', 'coreg_ssim_passed', 'coreg_ncc_passed',
    # metaqc.run_qc() — metadata + image-feature QC
    'metaqc_status', 'metaqc_reasons',
    'metaqc_foreground_fraction', 'metaqc_intensity_mean',
    'metaqc_intensity_std', 'metaqc_centroid_offset_mm',
    'metaqc_metadata_status',
]

# Regression model classes (no 'clean'); prob_clean remains empty in every row.
ARTIFACT_CLASSES = ['motion', 'noise', 'ghosting', 'bias_field', 'gibbs', 'zipper']

# Normal ranges derived from 30-patient MS dataset analysis.
IQM_RANGES = {
    'motion_blur_score': {
        'label':   'Motion / Blur Score (EFC)',
        'normal':  (0.74, 0.87),
        'warning': 0.70,
        'note':    'Lower = more motion / blur.  Normal T1w range: 0.74–0.87.',
    },
    'snr': {
        'label':   'Signal-to-Noise Ratio',
        'normal':  (1.11, 1.73),
        'warning': 1.00,
        'note':    'Lower = noisier image.  Normal T1w range: 1.11–1.73.',
    },
}

RECOMMENDATIONS = {
    'motion': (
        'Motion artifact',
        'Apply motion correction (e.g. ANTs, FSL MCFLIRT) or consider '
        're-acquiring the scan before running parcellation.',
    ),
    'noise': (
        'Elevated noise',
        'Apply denoising (e.g. MP-PCA, NLMeans) during preprocessing, '
        'or review scanner SNR settings.',
    ),
    'ghosting': (
        'Ghosting artifact',
        'Likely caused by patient motion during phase encoding. '
        'Consider re-acquisition or partial Fourier correction.',
    ),
    'bias_field': (
        'Bias field inhomogeneity',
        'Apply N4 bias field correction (ANTs, SimpleITK) '
        'before running segmentation or parcellation.',
    ),
    'gibbs': (
        'Gibbs ringing',
        'Apply Gibbs suppression during preprocessing '
        '(e.g. MRtrix3 mrdegibbs) before downstream analysis.',
    ),
    'zipper': (
        'Zipper / RF interference',
        'This is a hardware or environment issue. '
        'Contact your MRI physicist — re-acquisition is recommended.',
    ),
}

# Severity bands in the scaled [0,1] space.
# Mild = just over the per-class flag threshold (varies); Moderate ≥ 0.5; Severe ≥ 0.75.
SEVERITY_LEVELS = [
    (0.75, 1.01, 'Severe',   '#ef4444'),
    (0.50, 0.75, 'Moderate', '#f97316'),
    (0.00, 0.50, 'Mild',     '#f59e0b'),
]


def severity_label(scaled_score: float) -> tuple:
    """Return (label, colour_hex) for a scaled [0,1] artifact severity score."""
    for lo, hi, label, colour in SEVERITY_LEVELS:
        if lo <= scaled_score < hi:
            return label, colour
    return 'Mild', '#f59e0b'
