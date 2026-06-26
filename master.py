#!/usr/bin/env python3
"""ClinMRI-QC master pipeline.

Runs the full QC pipeline on a directory of T1w NIfTI scans and produces
a CSV record and HTML report.

Steps per scan
--------------
1. load_nifti        — load the image into a NumPy array
2. get_brain_mask    — skull-strip via brainchop (mindgrab model) to get a
                       native-resolution brain mask
3. detect_artifacts  — ResNet50 regression model; returns scaled [0,1]
                       severity scores for 6 artifact classes + IQMs
4. detect_contrast   — heuristic gadolinium screening
5. build_qc_record   — flatten all outputs into a flat dict
6. append_csv_record — append the dict to a growing CSV (header written once)

After all scans are processed:
7. generate_html_from_csv — render a self-contained HTML report

The artifact model is loaded once and shared across all scans to avoid
reloading the 90 MB checkpoint on every iteration.

Usage
-----
    # Full folder
    python master.py \\
        --images_dir /path/to/niftis \\
        --output_dir ./qc_output

    # Quick test on first 3 scans
    python master.py \\
        --images_dir /path/to/niftis \\
        --output_dir ./qc_output \\
        --limit 3

Options
-------
    --images_dir    Directory of .nii / .nii.gz T1w scans  [required]
    --output_dir    Where to write qc_results.csv and qc_report.html  [required]
    --device        'cpu' or 'cuda'  [default: auto-detect]
    --limit         Process only the first N scans
    --no_resume     Reprocess scans already present in the CSV
    --exclude_prefix
                    Skip files whose name starts with this string
                    [default: 'synthetic_']
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# clinmriqc/ lives alongside this script at the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from clinmriqc.general          import load_nifti, get_brain_mask, load_config
from clinmriqc.artifacts        import detect_artifacts
from clinmriqc.contrast         import detect_contrast_enhancement
from clinmriqc.coreg            import registration_qc
from clinmriqc                  import metaqc
from clinmriqc.generate_csv     import build_qc_record
from clinmriqc.append_csv       import append_csv_record
from clinmriqc.generate_report  import generate_html_from_csv
from clinmriqc.classifier.model import load_regression_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _already_processed(csv_path: Path) -> set:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with open(csv_path, newline='') as f:
        return {row['patient_id'] for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Per-scan pipeline
# ---------------------------------------------------------------------------

def process_scan(img_path: Path, device: str, cfg: dict) -> tuple:
    """Run the full QC pipeline for one scan.

    Args:
        img_path: Path to the NIfTI file.
        device: 'cpu' or 'cuda'.
        cfg: Config dict loaded from default.json (or a custom override).

    Returns:
        (artifacts_result, contrast_result, metaqc_result)
    """
    image = load_nifti(str(img_path))

    # Skull-strip via brainchop. get_brain_mask uses _save_inverse_conform so
    # the returned mask is already at native image resolution.
    brain_mask = get_brain_mask(str(img_path))
    
    # Empty mask (unusual acquisitions) — fall back to None so each downstream
    # function uses its own internal masking strategy.
    if brain_mask.sum() == 0:
        brain_mask = None

    art_cfg = cfg["check_artifacts"]
    _log(f'Loading artifact model from {art_cfg["model_path"]} ...')
    model = load_regression_model(art_cfg["model_path"], device=device)
    _log('Model loaded.')
    art = detect_artifacts(
        image,
        brain_mask=brain_mask,
        model=model,
        device=device,
        class_thresholds={k: v for k, v in art_cfg["class_thresholds"].items() if v is not None} or None,
    )

    con_cfg = cfg["check_contrast_enhancement"]
    con = detect_contrast_enhancement(
        image,
        brain_mask,
        vessel_ratio_threshold=con_cfg["vessel_ratio_threshold"],
        bright_fraction_threshold=con_cfg["bright_fraction_threshold"],
    )
    
  # check registration only if ref img is provided 
    coreg = None
    if ref_path is not None: 
        ref_arr = load_nifti(ref_path)
        ref_mask = get_brain_mask(ref_path) 
        min_shape = tuple(min(r,g) for r,g in zip(ref_arr.shape, image.shape))
        slices = tuple(slice(0,s) for s in min_shape)
        mask_crop = ref_mask[slices]
        ref_brain = ref_arr.copy()
        reg_brain = image.copy()
        ref_brain[~mask_crop] = 0.0
        reg_brain[~mask_crop] = 0.0
 
        coreg = registration_qc(
            ref_arr  = ref_brain,
            reg_arr  = reg_brain,
            ref_path = ref_path,
            reg_path = str(img_path),
            verbose  = True,
        )

    

    # Metadata + per-sample feature QC. Reuses the already-loaded image and the
    # brain mask computed above, so the volume is not read from disk again.
    meta_cfg = cfg.get("check_metadata", {})
    meta = metaqc.run_qc_arrays(
        str(img_path), image, brain_mask=brain_mask, thresholds=meta_cfg,
    )

    return art, con, coreg, meta


# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------

def run(
    images_dir: str,
    output_dir: str,
    device: str,
    limit: int,
    resume: bool,
    exclude_prefix: str,
    config_path: str | None = None,
    ref_path: str = None,
):
    cfg = load_config(config_path)
    _log(f'Config loaded from {config_path or "default"}')

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = output_dir / 'qc_results.csv'
    html_path = output_dir / 'qc_report.html'

    image_files = sorted(
        f for f in list(images_dir.glob('*.nii.gz')) + list(images_dir.glob('*.nii'))
        if not (exclude_prefix and f.name.startswith(exclude_prefix))
    )
    if not image_files:
        _log(f'ERROR: no NIfTI files found in {images_dir}')
        sys.exit(1)

    if limit:
        image_files = image_files[:limit]

    done = _already_processed(csv_path) if resume else set()
    if done:
        _log(f'Resuming: {len(done)} already done, '
             f'{len(image_files) - len(done)} remaining')

    n_done = n_skip = n_errors = 0
    t_start = time.time()

    for i, img_path in enumerate(image_files, 1):
        patient_id = img_path.name.replace('.nii.gz', '').replace('.nii', '')

        if patient_id in done:
            n_skip += 1
            continue

        t0 = time.time()
        try:
            art, con, coreg, meta = process_scan(img_path, device, cfg,ref_path=ref_path)

            record = build_qc_record(
                image_path=img_path,
                patient_id=patient_id,
                artifacts=art,
                contrast=con,
                meta=meta,
            )
            append_csv_record(record, str(csv_path))

            elapsed = time.time() - t0
            status  = ('PASS' if art['quality_passed']
                       else f'FAIL [{", ".join(art["artifacts_detected"])}]')
            _log(f'[{i}/{len(image_files)}] {patient_id:<40s}  {status}  ({elapsed:.1f}s)')
            n_done += 1

        except Exception as exc:
            elapsed = time.time() - t0
            _log(f'[{i}/{len(image_files)}] ERROR: {patient_id} — {exc}  ({elapsed:.1f}s)')
            traceback.print_exc()
            n_errors += 1

    total = time.time() - t_start
    _log(f'\nDone: {n_done} processed, {n_skip} skipped, '
         f'{n_errors} errors  ({total/60:.1f} min)')

    if csv_path.exists() and csv_path.stat().st_size > 0:
        _log('Generating HTML report ...')
        generate_html_from_csv(str(csv_path), str(html_path))
        _log(f'Report: {html_path}')
    else:
        _log('No output CSV — nothing to report.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description='ClinMRI-QC master pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--images_dir',     required=True,
                    help='Directory of T1w NIfTI scans')
    ap.add_argument('--output_dir',     required=True,
                    help='Output directory for CSV and HTML report')
    ap.add_argument('--device',         default=None,
                    help="'cpu' or 'cuda'  [default: auto-detect]")
    ap.add_argument('--limit',          type=int, default=None,
                    help='Process only the first N scans')
    ap.add_argument('--no_resume',      action='store_true',
                    help='Reprocess scans already in the CSV')
    ap.add_argument('--exclude_prefix', default='synthetic_',
                    help="Skip files starting with this prefix  [default: 'synthetic_']")
    ap.add_argument('--config',         default=None,
                    help='Path to JSON config file  [default: config/default.json]')
    ap.add_argument('--ref',            default=None,
                    help='Reference image path; when provided, registration QC is run '
                         'for every scan against this reference')
    args = ap.parse_args()

    run(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
        resume=not args.no_resume,
        exclude_prefix=args.exclude_prefix,
        config_path=args.config,
        ref_path=args.ref,
    )


if __name__ == '__main__':
    main()
