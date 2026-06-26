#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
import warnings
from typing import Tuple 
import numpy as np 
import nibabel
import nibabel as nib
from skimage.metrics import structural_similarity
from scipy.ndimage import zoom
from .general import load_nifti, get_brain_mask

# thresholds 
THRESHOLDS = {"ssim": 0.70, "ncc": 0.80,}
ANSI = {
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "RED":    "\033[91m",
    "RESET":  "\033[0m",
    "BOLD":   "\033[1m",
}
legend = {
    "GREEN":  "QC passed",
    "YELLOW": "Review recommended",
    "RED":    "QC failed",
    }

def preprocess(ref: np.ndarray, reg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_shape = tuple(min(r, g) for r, g in zip(ref.shape, reg.shape))
    slices = tuple(slice(0, s) for s in min_shape)
    ref = ref[slices]
    reg = reg[slices]

    def _znorm(arr: np.ndarray) -> np.ndarray:
        std = arr.std()
        if std < 1e-8:
            return arr - arr.mean()
        return (arr - arr.mean()) / std

    return _znorm(ref), _znorm(reg)

# metrics 
def compute_ssim(ref: np.ndarray, reg: np.ndarray) -> float:
    data_range = float(max(ref.max(), reg.max()) - min(ref.min(), reg.min()))

    if data_range < 1e-8:
        return 1.0

    if ref.ndim == 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                score = structural_similarity(
                    ref,
                    reg,
                    data_range=data_range,
                    win_size=7,
                    channel_axis=None,
                )
                return float(score)
            except Exception:
                scores = [
                    structural_similarity(
                        ref[..., i],
                        reg[..., i],
                        data_range=data_range,
                    )
                    for i in range(ref.shape[-1])
                ]
                return float(np.mean(scores))
    else:
        return float(
            structural_similarity(ref, reg, data_range=data_range)
        )

def compute_ncc(ref: np.ndarray, reg: np.ndarray) -> float:
    ref_flat = ref.ravel() - ref.mean()
    reg_flat = reg.ravel() - reg.mean()

    denom = np.linalg.norm(ref_flat) * np.linalg.norm(reg_flat)
    if denom < 1e-12:
        return 1.0  

    return float(np.dot(ref_flat, reg_flat) / denom)

# qc criteria 
def evaluate_flag(results: dict, thresholds: dict) -> str:
    failures = sum(1 for k, v in results.items() if v < thresholds[k])
    if failures == 0:
        return "GREEN"
    if failures == 1:
        return "YELLOW"
    return "RED"

def print_report(ref_path: str, reg_path: str,
                 results: dict, thresholds: dict, flag: str):
    colour = ANSI[flag]
    reset  = ANSI["RESET"]
    bold   = ANSI["BOLD"]

    flag_symbols = {"GREEN": "●", "YELLOW": "●", "RED": "●"}

    width = 60
    print()
    print(bold + "=" * width + reset)
    print(bold + "  Brain MRI Registration QC Report" + reset)
    print(bold + "=" * width + reset)
    print(f"  Reference : {ref_path}")
    print(f"  Registered: {reg_path}")
    print("-" * width)
    print(f"  {'Metric':<10}  {'Value':>8}  {'Threshold':>10}  {'Status'}")
    print("-" * width)

    for metric, value in results.items():
        thr    = thresholds[metric]
        passed = value >= thr
        status_colour = ANSI["GREEN"] if passed else ANSI["RED"]
        status = status_colour + ("PASS ✔" if passed else "FAIL ✘") + reset
        print(f"  {metric.upper():<10}  {value:>8.4f}  {thr:>10.4f}  {status}")

    print("-" * width)
    print(f"  Overall QC flag:  {colour}{bold}{flag_symbols[flag]} {flag}{reset} - {legend[flag]}")
    print(bold + "=" * width + reset)
    print()

# core function
def registration_qc(
    ref_arr: np.ndarray,
    reg_arr: np.ndarray,
    ref_path: str = "",      
    reg_path: str = "",
    ssim_threshold: float = THRESHOLDS["ssim"],
    ncc_threshold:  float = THRESHOLDS["ncc"],
    verbose: bool = True,
) -> dict:
    thresholds = {"ssim": ssim_threshold, "ncc": ncc_threshold}

    if verbose:
        print(f"> Reference shape : {ref_arr.shape}")
        print(f"> Registered shape: {reg_arr.shape}")
        print("> Pre-processing (shape alignment + normalisation) …")

    ref_arr, reg_arr = preprocess(ref_arr, reg_arr)

    if verbose:
        print("> Computing metrics …")

    ssim_val = compute_ssim(ref_arr, reg_arr)
    ncc_val  = compute_ncc(ref_arr, reg_arr)

    results = {"ssim": ssim_val, "ncc": ncc_val}
    flag    = evaluate_flag(results, thresholds)
    passed  = {k: v >= thresholds[k] for k, v in results.items()}

    if verbose:
        print_report(ref_path, reg_path, results, thresholds, flag)

    return {**results, "flag": flag, "passed": passed}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain MRI Registration QC (SSIM · NCC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-ref", "--reference",  required=True,
                        help="Path to reference image")
    parser.add_argument("-reg", "--registered", required=True,
                        help="Path to registered image")
    parser.add_argument("--ssim-threshold", type=float, default=THRESHOLDS["ssim"],
                        help=f"SSIM pass threshold (default: {THRESHOLDS['ssim']})")
    parser.add_argument("--ncc-threshold",  type=float, default=THRESHOLDS["ncc"],
                        help=f"NCC pass threshold  (default: {THRESHOLDS['ncc']})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output; print only the flag")
    return parser.parse_args()


def main():
    args = parse_args()

    for label, path in [("Reference", args.reference), ("Registered", args.registered)]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} path does not exist: {path}")

    print("> Loading reference image …")
    ref_arr = load_nifti(args.reference)

    print("> Loading registered image …")
    reg_arr = load_nifti(args.registered)

    # get brain mask 
    print("> Generating brain mask from reference image …")
    brain_mask = get_brain_mask(args.reference)  

    min_shape = tuple(min(r, g) for r, g in zip(ref_arr.shape, reg_arr.shape))
    slices    = tuple(slice(0, s) for s in min_shape)
    mask_crop = brain_mask[slices]

    ref_brain = ref_arr.copy()
    reg_brain = reg_arr.copy()
    ref_brain[~mask_crop] = 0.0
    reg_brain[~mask_crop] = 0.0

    print(f"> Brain voxels: {mask_crop.sum():,} / {mask_crop.size:,} "
          f"({100 * mask_crop.mean():.1f} %)")

    # qc on brain only 
    result = registration_qc(
        ref_arr        = ref_brain,
        reg_arr        = reg_brain,
        ref_path       = args.reference,
        reg_path       = args.registered,
        ssim_threshold = args.ssim_threshold,
        ncc_threshold  = args.ncc_threshold,
        verbose        = not args.quiet,
    )

    if args.quiet:
        print(legend[result["flag"]])

    exit_codes = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    sys.exit(exit_codes[result["flag"]])


if __name__ == "__main__":
    main()
