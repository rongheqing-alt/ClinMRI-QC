''' Script to test contrast detection on the open_ms_data (https://github.com/muschellij2/open_ms_data)
    Download the dataset using:
    git clone https://github.com/muschellij2/open_ms_data.git
'''

# %%
import glob
import os
from pathlib import Path

import pandas as pd
from quickbrain.contrast import detect_contrast_enhancement
from quickbrain.general import get_brain_mask, load_nifti

VESSEL_RATIO_THRESHOLD = 1.6
BRIGHT_FRACTION_THRESHOLD = 0.002


def process_image(image: str) -> dict:
    print(f"Processing: {image}")
    image_arr, _ = load_nifti(image)
    mask_arr = get_brain_mask(image)
    results = detect_contrast_enhancement(
        image_arr, mask_arr,
        vessel_ratio_threshold=VESSEL_RATIO_THRESHOLD,
        bright_fraction_threshold=BRIGHT_FRACTION_THRESHOLD,
    )
    results["id"] = Path(image).parent.name
    results["path"] = image
    results["contrast"] = "T1WKS" in Path(image).name
    return results

def run_batch(folder: str, subject_filter: list[str] | None = None) -> pd.DataFrame:
    subjects = [s for s in os.listdir(folder) if "patient" in s]
    if subject_filter:
        subjects = [s for s in subjects if s in subject_filter]

    images = []
    for subject in sorted(subjects):
        images.extend(glob.glob(os.path.join(folder, subject, "T1W*.nii.gz")))

    rows = []
    for i, image in enumerate(images, 1):
        try:
            rows.append(process_image(image))
            print(f"  [{i}/{len(images)}] done")
        except Exception as e:
            print(f"  [{i}/{len(images)}] ERROR {image}: {e}")

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["contrast", "enhanced"])
    return pd.DataFrame({
        "count": g["id"].count(),
        "vessel_ratio_mean": g["vessel_ratio"].mean(),
        "vessel_ratio_min": g["vessel_ratio"].min(),
        "vessel_ratio_max": g["vessel_ratio"].max(),
        "bright_voxel_fraction_mean": g["bright_voxel_fraction"].mean(),
        "bright_voxel_fraction_min": g["bright_voxel_fraction"].min(),
        "bright_voxel_fraction_max": g["bright_voxel_fraction"].max(),
    })


# # %% TEST ON RAW DATA
# folder_raw = "/Users/mathilderipart/Documents/work/260624_BMEIS_hackathon/open_ms_data/cross_sectional/coregistered"
# df_raw = run_batch(folder_raw)
# out = summarise(df_raw)
# print(out)

# %% TEST ON SYNTHETIC ARTEFACTS DATA
folder_art = "/Users/mathilderipart/Documents/work/260624_BMEIS_hackathon/open_ms_data_artefacts/cross_sectional/coregistered"
df_art = run_batch(folder_art, subject_filter=["patient01"])
out2 = summarise(df_art)
print(out2)

# %%
