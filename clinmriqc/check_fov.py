import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_fill_holes



def check_fov(input_scan: str, brain_mask: str, margin_threshold: int = 5) -> dict:
    """
    Check whether an MRI scan has full field-of-view (FOV).

    Args:
        input_scan (str): Path to input nifti scan.
        brain_mask (str): Path to brain mask.
        margin_threshold (int): Minimum mm between mask and scan edge. Defaults to 5.

    Returns:
        dict: {
            "Full field-of-view check": "PASSED" or "FAILED",
            "Checks failed": which check is failed (if any),
            "Margin used": int
        }
    """
    # print(f"Checking FOV for scan: {input_scan}")
    # get brain mask
    img_mask = nib.load(brain_mask)
    img_mask = nib.as_closest_canonical(img_mask)  # force RAS orientation
    data_mask = img_mask.get_fdata()
    
    mask = data_mask > 0


    # get scan signal (non-zero voxels)
    img_scan = nib.load(input_scan)
    img_scan = nib.as_closest_canonical(img_scan)  # force RAS orientation
    data_scan = img_scan.get_fdata()
    scan_signal = data_scan > 0

    # fill holes in scan signal
    scan_signal = binary_fill_holes(scan_signal)


    # TEST: save scan signal as a mask for visualisation
    #scan_signal_img = nib.Nifti1Image(scan_signal.astype(np.uint8), img_scan.affine, img_scan.header)
    #nib.save(scan_signal_img, 'TEST_scan_signal_mask.nii.gz')


    coords_scan = np.where(scan_signal)
    scan_min_x, scan_max_x = coords_scan[0].min(), coords_scan[0].max()
    scan_min_y, scan_max_y = coords_scan[1].min(), coords_scan[1].max()
    scan_min_z, scan_max_z = coords_scan[2].min(), coords_scan[2].max()


    check_passes = 0

    cutoff_axes = []

    # check 1: if any voxels in the brain mask are touching edge of scan or outer edge of scan signal
    if np.any(mask[0, :, :])  or np.any(mask[-1, :, :]) or np.any(mask[scan_min_x, :, :]) or np.any(mask[scan_max_x, :, :]): cutoff_axes.append('x axis failed')
    if np.any(mask[:, 0, :])  or np.any(mask[:, -1, :]) or np.any(mask[:, scan_min_y, :]) or np.any(mask[:, scan_max_y, :]): cutoff_axes.append('y axis failed')
    if np.any(mask[:, :, 0])  or np.any(mask[:, :, -1]) or np.any(mask[:, :, scan_min_z]) or np.any(mask[:, :, scan_max_z]): cutoff_axes.append('z axis failed')
    
    if not cutoff_axes:
        cutoff_axes.append('Passed')
        check_passes +=1




    # check 2: if the brain mask is within a certain margin of the edge of the max/min scan signal (default=5 voxels)
    cutoff_axes_2 = []

    # convert 5mm margin to voxels 
    voxel_spacing = img_scan.header.get_zooms()[:3]
    # print(f"voxel spacing: {voxel_spacing} mm")
    margin_threshold_vox  = int(round(margin_threshold / voxel_spacing[0])) 

    # erode the scan signal by the margin
    scan_signal_eroded = binary_erosion(scan_signal, iterations=margin_threshold_vox)
    too_close = mask & ~scan_signal_eroded

    if np.any(too_close): # if any mask voxels are outside of eroded scan signal
        coords_close = np.where(too_close)
    
        min_x, max_x = coords_close[0].min(), coords_close[0].max()
        min_y, max_y = coords_close[1].min(), coords_close[1].max()
        min_z, max_z = coords_close[2].min(), coords_close[2].max()
    
    
        # flag which axes have the problem
        if min_x <= scan_min_x + margin_threshold_vox or max_x >= scan_max_x - margin_threshold_vox:
            cutoff_axes_2.append('x axis failed')
        if min_y <= scan_min_y + margin_threshold_vox or max_y >= scan_max_y - margin_threshold_vox:
            cutoff_axes_2.append('y axis failed')
        if min_z <= scan_min_z + margin_threshold_vox or max_z >= scan_max_z - margin_threshold_vox:
            cutoff_axes_2.append('z axis failed')

    if not cutoff_axes_2:
        cutoff_axes_2.append('Passed')
        check_passes +=1



    # check 3: calculate the distance transform of the scan signal and find the minimum distance to the mask
    distance_check = []
    scan_distance = distance_transform_edt(scan_signal) # this finds the distance of each voxel in scan signal to the edge of scan signal

    # get the distance values only where the mask is
    mask_distances = scan_distance[mask]

    # minimum distance between mask and scan signal boundary 
    min_distance = mask_distances.min()
    # print(f"minimum distance between mask and scan edge: {min_distance} voxels")

    # are there zero voxels inside the mask?
    holes = mask & ~scan_signal
    # print(f"holes inside mask: {np.sum(holes)}")

    if min_distance < 1:
        distance_check.append('Failed')
    else: 
        distance_check.append('Passed')
        check_passes +=1



    # calculate overall p/f


    overall = "Passed" if check_passes >= 2 else "Failed"
    

    return {
            "Check 1 (scan edge proximity)": cutoff_axes,
            "Check 2 (margin proximity)": cutoff_axes_2,
            "Check 3 (distance check)": distance_check,
            "Overall": overall
            }
