import numpy as np
import nibabel as nib


# ----- LOADING .NII FILE VOXEL DATA -----
def load_nifti_data(nifti_path):
    img = nib.load(str(nifti_path)) # "open the file"
    data = img.get_fdata() # "give me the actual numbers"

    return data

# ----- LOADING ATLAS FILE VOXEL DATA (WHICH REGION DOES EACH VOXEL BELONG TO) -----
def load_atlas(atlas_path):
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()

    return atlas_data


# ----- AVERAGING ACTIVITY OF ALL VOXELS IN EACH REGION TO OBTAIN (T × nodes) -----
# BOLD NIfTI: X × Y × Z × T
# Atlas:      X × Y × Z
# Output:     T × nodes = T x 268
def extract_atlas_timeseries(nifti_path, atlas_path):
    bold = load_nifti_data(nifti_path)
    atlas = load_atlas(atlas_path)

    if bold.shape[:3] != atlas.shape:
        raise ValueError(
            f"BOLD spatial shape {bold.shape[:3]} does not match atlas shape {atlas.shape}"
        )

    labels = np.unique(atlas)
    labels = labels[labels != 0]

    timeseries = []

    for label in labels:
        mask = atlas == label # finds all (x,y,z) where mask == True

        voxel_timeseries = bold[mask, :] # pulls voxels where mask == True out; keeps their full time series

        region_mean = voxel_timeseries.mean(axis=0) # average them → 1 signal per region

        timeseries.append(region_mean)

    timeseries = np.array(timeseries).T

    return timeseries