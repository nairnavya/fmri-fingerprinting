from pathlib import Path


from src.load_data import find_condition_files
from src.preprocess import load_motion_regressors, preprocess_node_timeseries
from src.apply_atlas import extract_atlas_timeseries


def main():
    subject_dir = Path("data/raw/100307")
    atlas_path = Path("atlas/shen_268_atlas.nii.gz")

    condition = "tfMRI_WM"


    files = find_condition_files(subject_dir, condition)


    for direction in ["LR", "RL"]:
        bold_path = files[direction]["bold"]
        motion_path = files[direction]["motion"]

        print("Processing:", condition, direction)

        # raw NIfTI:
        # X × Y × Z × T
        node_timeseries = extract_atlas_timeseries(
            nifti_path=bold_path,
            atlas_path=atlas_path
        )
        # after atlas:
        # T × 268

        # motion regressors:
        # T × 12
        motion_regressors = load_motion_regressors(motion_path)

        # cleaned node time series:
        # T × 268
        cleaned_timeseries = preprocess_node_timeseries(
            node_timeseries=node_timeseries,
            motion_regressors=motion_regressors,
            tr=0.72
        )

        print("Cleaned shape:", cleaned_timeseries.shape)

    
    
    # @TODO #
    # Left to implement: 
    # - concatenate LR/RL
    # - compute connectivity matrix
    # - fingerprint subjects


if __name__ == "__main__":
    main()