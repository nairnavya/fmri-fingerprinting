from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import load_data
from src import preprocess
from src import apply_atlas
from src import connectivity


def process_condition(subject_dir, atlas_path, condition, output_dir):
    files = load_data.find_condition_files(subject_dir, condition)

    run_timeseries = {}

    for direction in ["LR", "RL"]:
        bold_path = files[direction]["bold"]
        motion_path = files[direction]["motion"]

        print(f"Processing {subject_dir.name}: {condition}_{direction}")

        node_ts = apply_atlas.extract_atlas_timeseries(
            nifti_path=bold_path,
            atlas_path=atlas_path
        )

        motion = preprocess.load_motion_regressors(motion_path)

        cleaned_ts = preprocess.preprocess_node_timeseries(
            node_timeseries=node_ts,
            motion_regressors=motion,
            tr=0.72
        )

        run_timeseries[direction] = cleaned_ts

        np.save(
            output_dir / f"{condition}_{direction}_timeseries.npy",
            cleaned_ts
        )

    combined_ts = connectivity.concatenate_runs(
        run_timeseries["LR"],
        run_timeseries["RL"]
    )

    fc_matrix = connectivity.compute_fc_matrix(combined_ts)

    np.save(output_dir / f"{condition}_fc.npy", fc_matrix)

    return fc_matrix


def process_subject(subject_dir, atlas_path, conditions, output_root):
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name

    output_dir = Path(output_root) / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_matrices = {}

    for condition in conditions:
        fc_matrix = process_condition(
            subject_dir=subject_dir,
            atlas_path=atlas_path,
            condition=condition,
            output_dir=output_dir
        )

        subject_matrices[condition] = fc_matrix

    return subject_matrices


def main():

    SUBJECTS = [
        103414, 105115, 110411, 113619, 115320, 133827, 135932, 136833, 149539,
        151223, 151627, 158540, 175439, 193239, 205725, 214423, 298051, 448347,
        581349, 654754, 702133, 788876, 857263, 885975, 932554, 984472, 992774
    ]

    for subjects in SUBJECTS: 


        subject_dir = PROJECT_ROOT / "data" / "raw" / "extracted_data" / f"{subjects}"
        print(subject_dir) 
        atlas_path = PROJECT_ROOT / "atlas" / "shen_2mm_268_parcellation.nii"
        output_root = PROJECT_ROOT / "data" / "processed" 
        conditions = [
            "rfMRI_REST1",
            "rfMRI_REST2",
        ]

        
        results = process_subject(  
            subject_dir=subject_dir,
            atlas_path=atlas_path,
            conditions=conditions,
            output_root=output_root
        )

        print(results.keys())


if __name__ == "__main__":
    main()