from pathlib import Path


# ----- LOOKING FOR VALID .NII FILE & MOVEMENT_REGRESSORS.TXT FOR EACH CONDITION -----
def find_condition_files(subject, condition):
    results_dir = Path(subject) / "MNINonLinear" / "Results"

    files = {}

    for direction in ["LR", "RL"]:
        run_name = f"{condition.upper()}_{direction}" # begin with 
        run_dir = results_dir / run_name

        if not run_dir.exists():
            raise FileNotFoundError(f"Missing folder: {run_dir}")

        nii_files = list(run_dir.glob("*clean*.nii*"))

        if len(nii_files) == 0:
            raise FileNotFoundError(f"No cleaned NIfTI found in {run_dir}")
        
        nii_files = [ # filtering out CIFTI files, prefering volumetric NIfTI instead
            f for f in nii_files
            if not str(f).endswith(".dtseries.nii")
            and not str(f).endswith(".dscalar.nii")
        ]

        if len(nii_files) == 0:
            raise FileNotFoundError(f"No volumetric cleaned NIfTI found in {run_dir}")

        if len(nii_files) > 1:
            print("Files after filter:", nii_files)
            raise FileNotFoundError(f"Greater than 1 cleaned NIfTI found in {run_dir}")

        bold_path = nii_files[0]
        motion_path = run_dir / "Movement_Regressors.txt"

        files[direction] = {
            "bold": bold_path,
            "motion": motion_path if motion_path.exists() else None,
        }

    return files


# ----- COLLECTING ALL 6 CONDITION .NII FILES & MOVEMENT_REGRESSORS.TXT FOR EACH SUBJECT -----
def get_subject_files(subject_dir, conditions):
    subject_data = {} 

    for condition in conditions:
        subject_data[condition] = find_condition_files(subject_dir, condition)

    return subject_data

