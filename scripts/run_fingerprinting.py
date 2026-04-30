from pathlib import Path
import argparse
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import fingerprint


def load_condition_matrices(processed_root, condition, subjects=None):
    processed_root = Path(processed_root)
    matrices = {}

    if subjects is None:
        subject_dirs = [p for p in processed_root.iterdir() if p.is_dir()]
    else:
        subject_dirs = [processed_root / subject for subject in subjects]

    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.name
        matrix_path = subject_dir / f"{condition}_fc.npy"

        if not matrix_path.exists():
            continue

        matrix = np.load(matrix_path)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"FC matrix for subject {subject_id} is not square: {matrix_path}"
            )

        matrices[subject_id] = matrix

    return matrices


def run_identification(target_matrices, database_matrices, exclude_self=False):
    if exclude_self:
        correct = 0
        predictions = {}

        for true_subject, target_matrix in target_matrices.items():
            database_without_self = {
                subject_id: matrix
                for subject_id, matrix in database_matrices.items()
                if subject_id != true_subject
            }

            if not database_without_self:
                raise ValueError(
                    "Database is empty after excluding self; need at least 2 subjects."
                )

            predicted_subject, _ = fingerprint.identify_subject(
                target_matrix,
                database_without_self
            )

            predictions[true_subject] = predicted_subject

            if predicted_subject == true_subject:
                correct += 1

        accuracy = correct / len(target_matrices)
        return accuracy, predictions

    return fingerprint.fingerprint_accuracy(target_matrices, database_matrices)


def main():
    parser = argparse.ArgumentParser(
        description="Run subject fingerprinting from saved FC matrices."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data_processed",
        help="Root folder containing per-subject FC files."
    )
    parser.add_argument(
        "--target-condition",
        required=True,
        help="Condition used for target matrices (example: rfMRI_REST1)."
    )
    parser.add_argument(
        "--database-condition",
        required=True,
        help="Condition used for database matrices (example: rfMRI_REST2)."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional explicit subject IDs to include."
    )
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude each subject from its own database during identification."
    )

    args = parser.parse_args()

    if not args.processed_root.exists():
        raise FileNotFoundError(f"Missing processed root: {args.processed_root}")

    target_all = load_condition_matrices(
        processed_root=args.processed_root,
        condition=args.target_condition,
        subjects=args.subjects,
    )
    database_all = load_condition_matrices(
        processed_root=args.processed_root,
        condition=args.database_condition,
        subjects=args.subjects,
    )

    common_subjects = sorted(set(target_all) & set(database_all))

    if len(common_subjects) < 2:
        raise ValueError(
            "Need at least 2 subjects with both target and database condition FC files."
        )

    target_matrices = {subject: target_all[subject] for subject in common_subjects}
    database_matrices = {subject: database_all[subject] for subject in common_subjects}

    accuracy, predictions = run_identification(
        target_matrices=target_matrices,
        database_matrices=database_matrices,
        exclude_self=args.exclude_self
    )

    print("Fingerprinting complete")
    print(f"Target condition:   {args.target_condition}")
    print(f"Database condition: {args.database_condition}")
    print(f"Subjects used:      {len(common_subjects)}")
    print(f"Exclude self:       {args.exclude_self}")
    print(f"Accuracy:           {accuracy:.4f}")
    print("")
    print("Predictions")
    for subject in common_subjects:
        print(f"{subject} -> {predictions[subject]}")


if __name__ == "__main__":
    main()
