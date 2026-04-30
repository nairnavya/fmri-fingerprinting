from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import fingerprint


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CONDITION_A = "rfMRI_REST1"
CONDITION_B = "rfMRI_REST2"


def _load_fc_matrix(matrix_path: Path, subject_id: str) -> np.ndarray:
    """Load one FC matrix and validate square shape."""
    matrix = np.load(matrix_path)

    if matrix.ndim != 2:
        raise ValueError(
            f"FC matrix for subject {subject_id} is not 2D: {matrix_path} (shape={matrix.shape})"
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"FC matrix for subject {subject_id} is not square: {matrix_path} (shape={matrix.shape})"
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            f"FC matrix for subject {subject_id} must be at least 2x2: {matrix_path}"
        )

    return matrix


def load_rest_matrices(processed_root: Path):
    """
    Scan processed_root and load subjects with both REST1 and REST2 FC matrices.

    Returns
    -------
    included_subjects : list[str]
    rest1_matrices : dict[str, np.ndarray]
    rest2_matrices : dict[str, np.ndarray]
    """
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed folder does not exist: {processed_root}")

    included_subjects = []
    rest1_matrices = {}
    rest2_matrices = {}

    for subject_dir in sorted([p for p in processed_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        subject_id = subject_dir.name
        rest1_path = subject_dir / f"{CONDITION_A}_fc.npy"
        rest2_path = subject_dir / f"{CONDITION_B}_fc.npy"

        if not rest1_path.exists() or not rest2_path.exists():
            continue

        rest1_matrix = _load_fc_matrix(rest1_path, subject_id)
        rest2_matrix = _load_fc_matrix(rest2_path, subject_id)

        if rest1_matrix.shape != rest2_matrix.shape:
            raise ValueError(
                "REST1 and REST2 matrix shapes do not match for subject "
                f"{subject_id}: {rest1_matrix.shape} vs {rest2_matrix.shape}"
            )

        included_subjects.append(subject_id)
        rest1_matrices[subject_id] = rest1_matrix
        rest2_matrices[subject_id] = rest2_matrix

    return included_subjects, rest1_matrices, rest2_matrices


def save_predictions_csv(output_csv: Path, matches):
    """Save prediction rows with columns true_subject, predicted_subject, correct."""
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["true_subject", "predicted_subject", "correct"],
        )
        writer.writeheader()

        for row in matches:
            writer.writerow(
                {
                    "true_subject": row["true_subject"],
                    "predicted_subject": row["predicted_subject"],
                    "correct": row["correct"],
                }
            )


def save_labels_txt(output_txt: Path, row_subjects, column_subjects):
    """Save row/column subject order used by the similarity matrix."""
    lines = ["rows (target subjects):"]
    lines.extend(str(subject) for subject in row_subjects)
    lines.append("")
    lines.append("columns (database subjects):")
    lines.extend(str(subject) for subject in column_subjects)

    output_txt.write_text("\n".join(lines) + "\n")


def run_direction(
    *,
    target_label: str,
    database_label: str,
    target_matrices,
    database_matrices,
    predictions_csv_path: Path,
    similarity_npy_path: Path,
    labels_txt_path: Path,
):
    """Run one fingerprinting direction and save outputs."""
    accuracy, _, details = fingerprint.fingerprint_accuracy(
        target_matrices=target_matrices,
        database_matrices=database_matrices,
    )

    correct_count = details["correct_count"]
    total_subjects = details["total_subjects"]
    matches = details["matches"]
    similarity_matrix = details["similarity_matrix"]
    row_subjects = details["target_subjects"]
    column_subjects = details["database_subjects"]

    save_predictions_csv(predictions_csv_path, matches)
    np.save(similarity_npy_path, similarity_matrix)
    save_labels_txt(labels_txt_path, row_subjects, column_subjects)

    print("=" * 72)
    print(f"Fingerprinting: {target_label} -> {database_label}")
    print(f"Included subjects: {total_subjects}")
    print(f"Target condition:  {target_label}")
    print(f"Database condition:{database_label}")
    print(f"Correct / Total:   {correct_count} / {total_subjects}")
    print(f"Accuracy:          {accuracy * 100:.2f}%")
    print("Per-subject predictions:")

    for row in matches:
        status = "correct" if row["correct"] else "incorrect"
        print(
            f"  true={row['true_subject']}  predicted={row['predicted_subject']}  {status}"
        )

    print(f"Saved predictions CSV: {predictions_csv_path}")
    print(f"Saved similarity NPY:  {similarity_npy_path}")
    print(f"Saved labels TXT:      {labels_txt_path}")
    print()


def main():
    included_subjects, rest1_matrices, rest2_matrices = load_rest_matrices(PROCESSED_ROOT)

    if len(included_subjects) == 0:
        raise ValueError(
            f"No subjects with both {CONDITION_A}_fc.npy and {CONDITION_B}_fc.npy were found in {PROCESSED_ROOT}."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_direction(
        target_label=CONDITION_A,
        database_label=CONDITION_B,
        target_matrices=rest1_matrices,
        database_matrices=rest2_matrices,
        predictions_csv_path=RESULTS_DIR / "rest1_to_rest2_predictions.csv",
        similarity_npy_path=RESULTS_DIR / "rest1_to_rest2_similarity.npy",
        labels_txt_path=RESULTS_DIR / "rest1_to_rest2_labels.txt",
    )

    run_direction(
        target_label=CONDITION_B,
        database_label=CONDITION_A,
        target_matrices=rest2_matrices,
        database_matrices=rest1_matrices,
        predictions_csv_path=RESULTS_DIR / "rest2_to_rest1_predictions.csv",
        similarity_npy_path=RESULTS_DIR / "rest2_to_rest1_similarity.npy",
        labels_txt_path=RESULTS_DIR / "rest2_to_rest1_labels.txt",
    )


if __name__ == "__main__":
    main()
