"""Utilities for functional connectome fingerprinting."""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

import numpy as np
from scipy.stats import pearsonr


SubjectId = Hashable
MatrixDict = Dict[SubjectId, np.ndarray]


def _as_square_matrix(matrix: np.ndarray, matrix_name: str) -> np.ndarray:
    """Validate and return a square 2D matrix as a NumPy array."""
    array = np.asarray(matrix)

    if array.ndim != 2:
        raise ValueError(
            f"{matrix_name} must be a 2D square matrix, but got shape {array.shape}."
        )

    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{matrix_name} must be square, but got shape {array.shape}."
        )

    if array.shape[0] < 2:
        raise ValueError(
            f"{matrix_name} must be at least 2x2 to compute upper-triangle edges."
        )

    return array


def _sorted_subject_ids(matrix_dict: MatrixDict) -> List[SubjectId]:
    """Return subject IDs in deterministic order."""
    return sorted(matrix_dict.keys(), key=str)


def vectorize_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the upper-triangle edges (excluding diagonal) as a 1D vector."""
    array = _as_square_matrix(matrix, "matrix")
    upper_indices = np.triu_indices(array.shape[0], k=1)
    return array[upper_indices]


def matrix_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Compute Pearson similarity between upper-triangle FC edges."""
    array_a = _as_square_matrix(matrix_a, "matrix_a")
    array_b = _as_square_matrix(matrix_b, "matrix_b")

    if array_a.shape != array_b.shape:
        raise ValueError(
            "matrix_a and matrix_b must have the same shape, "
            f"but got {array_a.shape} and {array_b.shape}."
        )

    upper_a = vectorize_upper_triangle(array_a)
    upper_b = vectorize_upper_triangle(array_b)

    r, _ = pearsonr(upper_a, upper_b)

    if np.isnan(r):
        raise ValueError(
            "Pearson correlation is undefined for at least one matrix "
            "(for example, constant upper-triangle values)."
        )

    return float(r)


def identify_subject(
    target_matrix: np.ndarray,
    database_matrices: MatrixDict,
) -> Tuple[SubjectId, Dict[SubjectId, float]]:
    """Identify the most similar subject in the database for one target matrix."""
    if not database_matrices:
        raise ValueError("database_matrices is empty; provide at least one subject matrix.")

    scores: Dict[SubjectId, float] = {}

    for subject_id in _sorted_subject_ids(database_matrices):
        db_matrix = database_matrices[subject_id]
        scores[subject_id] = matrix_similarity(target_matrix, db_matrix)

    best_score = max(scores.values())
    tied_subjects = [sid for sid, score in scores.items() if np.isclose(score, best_score)]
    predicted_subject = sorted(tied_subjects, key=str)[0]

    return predicted_subject, scores


def compute_similarity_matrix(
    target_matrices: MatrixDict,
    database_matrices: MatrixDict,
) -> np.ndarray:
    """
    Compute subject-by-subject similarity matrix.

    Rows follow sorted(target_matrices.keys()) and columns follow
    sorted(database_matrices.keys()).
    """
    if not target_matrices:
        raise ValueError("target_matrices is empty; provide at least one subject matrix.")

    if not database_matrices:
        raise ValueError("database_matrices is empty; provide at least one subject matrix.")

    target_subjects = _sorted_subject_ids(target_matrices)
    database_subjects = _sorted_subject_ids(database_matrices)

    similarity = np.zeros((len(target_subjects), len(database_subjects)), dtype=float)

    for row_idx, target_subject in enumerate(target_subjects):
        target_matrix = target_matrices[target_subject]

        for col_idx, database_subject in enumerate(database_subjects):
            database_matrix = database_matrices[database_subject]
            similarity[row_idx, col_idx] = matrix_similarity(target_matrix, database_matrix)

    return similarity


def fingerprint_accuracy(
    target_matrices: MatrixDict,
    database_matrices: MatrixDict,
):
    """
    Run fingerprint identification across all target subjects.

    Returns
    -------
    accuracy : float
        Correct predictions / total targets.
    predictions : dict
        Mapping true_subject -> predicted_subject.
    details : dict
        Includes correct_count, total_subjects, matches (list of dict rows),
        similarity_matrix, target_subjects, and database_subjects.
    """
    target_subjects = _sorted_subject_ids(target_matrices)
    database_subjects = _sorted_subject_ids(database_matrices)

    similarity_matrix = compute_similarity_matrix(target_matrices, database_matrices)

    predictions: Dict[SubjectId, SubjectId] = {}
    matches = []
    correct_count = 0

    for row_idx, true_subject in enumerate(target_subjects):
        row_scores = similarity_matrix[row_idx]
        best_score = np.max(row_scores)

        best_columns = np.where(np.isclose(row_scores, best_score))[0]
        predicted_candidates = [database_subjects[col_idx] for col_idx in best_columns]
        predicted_subject = sorted(predicted_candidates, key=str)[0]

        is_correct = predicted_subject == true_subject
        if is_correct:
            correct_count += 1

        predictions[true_subject] = predicted_subject
        matches.append(
            {
                "true_subject": true_subject,
                "predicted_subject": predicted_subject,
                "correct": is_correct,
            }
        )

    total_subjects = len(target_subjects)
    accuracy = correct_count / total_subjects

    details = {
        "correct_count": correct_count,
        "total_subjects": total_subjects,
        "matches": matches,
        "similarity_matrix": similarity_matrix,
        "target_subjects": target_subjects,
        "database_subjects": database_subjects,
    }

    return accuracy, predictions, details
