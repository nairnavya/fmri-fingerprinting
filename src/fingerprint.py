# @TODO #

import numpy as np
from scipy.stats import pearsonr

# ----- CALCULATING PEARSON'S SIMILARITY COEFFICIENT B/W TWO MATRICES -----
def matrix_similarity(matrix_a, matrix_b):
    upper_a = matrix_a[np.triu_indices_from(matrix_a, k=1)]
    upper_b = matrix_b[np.triu_indices_from(matrix_b, k=1)]

    r, _ = pearsonr(upper_a, upper_b)

    return r


# ----- CALCULATING PEARSON'S SIMILARITY COEFFICIENT B/W TWO MATRICES -----
def identify_subject(target_matrix, database_matrices):
    # database_matrices: dict where keys are subject IDs and values are FC matrices
    # when was database_matrices created?
    scores = {}

    for subject_id, db_matrix in database_matrices.items():
        scores[subject_id] = matrix_similarity(target_matrix, db_matrix)

    predicted_subject = max(scores, key=scores.get)

    return predicted_subject, scores


def fingerprint_accuracy(target_matrices, database_matrices):
    correct = 0
    predictions = {}

    for true_subject, target_matrix in target_matrices.items():
        predicted_subject, scores = identify_subject(target_matrix, database_matrices)

        predictions[true_subject] = predicted_subject

        if predicted_subject == true_subject:
            correct += 1

    accuracy = correct / len(target_matrices)

    return accuracy, predictions