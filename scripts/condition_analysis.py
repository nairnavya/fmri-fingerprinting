from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  #root directory is defined as this file's path, going back two steps (scripts -> fmri fingerprinting)
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from src import fingerprint
import matplotlib.pyplot as plt
from src import figures

#define constants 
OUTPUT_DIR = PROJECT_ROOT / "figures" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SUBJECTS = [
        103414, 105115, 110411, 113619, 115320, 133827, 135932, 136833, 149539,
        151223, 151627, 158540, 175439, 193239, 205725, 214423, 298051, 448347,
        581349, 654754, 702133, 788876, 857263, 885975, 932554, 984472, 992774
    ]

#create dictionary 
rest1_matrices = {}
rest2_matrices = {}

for subject in SUBJECTS:

    subject_dir_rest1 = DATA_DIR / f"{subject}" / f"rfMRI_REST1_fc.npy"
    subject_dir_rest2 = DATA_DIR / f"{subject}" / f"rfMRI_REST2_fc.npy"

    rest1_matrices[subject] = np.load(subject_dir_rest1)
    rest2_matrices[subject] = np.load(subject_dir_rest2)


#function : 
def analyze_subject(subject_id, rest1_matrices, rest2_matrices, output_dir):

    #compare 
    results_rest2_to_rest1 = []
    results_rest1_to_rest2 = []

    subject_id_path = PROJECT_ROOT / "data" / "processed" / f"{subject_id}"
    output_dir_path = OUTPUT_DIR / f"{subject_id}"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    rest2_matrix = np.load(subject_id_path / "rfMRI_REST2_fc.npy")
    rest1_matrix = np.load(subject_id_path / "rfMRI_REST1_fc.npy")


    for subject, matrix in rest1_matrices.items(): 

        r_score = fingerprint.matrix_similarity(rest2_matrix, matrix)  #take single rest 2 and compare to rest 1 
        results_rest2_to_rest1.append({"subject_id": subject, "r_score": r_score})

    for subject, matrix in rest2_matrices.items(): 
        r_score = fingerprint.matrix_similarity(rest1_matrix, matrix) #take single rest 1 and compare to rest 2 
        results_rest1_to_rest2.append({"subject_id": subject, "r_score": r_score})


    df_rest2_to_rest1 = pd.DataFrame(results_rest2_to_rest1).sort_values("r_score", ascending=False)
    df_rest1_to_rest2 = pd.DataFrame(results_rest1_to_rest2).sort_values("r_score", ascending=False)

    figures.plot_similarity_barchart(
        subject_id= subject_id,
        df_rest1_to_rest2=df_rest1_to_rest2,
        df_rest2_to_rest1=df_rest2_to_rest1,
        output_dir= output_dir_path 
    ) 

        # print(f"rest2->rest1: {df_rest2_to_rest1}")
        # print(f"rest1->rest2: {df_rest1_to_rest2}")

    df_rest2_to_rest1.to_csv(OUTPUT_DIR / f"{subject_id}"/ "similarity_analysis_"f"{subject_id}_" "rest2_to_rest1.csv", index=False)
    df_rest1_to_rest2.to_csv(OUTPUT_DIR / f"{subject_id}"/ "similarity_analysis_" f"{subject_id}_" "rest1_to_rest2.csv", index=False)


#calling function for all 
for subject in SUBJECTS:
    print(f"Analyzing subject {subject}...")
    analyze_subject(subject, rest1_matrices, rest2_matrices, OUTPUT_DIR)

