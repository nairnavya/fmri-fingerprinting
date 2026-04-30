from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]  #root directory is defined as this file's path, going back two steps (scripts -> fmri fingerprinting)
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from src import fingerprint
from src import figures

#define constants 

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "figures" 

SUBJECTS = [
        103414, 105115, 110411, 113619, 115320, 133827, 135932, 136833, 149539,
        151223, 151627, 158540, 175439, 193239, 205725, 214423, 298051, 448347,
        581349, 654754, 702133, 788876, 857263, 885975, 932554, 984472, 992774
    ]

#create dictionaries 
rest1_matrices = {}
rest2_matrices = {}

for subject in SUBJECTS:

    subject_dir_rest1 = DATA_DIR / f"{subject}" / f"rfMRI_REST1_fc.npy"
    subject_dir_rest2 = DATA_DIR / f"{subject}" / f"rfMRI_REST2_fc.npy"

    rest1_matrices[subject] = np.load(subject_dir_rest1)
    rest2_matrices[subject] = np.load(subject_dir_rest2)


#concatenate correlation coefficients

self_scores = []    
other_scores = []

for subject_a, matrix_a in rest1_matrices.items():  #think of the math here: 27 individuals ran a loop comparing each matrix to the other... 27^2 possible r values: 27 out of those similar to each other 702 not similar 

    for subject_b, matrix_b in rest2_matrices.items():

        r_score = fingerprint.matrix_similarity(matrix_a, matrix_b)

        if subject_a == subject_b:
            self_scores.append(r_score)
        else: 
            other_scores.append(r_score)


#plot histogram

figures.plot_similarity_distribution(
    self_scores,
    other_scores,
    title= "Histogram_similarity_distribution",  # title
    output_dir= OUTPUT_DIR   # output dir
)

#mean scores : print 

print(f"Mean self similarity: {np.mean(self_scores):.3f}")
print(f"Mean other similarity: {np.mean(other_scores):.3f}")
    

