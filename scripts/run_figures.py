from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from src import figures 
from src import connectivity


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "figures"

CONDITIONS = [
    "rfMRI_REST1",
    "rfMRI_REST2",
]

SUBJECTS = [
        103414, 105115, 110411, 113619, 115320, 133827, 135932, 136833, 149539,
        151223, 151627, 158540, 175439, 193239, 205725, 214423, 298051, 448347,
        581349, 654754, 702133, 788876, 857263, 885975, 932554, 984472, 992774
    ]

#time series : heatmap and plot 


for subject in SUBJECTS: 


    subject_dir = DATA_DIR / "processed" / f"{subject}"
    print(subject_dir) 
    output_root =  OUTPUT_DIR / f"{subject}" 
    output_root.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:

        LR_file = np.load(subject_dir / f"{condition}_LR_timeseries.npy")
        RL_file = np.load(subject_dir / f"{condition}_RL_timeseries.npy")
        concatenated_file = connectivity.concatenate_runs(LR_file, RL_file)

        fc_matrix = np.load(subject_dir / f"{condition}_fc.npy")

        title_heatmap = f"Heatmap_{subject}_{condition}"
        title_plot = f"20_ROI_Plot_{subject}_{condition}"
        title_matrix = f"Matrix_{subject}_{condition}"

        #heatmap 
        figures.plot_heat_map(concatenated_file, title_heatmap, output_root)
        #plot
        figures.plot_timeseries(concatenated_file, title_plot, output_root)
        #connectivity matrix heatmap 
        figures.plot_fc_matrix(fc_matrix, title_matrix, output_root)
            





