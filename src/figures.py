import numpy as np
import matplotlib.pyplot as plt

#tr as default number, but CAN override it 
#function plots heat map of timeseries with time on x axis and ROIs on y axis

def plot_heat_map(timeseries, title, output_dir, tr=0.72): 

    #timeseries shape (1200, 268)

    n_timepoints = timeseries.shape[0]
    time_axis = np.arange(n_timepoints) * tr #time to repitition

    plt.figure(figsize=(14, 6))
    plt.imshow(
        timeseries.T,        # why do we need to transpose?
        aspect="auto",
        cmap="RdBu_r",
        extent=[time_axis[0], time_axis[-1], 1, 268],
        vmin=-200,
        vmax=200
    )
    plt.colorbar(label="BOLD signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Brain region")
    plt.title(title)
    plt.savefig(output_dir / f"{title}.png")
    plt.close()


def plot_timeseries(timeseries, title, output_dir, tr=0.72): 

    np.random.seed(42)
    selected = np.random.choice(268, 20, replace = False)

    n_timepoints = timeseries.shape[0]
    time_axis = np.arange(n_timepoints) * tr #time to repitition

    plt.figure(figsize=(14, 6))

    for region in selected:

        plt.plot(time_axis, timeseries[:, region], label= f"Region {region}") #plt.plot(x,y) where x is time points and y is value in specific region
    
    plt.legend()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Bold Signal")
    plt.title(title)
    plt.savefig(output_dir / f"{title}.png")
    plt.close()


def plot_fc_matrix(fc_matrix, title, output_dir): 

    # Display the heatmap
    plt.imshow(fc_matrix, cmap='RdBu_r')
    # Add a colorbar
    plt.colorbar(label="Fisher Z-Score")
    plt.xlabel("Brain Regions (ROI)")
    plt.ylabel("Brain Regions (ROI)")
    plt.title(title)
    plt.savefig(output_dir / f"{title}.png")
    plt.close()

def plot_similarity_distribution(self_scores, other_scores, title, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(self_scores, bins=10, alpha=0.7, label="Self", color="red")
    plt.hist(other_scores, bins=30, alpha=0.7, label="Other", color="blue")
    plt.legend()
    plt.xlabel("Pearson Correlation (r)")
    plt.ylabel("Count")
    plt.title(title)
    plt.savefig(output_dir / f"{title}.png")
    plt.close()

def plot_similarity_barchart(subject_id, df_rest1_to_rest2, df_rest2_to_rest1, output_dir):

    # sort both by subject_id
    df1 = df_rest1_to_rest2.sort_values("subject_id")
    df2 = df_rest2_to_rest1.sort_values("subject_id")

    # get self scores
    self_score_1 = df1[df1["subject_id"] == subject_id]["r_score"].values[0]
    self_score_2 = df2[df2["subject_id"] == subject_id]["r_score"].values[0]

    # bar colors
    # colors — compare as strings
    colors1 = ["red" if str(sid) == str(subject_id) else "steelblue" for sid in df1["subject_id"]]
    colors2 = ["red" if str(sid) == str(subject_id) else "steelblue" for sid in df2["subject_id"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT PLOT
    ax1.bar(df1["subject_id"].astype(str), df1["r_score"], color=colors1)
    ax1.axhline( y=self_score_1, color="red", linestyle="--", label= f"{subject_id}self score")
    ax1.set_xlabel("Subject ID")
    ax1.set_ylabel("Pearson Correlation (r)")
    ax1.set_title("REST1 → REST2")
    ax1.tick_params(axis="x", rotation=90)
    ax1.legend()

    # RIGHT PLOT
    ax2.bar(df2["subject_id"].astype(str), df2["r_score"], color=colors2)
    ax2.axhline(y= self_score_2, color="red", linestyle="--", label= f"{subject_id}self score")
    ax2.set_xlabel("Subject ID")
    ax2.set_title("REST2 → REST1")
    ax2.tick_params(axis="x", rotation=90)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"{subject_id}_similarity_barchart.png")
    plt.close()
