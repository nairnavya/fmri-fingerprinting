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

