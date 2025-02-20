import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["image.origin"] = "lower"

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

def plot_trajectory(trial, stops, ax):
    # arena and targets
    targets = trial.draw_target(ax, alpha=0.0, draw_hex=True)
    trial.draw_boundary(ax)

    # targets and tile visits
    for idx in range(len(targets)):
        if stops.target_visit[idx]:
            targets[idx].set_facecolor('orange')
            targets[idx].set_alpha(0.6)

    if stops.start_target is not None:
        targets[stops.start_target].set_facecolor('g')
        targets[stops.start_target].set_alpha(0.6)

    targets[stops.end_target].set_facecolor('r')
    targets[stops.end_target].set_alpha(0.6)

    # plot the trajectory with changing color
    points = np.array([trial.x, trial.y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(trial.time.min(), trial.time.max())

    # create a LineCollection with color based on the normalized time values
    lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.5)
    lc.set_array(trial.time)
    lc.set_linewidth(1.0)
    ax.add_collection(lc)

    # plot stop locations
    ax.scatter(trial.chirp_x, trial.chirp_y, color='orange')
    ax.plot(trial.x[0], trial.y[0], 'g*', markersize=10)
    ax.plot(trial.x[-1], trial.y[-1], 'r*', markersize=10)

    # axis format
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(0, 2300)
    ax.set_ylim(-100, 2200)
    ax.set_aspect('equal')
    
    return lc

def plot_trial(trial):
    '''
    Plot a single trial with detailed 
    trajectory information and a summary panel.
    '''
    stops = trial.stop_data() # get stop (chirp) data    

    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 6))  # Adjust width for new panels
    gs = GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[3, 3, 1], figure=fig)

    # Main trajectory plot
    ax = fig.add_subplot(gs[:, 0])  # Occupies all rows of first column
    lc = plot_trajectory(trial, stops, ax)
    
    # Add panels
    ax_text = fig.add_subplot(gs[0, 1])
    ax_line = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[2, 1])

    # add text
    ax_text.axis('off')
    catch_text = f'Catch {trial.trial_idx + 1}' if trial.catch else 'EOS'
    text_content = (
        f"Session {trial.session}, Animal {trial.name}, {catch_text}\n\n"
        f"# of chirps: {stops.loc.shape[1]}\n"
        f"# of chirp bouts: {stops.bout_loc.shape[1]}\n"
        f"# of tile checks: {stops.target_unique_visit()}")

    ax_text.text(0.05, 0.5, text_content, fontsize=14,
                verticalalignment='center',
                transform=ax_text.transAxes)

    # plot chirp bouts
    for start, end in zip(stops.bout_start, stops.bout_end):
        ax_line.axvspan(start, end, ymin=0, ymax=1,
                        facecolor='orange',
                        alpha=0.5, edgecolor='none')

        if start == end:
            ax_line.axvline(start, color='orange', linewidth=1.0)

    if trial.catch and stops.t.size > 0:
        ax_line.axvline(stops.t[-1], color='red', linewidth=2.5)

    ax_line.set_xlim(0, trial.time.max())
    ax_line.set_xticks([])
    ax_line.set_yticks([])
    ax_line.set_title('Chirp Bouts')

    # add colorbar
    cbar = fig.colorbar(lc, cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Time (s)')

    plt.tight_layout()
    return fig