import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["image.origin"] = "lower"

import os, tempfile
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps
from PIL import Image

# Create a color map for keypoints
NUM_POINTS = 37
KP_COLORS = []
cmap = colormaps.get_cmap('Spectral')
for i in range(NUM_POINTS):
    KP_COLORS.append(cmap(i / (NUM_POINTS - 1)))
KP_COLORS = np.array(KP_COLORS)

SEC_TO_MS = 1000

def plot_arena(trial, stops, ax, full_arena=False):
    '''
    Plot arena with target visits and chirp locations.
    '''
    # arena and targets
    if full_arena:
        targets = trial.draw_target(ax, alpha=0.2, draw_hex=False)
        trial.draw_arena(ax, alpha=0.2)
    else:
        targets = trial.draw_target(ax, alpha=0.0, draw_hex=True)

    trial.draw_boundary(ax)

    # start and end tiles
    if stops.start_target is not None:
        targets[stops.start_target].set_facecolor('g')
        targets[stops.start_target].set_alpha(0.75)

    targets[stops.end_target].set_facecolor('r')
    targets[stops.end_target].set_alpha(0.75)

    # axis format
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(0, 2300)
    ax.set_ylim(-100, 2200)
    ax.set_aspect('equal')

    return targets

def plot_trajectory(trial, stops, ax, full_arena=False):
    '''
    Plot trajectory of a single trial with
    stops and target visits.
    '''
    targets = plot_arena(trial, stops, ax, full_arena)

    # targets and tile visits
    for idx in range(len(targets)):
        if stops.target_visit[idx]:
            if idx == stops.start_target or \
                idx == stops.end_target:
                 continue

            targets[idx].set_facecolor('orange')
            targets[idx].set_alpha(0.6)

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

def plot_trial(trial, full_arena=False):
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
    lc = plot_trajectory(trial, stops, ax, full_arena)

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

def movie_to_gifs(movie_frames, frame_rate, pre, gif_filename):
    n_sub = np.ceil(np.sqrt(movie_frames.shape[0])).astype(int)
    fig, axes = plt.subplots(n_sub, n_sub, figsize=(n_sub * 3, n_sub * 3), dpi=150)

    # Create a temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_paths = []

        for frame_idx in range(movie_frames.shape[1]):
            for i, ax in enumerate(axes.flat):
                ax.clear()
                frame = movie_frames[i, frame_idx]
                ax.imshow(frame, cmap='gray')

                # Add frame title and markers
                if i == 1:
                    ax.title.set_text('Time %.1f ms' % (frame_idx / frame_rate * SEC_TO_MS))
                if frame_idx >= int(frame_rate * pre):
                    ax.scatter(975, 975, s=400, marker='s', color='tab:blue')

                ax.set_xlim(0, 1024)
                ax.set_ylim(0, 1024)
                ax.invert_yaxis()
                ax.axis("off")  # Hide axes

            # Save the current figure as an image file
            fig.tight_layout()
            fig.canvas.draw()
            frame_img = Image.fromarray(np.array(fig.canvas.buffer_rgba()))

            # Save to temporary directory
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            frame_img.save(frame_path)
            frame_paths.append(frame_path)

        # Load all saved frames and create the final GIF
        images = [Image.open(fp) for fp in frame_paths]
        images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=10, loop=0)

    plt.close(fig)  # Close to free memory

def pose_to_gifs(pose_frames, frame_rate, pre, center, rotate, gif_filename, exclude_points=None):
    # Create a list of combined frames
    combined_frames = []

    n_sub = np.ceil(np.sqrt(pose_frames.shape[0])).astype(int)
    for frame_idx in range(pose_frames.shape[2]):
        fig, axes = plt.subplots(n_sub, n_sub, figsize=(n_sub * 3, n_sub * 3), dpi=100)

        # Plot each movie's frame
        for i, ax in enumerate(axes.flat):
            pose_frame = pose_frames[i, :, frame_idx].reshape(-1, 2)

            for j in range(pose_frame.shape[0]):
                # exclude points
                if exclude_points is not None and j in exclude_points:
                    continue

                # plot the keypoints
                ax.scatter(pose_frame[j, 0], pose_frame[j, 1],
                           c=KP_COLORS[j], alpha=0.90, marker='+')

            # write out some information
            if i == 1:
                ax.title.set_text('Time %.1f ms' % (frame_idx / frame_rate * SEC_TO_MS))
            if frame_idx >= int(frame_rate * pre):
                if center:
                    ax.scatter(465, -465, s=400, marker='s', color='tab:blue')
                else:
                    ax.scatter(975, 50, s=400, marker='s', color='tab:blue')

            if center:
                ax.set_xlim(-512, 512)
                ax.set_ylim(-512, 512)
            else:
                ax.set_xlim(0, 1024)
                ax.set_ylim(0, 1024)

            if not rotate:
                ax.invert_yaxis()

            # put a box with no ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_visible(True)  # Ensure spines are visible
                spine.set_linewidth(1)

        # Save the current figure as an image in memory
        fig.tight_layout()
        fig.canvas.draw()
        frame_img = Image.fromarray(np.array(fig.canvas.buffer_rgba()))
        combined_frames.append(frame_img)

        plt.close(fig)  # Close to free memory

    combined_frames[0].save(gif_filename, save_all=True,
                            append_images=combined_frames[1:],
                            duration=10, loop=0)