import cv2
import numpy as np
import tqdm
import os
from .data_loader import ZABER_TO_MM, TILE_CENTER, TILE_RAD_MM, TILE_ANGLE

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
matplotlib.rcParams["image.origin"] = "lower"

class SessionData:

    def __init__(self, name, ses, df, video_path):
        # name and session
        self.name = name
        self.session = ses

        # x-y corrdinates
        self.zaber_x = df['zaber_x'].to_numpy()
        self.zaber_y = df['zaber_y'].to_numpy()

        # use mm units
        self.x = self.zaber_x * ZABER_TO_MM
        self.y = self.zaber_y * ZABER_TO_MM

        # cricket tiles
        self.zaber_target = self._target(df['locations'][0])
        self.target = self.zaber_target * ZABER_TO_MM

        # time
        self.time = df['relative_time'].to_numpy()
        self.frame = df['frame_no'].to_numpy().astype(int)

        # chirp
        self.chirped = self._int_array(df['chirped'].to_numpy())
        self.chirp_loc = self._int_array(df['chirp_loc'].to_numpy()) - 1
        self.chirp_bout = self._int_array(df['chirp_bouts'].to_numpy())
        self.triggered = self._int_array(df['triggered'].to_numpy())

        # video
        self.video = cv2.VideoCapture(video_path)

    def _target(self, loc):
        loc = loc[1:-1].split(',')
        loc = np.array([float(l) for l in loc])
        loc_x = loc[0::2]
        loc_y = loc[1::2]

        return np.array([loc_x, loc_y])

    def _int_array(self, arr):
        arr[np.isnan(arr)] = 0
        return arr.astype(int)

    def get_frame(self, index):
    # get frame from video
        self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        _, frame = self.video.read()

        return cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

    def _draw_arena(self, plot_ax):
            for idx in range(len(TILE_CENTER[0])):
                plot_ax.add_patch(patches.RegularPolygon(
                    (TILE_CENTER[0][idx],
                     TILE_CENTER[1][idx]),
                    numVertices=6, radius=TILE_RAD_MM,
                    orientation=TILE_ANGLE,
                    facecolor='w', edgecolor='g',lw=1))

    def _frame_index(self, trial_idx):
        ISI_FRAME = 540

        trigger_time = np.where(self.triggered == 1)[0]
        session_length = self.time.shape[0]

        # if no cricket catch
        if len(trigger_time) == 0:
            return 0, session_length

        # with cricket catch
        if trial_idx == 0:
            base_shift = 0
            n_frame = trigger_time[0] + ISI_FRAME
        else:
            base_shift = trigger_time[trial_idx - 1] + int(ISI_FRAME / 2)
            n_frame = trigger_time[trial_idx] - base_shift + ISI_FRAME

        # check if end of session
        if base_shift + n_frame > session_length:
            n_frame = session_length - base_shift

        return base_shift, n_frame

    def trial_video(self, trial_idx):
        base_shift, n_frame = self._frame_index(trial_idx)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # left plot
        # draw tiles
        self._draw_arena(axs[0])

        # trajectory and target
        ll, = axs[0].plot([], [], 'orange', alpha=0.5)
        targets = [axs[0].scatter(*target, s=125, alpha=0.5,
                                facecolors='r',
                                edgecolors='none')
                                for target in self.target.T]

        radius = 35
        circle = plt.Circle((0, 0), radius, color='tab:blue',
                            linewidth=2, fill=False)
        axs[0].add_patch(circle)

        # right plot
        im = axs[1].imshow(self.get_frame(0), cmap='gray')
        axs[1].invert_xaxis()

        # axis format
        axs[0].set_xlim(0, 2400)
        axs[0].set_ylim(0, 2400)
        axs[0].set_aspect('equal')

        axs[1].axis('off')
        axs[1].set_aspect('equal')

        # variables to keep track of chirps
        self._chirp_count = 0
        self._chirp_active = False
        self._chirp_point = None
        self._step_count = 0
        pbar = tqdm.tqdm(total=n_frame + 1, position=0, leave=True)

        def animate(i):
            axs[0].set_title('Frame %d, Time %.3f Sec' % (i, self.time[i]))

            # active -> inactive
            if self._chirp_active:
                self._step_count += 1
                if self._step_count == 10:
                    self._chirp_active = False
                    self._chirp_point.set_facecolor('r')
                    self._step_count = 0

            # inactive -> active
            if self.chirped[i] == 1:
                self._chirp_count += 1
                self._chirp_active = True

                self._chirp_point = targets[self.chirp_loc[i]]
                self._chirp_point.set_facecolor('b')

                axs[0].scatter(self.x[i], self.y[i], marker='+', c='r')

                # write text next to the chirp point
                radius = 35
                angle = np.random.uniform(-np.pi, np.pi)
                x = self.x[i] + radius * np.cos(angle)
                y = self.y[i] + radius * np.sin(angle)
                axs[0].text(x, y, str(self._chirp_count), fontsize=10, color='black')

            ll.set_data(self.x[:i], self.y[:i])
            circle.center = (self.x[i], self.y[i])
            im.set_data(self.get_frame(i))

            pbar.update(1)

        # shift indexing
        animate_shifted = lambda i: animate(i + base_shift)

        # create video
        ani = animation.FuncAnimation(fig, animate_shifted,
                                    frames=n_frame,
                                    interval=10)

        # video path
        target_fps = 10
        file_path = os.path.join('./data', self.name)
        file_name = 's%d_t%d' % (self.session, trial_idx) + '.mp4'

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # save video
        writer = animation.writers['ffmpeg'](fps=target_fps, bitrate=4096)
        ani.save(os.path.join(file_path, file_name), writer=writer)
        pbar.close()