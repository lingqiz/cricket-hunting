from abc import ABC
import cv2
import numpy as np
import tqdm
import os
from scipy.signal import butter, filtfilt

from .data_loader import ZABER_TO_MM, DLC_TO_MM, ISI_FRAME, TRK_CTR, \
    TILE_CENTER, TILE_RAD_MM, TILE_ANGLE, ARENA_CENTER, TILE_RADIUS

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
matplotlib.rcParams["image.origin"] = "lower"


class ArenaMap(ABC):

    def draw_arena(self, plot_ax, alpha=1):
        for idx in range(len(TILE_CENTER[0])):
            plot_ax.add_patch(patches.RegularPolygon(
                (TILE_CENTER[0][idx],
                 TILE_CENTER[1][idx]),
                numVertices=6, radius=TILE_RAD_MM,
                orientation=TILE_ANGLE,
                facecolor='w', edgecolor='g',
                lw=1, alpha=alpha))

    def draw_target(self, plot_ax, alpha=0.5):
        return [plot_ax.scatter(*target, s=125, alpha=alpha,
                                facecolors='r',
                                edgecolors='none')
                for target in self.target.T]

    def get_center(self):
        return ARENA_CENTER


class SessionData(ArenaMap):

    def __init__(self, name, ses, df, video_path):
        # name and session
        self.name = name
        self.session = ses

        # x-y corrdinates, use mm units
        self.zaber_x = df['zaber_x'].to_numpy() * ZABER_TO_MM
        self.zaber_y = df['zaber_y'].to_numpy() * ZABER_TO_MM

        self.dlc_x = (df['dlc_x'].to_numpy() - TRK_CTR) * DLC_TO_MM
        self.dlc_y = (df['dlc_y'].to_numpy() - TRK_CTR) * DLC_TO_MM

        # tracking coordinates
        self.x = self.zaber_x - self.dlc_x
        self.y = self.zaber_y + self.dlc_y

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
        split_path = video_path.split('/')
        split_path[-1] = 'corrected_' + split_path[-1][:-4] + '.mp4'
        corrected_path = '/'.join(split_path)
        if os.path.exists(corrected_path):
            self.video = cv2.VideoCapture(corrected_path)
        else:
            self.video = cv2.VideoCapture(video_path)

        # cricket catch
        self.n_catch = np.sum(self.triggered)
        self.trigger_time = np.where(self.triggered == 1)[0]

    def to_trials(self, non_catch=False):
        if self.n_catch == 0:
            trial = self._construct_trial(0)
            trial.catch = False

            # include non-catch trials
            if non_catch:
                return [trial]
            else:
                return []

        return [self._construct_trial(idx) for idx in range(self.n_catch)]

    def _construct_trial(self, trial_idx):
        start_idx, n_frame = self._trial_index(trial_idx)
        end_idx = start_idx + n_frame

        return TrialData(self, self.name, self.session, trial_idx, self.time[start_idx:end_idx], self.chirped[start_idx:end_idx],
                         self.x[start_idx:end_idx], self.y[start_idx:end_idx], self.target, self.chirp_loc[start_idx:end_idx])

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

    # note that average frame rate * 30 sec ISI ~= 535 frames
    def _trial_index(self, trial_idx, prepend=ISI_FRAME, append=0):
        trigger_time = np.where(self.triggered == 1)[0]
        session_length = self.time.shape[0]

        # if no cricket catch
        if len(trigger_time) == 0:
            return 0, session_length

        # with cricket catch
        if trial_idx == 0:
            start_idx = 0
            n_frame = trigger_time[0] + append

        else:
            start_idx = trigger_time[trial_idx - 1] + prepend
            n_frame = trigger_time[trial_idx] - start_idx + append

        # check if end of session
        if start_idx + n_frame > session_length:
            n_frame = session_length - start_idx

        return start_idx, n_frame

    # append 550 frames to the end of the trial for visualization
    def _frame_index(self, trial_idx):
        return self._trial_index(trial_idx, append=550)

    def all_video(self):
        n_trigger = np.where(self.triggered == 1)[0].shape[0]

        if n_trigger == 0:
            self.trial_video(0)
        else:
            for idx in range(n_trigger):
                self.trial_video(idx)

# TODO: Refactor the code to generate video from trial data
    def trial_video(self, trial_idx):
        print('%s s%d, t%d' % (self.name, self.session, trial_idx))
        self.start_idx, self.n_frame = self._frame_index(trial_idx)

        n_chip = self.chirped[self.start_idx:self.start_idx +
                              self.n_frame].sum()
        print('number of chirps: %d' % n_chip)
        # create a continous color map
        colors = plt.cm.viridis(np.linspace(0, 1, n_chip + 1))

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # left plot
        # draw tiles
        self.draw_arena(axs[0])

        # trajectory and target
        ll, = axs[0].plot([], [], 'orange', alpha=0.5)
        targets = self.draw_target(axs[0])

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
        pbar = tqdm.tqdm(total=self.n_frame + 1, position=0, leave=True)

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

                axs[0].scatter(self.x[i], self.y[i], marker='s',
                               color=colors[self._chirp_count - 1])

            ll.set_data(self.x[self.start_idx:i], self.y[self.start_idx:i])
            circle.center = (self.x[i], self.y[i])
            im.set_data(self.get_frame(i))

            pbar.update(1)

        # shift indexing
        def animate_shifted(i): return animate(i + self.start_idx)

        # create video
        ani = animation.FuncAnimation(fig, animate_shifted,
                                      frames=self.n_frame,
                                      interval=10)

        # video path
        target_fps = 10
        file_path = os.path.join('./data/video', self.name)
        file_name = 's%d_t%d' % (self.session, trial_idx) + '.mp4'

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # save video
        writer = animation.writers['ffmpeg'](fps=target_fps, bitrate=4096)
        ani.save(os.path.join(file_path, file_name), writer=writer)
        pbar.close()

class TrialData(ArenaMap):

    def __init__(self, ses_ref, name, session, trial_idx, time,
                 chirp, x, y, target, chirp_loc, catch=True):

        # record trial information
        self.ses_ref = ses_ref
        self.name = name
        self.session = session
        self.trial_idx = trial_idx

        self.time = time - time[0]
        self.length = time[-1] - time[0]

        self.chirp = chirp
        self.n_chirp = np.sum(chirp)

        self.target = target
        chirp_loc = np.sort(np.unique(chirp_loc))
        assert len(chirp_loc) >= 2 and len(chirp_loc) <= 3
        self.trial_target = target[:, chirp_loc[-1]]

        self.catch = catch

        # x, y data
        self.x = x
        self.y = y

        self._smooth_trajectory()

        # chirp coordinates and time
        self.chirp_x = x[chirp == 1]
        self.chirp_y = y[chirp == 1]
        self.chirp_time = self.time[chirp == 1]

    def _smooth_trajectory(self):
        # sampling rate = 17.8 Hz
        # cutoff frequency = 3.5 Hz
        fs = 17.8
        nyquist = 0.5 * fs
        cutoff = 3.5
        b, a = butter(N=2, Wn=cutoff/nyquist,
                      btype='low', analog=False)

        self.x = filtfilt(b, a, self.x)
        self.y = filtfilt(b, a, self.y)

    def run_distance(self):
        MM_TO_M = 1e-3
        dx = np.diff(self.x)
        dy = np.diff(self.y)

        return np.sum(np.sqrt(dx**2 + dy**2)) * MM_TO_M

    # data reduction into stop locations
    def stop_location(self, rotate=False, center=False, filter_stop=False):
        return StopLocation(np.array([self.chirp_x, self.chirp_y]),
                            np.array([self.x[0], self.y[0]]),
                            np.array([self.x[-1], self.y[-1]]),
                            self.time[self.chirp == 1],
                            self.target, rotate=rotate,
                            center=center, filter_stop=filter_stop)

# class for different data reductions of the trial
class StopLocation(ArenaMap):
    def __init__(self, loc, start, end, t, target,
                 rotate=False, center=False,
                 filter_stop=False):
        self.loc = loc
        self.start = start.reshape(-1, 1)
        self.end = end.reshape(-1, 1)
        self.t = t
        self.target = target

        if rotate:
            self._rotate()

        if center:
            self._center()

        if filter_stop:
            self._filter_stop()

    def _rotate(self):
        center = np.array(self.get_center()).reshape(-1, 1)
        self.start = self.start - center
        self.loc = self.loc - center
        self.end = self.end - center

        self.rot_angle = - np.arctan2(self.end[1], self.end[0]).squeeze() + np.pi
        rotate_matrix = np.array([[np.cos(self.rot_angle), -np.sin(self.rot_angle)],
                                  [np.sin(self.rot_angle), np.cos(self.rot_angle)]])

        self.loc = rotate_matrix @ self.loc
        self.start = rotate_matrix @ self.start
        self.end = rotate_matrix @ self.end
        self.target = rotate_matrix @ self.target

    def _center(self):
        self.start -= self.end
        self.loc -= self.end
        self.target -= self.end
        self.end -= self.end

    # filter out stops that are less than 10 mm from the last one
    def _filter_stop(self, threshold=10):
        index = np.where(self.delta_distance() < threshold)[0] + 1
        self.loc = np.delete(self.loc, index, axis=1)

    def tile_visit_unique(self):
        '''
        Unique tile visit count for the trial,
        excluding the start tile, and including the end tile
        '''
        locations = np.concatenate([self.loc, self.end], axis=1)

        counter = 0
        for idx in range(self.target.shape[1]):
            dist = np.linalg.norm(locations - self.target[:, idx].reshape(-1, 1), axis=0)
            tile_visit = (dist <= TILE_RADIUS).astype(int)
            counter += (np.sum(tile_visit) > 0).astype(int)

        start_ind = np.any(np.linalg.norm(self.target - self.start, axis=0) <= TILE_RADIUS)
        return counter - start_ind.astype(int)

    # useful quantities to calculate
    def distance(self):
        return np.linalg.norm(self.loc, axis=0)

    def angle(self):
        return np.rad2deg(np.arctan2(self.loc[1], self.loc[0]))

    def delta(self):
        return self.loc[:, 1:] - self.loc[:, :-1]

    def delta_distance(self):
        return np.linalg.norm(self.delta(), axis=0)

    def delta_angle(self):
        return np.rad2deg(np.arctan2(self.delta()[1], self.delta()[0]))