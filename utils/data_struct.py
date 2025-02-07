import cv2
import numpy as np
import tqdm
import os
import warnings
import scipy.io
from scipy.signal import butter, filtfilt

from .data_loader import ZABER_TO_MM, DLC_TO_MM, ISI, TRK_CTR, \
    TILE_CENTER, TILE_RAD_MM, TILE_ANGLE, ARENA_CENTER, VERT_TILE, TRIG_RADIUS

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
matplotlib.rcParams["image.origin"] = "lower"


class ArenaMap():

    def __init__(self):
        # tile coordinates [2, N]
        self.tile_center = TILE_CENTER
        self.tiles = np.array([self.tile_center[0],
                               self.tile_center[1]])

        self.arena_center = ARENA_CENTER

        # index of tiles that define the vertices of the arena
        self.vert_tile = np.array(VERT_TILE)

        # compute boundary information
        self._init_boundary()

    def _init_boundary(self):
        # boundary defined by hyperplane n @ v + b <= 0
        n_bounds = 6 # hexagon
        shift = 80
        self.A = np.zeros([2, n_bounds])
        self.b = np.zeros([n_bounds, 1])
        for idx in range(n_bounds):
            # vertices
            v1 = self.tiles[:, self.vert_tile[idx]]
            v2 = self.tiles[:, self.vert_tile[(idx + 1) % n_bounds]]

            # unit vector
            n = (v2 - v1) / np.linalg.norm(v2 - v1)
            # rotate n by 90 degrees counter-clockwise to get norm vector
            n = np.array([-n[1], n[0]])

            # n @ v = c, b = -c
            b = - np.dot(n, v1) - shift

            self.A[:, idx] = n
            self.b[idx] = b

        # compute boundary vertices using intersection
        self.vert_bound = np.zeros([2, n_bounds])
        for idx in range(n_bounds):
            n1 = self.A[:, idx]
            b1 = self.b[idx].squeeze()

            n2 = self.A[:, (idx + 1) % n_bounds]
            b2 = self.b[(idx + 1) % n_bounds].squeeze()

            # intersection
            x = np.linalg.solve(np.array([n1, n2]), -np.array([b1, b2]))
            self.vert_bound[:, idx] = x

    def check_boundary(self, pos):
        # check if the position is within the boundary
        return np.all(self.A.T @ pos + self.b <= 0)

    def _draw_hex(self, plot_ax, center, alpha=1,
                  label=False, index=None):
        plot_ax.add_patch(patches.RegularPolygon(
            center, numVertices=6,
            radius=TILE_RAD_MM,
            orientation=TILE_ANGLE,
            facecolor='w', edgecolor='g',
            lw=1, alpha=alpha))

        # write index on the tile
        if label:
            plot_ax.text(center[0], center[1], str(index),
                         fontsize=8, ha='center', va='center')

    def draw_arena(self, plot_ax, alpha=1, label=False):
        for idx in range(len(self.tile_center[0])):
            self._draw_hex(plot_ax, (self.tile_center[0][idx],
                                     self.tile_center[1][idx]),
                                     alpha, label, index=idx)

    def draw_boundary(self, plot_ax):
        # draw a hexagon boundary using the vertices in vert_bound
        n_bounds = 6 # hexagon
        for idx in range(n_bounds):
            v1 = self.vert_bound[:, idx]
            v2 = self.vert_bound[:, (idx + 1) % n_bounds]
            plot_ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r--')

    def draw_target(self, plot_ax, alpha=0.5, draw_hex=False):
        if draw_hex:
            for t in self.target.T:
                self._draw_hex(plot_ax, t)

        return [plot_ax.scatter(*target, s=256,
                                alpha=alpha,
                                facecolors='r',
                                edgecolors='none')
                for target in self.target.T]

    # Deprecated: Will switch to CCF at some point
    def get_center(self):
        return self.arena_center


class SessionData(ArenaMap):

    def __init__(self, name, ses, df, video_path, hs_path):
        super().__init__()

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
        self._smooth_trajectory()

        # cricket tiles
        self.zaber_target = self._target(df['locations'][0])
        self.target = self.zaber_target * ZABER_TO_MM

        # time
        self.time = df['relative_time'].to_numpy()
        self.frame = df['frame_no'].to_numpy().astype(int)
        self.frame_rate = self.frame[-1] / self.time[-1]

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

        # hs video and tracking
        # note pose data is not loaded by default
        # call _load_pose() to load tracking data
        self.hs_path = hs_path
        self.track_path = hs_path[:-4] + '.mat'
        self.calib_path = hs_path[:-4] + '_calib.csv'

        # cricket catch
        self.n_catch = np.sum(self.triggered)
        self.trigger_time = np.where(self.triggered == 1)[0]

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

    def _load_pose(self):
        if not os.path.exists(self.track_path):
            self.pose = None
            return

        # load tracking data
        tracking = scipy.io.loadmat(self.track_path)

        # (n_points, (x, y), n_frame)
        points = tracking['points']
        points = points.reshape([-1, points.shape[-1]])
        self.track_conf = tracking['conf'] - 1.0

        # sampling rate ~= 120 Hz
        # cutoff frequency = 30 Hz
        fs = 120
        nyquist = 0.5 * fs
        cutoff = 30

        b, a = butter(N=2, Wn=cutoff/nyquist,
                      btype='low', analog=False)

        self.pose = filtfilt(b, a, points, axis=-1)

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

    def _construct_trial(self, trial_idx, eos=False):
        start_idx, n_frame = self._trial_index(trial_idx, eos=eos)
        end_idx = start_idx + n_frame

        return TrialData(self, self.name, self.session, trial_idx, self.time[start_idx:end_idx],
                         self.chirped[start_idx:end_idx], self.x[start_idx:end_idx],
                         self.y[start_idx:end_idx], self.target, self.chirp_loc[start_idx:end_idx])

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

    # 30 sec ISI * average frame rate
    def _trial_index(self, trial_idx, prepend=None, append=0, eos=False):
        trigger_time = np.where(self.triggered == 1)[0]
        session_length = self.time.shape[0]

        if prepend is None:
            prepend = int(self.frame_rate * ISI)

        # if no cricket catch
        if len(trigger_time) == 0:
            return 0, session_length

        # with cricket catch
        if eos:
            start_idx = trigger_time[-1] + prepend
            n_frame = session_length - start_idx

        else:
            if trial_idx == 0:
                start_idx = 0
                n_frame = trigger_time[0] + append
            else:
                start_idx = trigger_time[trial_idx - 1] + prepend
                n_frame = trigger_time[trial_idx] - start_idx + append

            # check if exceed end of session
            if start_idx + n_frame > session_length:
                n_frame = session_length - start_idx

        return start_idx, n_frame

    # 30 sec ISI * average frame rate
    def _frame_index(self, trial_idx, eos=False):
        append = int(self.frame_rate * ISI)
        return self._trial_index(trial_idx, append=append, eos=eos)

    def all_video(self, max_frame=16500, eos=False):
        n_trigger = np.where(self.triggered == 1)[0].shape[0]

        if n_trigger == 0:
            print('No cricket capture')
            DataPlot().trial_video(self, 0)
        else:
            for idx in range(n_trigger):
                _, n_frame = self._frame_index(idx)
                if n_frame > max_frame:
                    print('Skip Session %d, Trial %d: %d frames' % (self.session, idx, n_frame))
                else:
                    DataPlot().trial_video(self, idx)

            # include end of session
            if eos:
                DataPlot().trial_video(self, n_trigger, eos=True)


class DataPlot():

    def __init__(self):
        self.target_fps = 10

    def _init_vars(self):
        self.chirp_count = 0
        self.chirp_active = False
        self.chirp_point = None
        self.step_count = 0
        self.captured = False
        self.tile_visited = np.zeros(len(self.targets)).astype(bool)

    def _check_tile_visit(self, stop_point):
        dist = np.linalg.norm(self.target_xy - stop_point.reshape([2, 1]), axis=0)
        tile_visit = (dist <= TRIG_RADIUS)
        if np.sum(tile_visit) > 0:
            tile_index = np.where(tile_visit)[0][0]
            self.tile_visited[tile_index] = True
            self.targets[tile_index].set_facecolor('g')

    def _init_plot(self, ses_obj, trial_obj):
        # create figure
        self.fig, self.axs = plt.subplots(1, 2, figsize=(20, 10))

        # LEFT PLOT
        # TODO: correct tiles coordinates using CCF
        # ses_obj.draw_arena(axs[0])
        ses_obj.draw_boundary(self.axs[0])
        self.target_xy = ses_obj.target
        self.targets = ses_obj.draw_target(self.axs[0], draw_hex=True)

        # trajectory and target
        self.ll, = self.axs[0].plot([], [], 'orange', alpha=0.25)

        # put a cross on the current trial target
        trial_target = trial_obj.trial_target
        self.axs[0].scatter(*trial_target, s=125, alpha=0.5,
                       marker='x', linewidths=3)

        # mouse location
        radius = 35
        self.circle = plt.Circle((0, 0), radius, color='tab:blue',
                                 linewidth=2, fill=False)
        self.axs[0].add_patch(self.circle)

        # write information
        self.text = self.axs[0].text(20, 20, 'Chirps: 0, Tile Visit: 0', fontsize=12)

        # RIGHT PLOT
        self.im = self.axs[1].imshow(ses_obj.get_frame(0), cmap='gray')
        self.axs[1].invert_xaxis()

        # add an indicator for chirp
        self.ind_right = self.axs[1].scatter(50, 50, s=625, marker='s',
                                             color='tab:blue', label='Chirp')
        self.ind_right.set_visible(False)

        # axis format
        self.axs[0].set_xlim(0, 2400)
        self.axs[0].set_ylim(0, 2400)
        self.axs[0].set_aspect('equal')

        self.axs[1].axis('off')
        self.axs[1].set_aspect('equal')

    def select_color(self, index):
        if index >= len(self.colors):
            return self.colors[-1]

        return self.colors[index]

    def animate(self, ses_obj, i):
        self.axs[0].set_title('Frame %d, Time %.3f Sec' % (i, ses_obj.time[i]))
        self.text.set_text('# Chirps: %d, # of Tile Visit: %d' %
                           (self.chirp_count, np.sum(self.tile_visited)))

        if not self.captured:
            # active -> inactive
            if self.chirp_active:
                self.step_count += 1
                if self.step_count == 10:
                    self.chirp_active = False
                    self.chirp_point.set_facecolor('r')
                    self.ind_right.set_visible(False)
                    self.step_count = 0

            # inactive -> active
            if ses_obj.chirped[i] == 1:
                self.chirp_count += 1
                self.chirp_active = True

                self.chirp_point = self.targets[ses_obj.chirp_loc[i]]
                self.chirp_point.set_facecolor('b')
                self.ind_right.set_visible(True)

                self._check_tile_visit(np.array([ses_obj.x[i], ses_obj.y[i]]))

                self.axs[0].scatter(ses_obj.x[i], ses_obj.y[i], marker='s',
                                    color=self.select_color(self.chirp_count - 1))

            if ses_obj.triggered[i] == 1:
                self.captured = True
                self.ind_right.set_facecolor('g')
                self.ind_right.set_visible(True)

        # plot trajectory data
        self.ll.set_data(ses_obj.x[self.start_idx:i], ses_obj.y[self.start_idx:i])
        self.circle.center = (ses_obj.x[i], ses_obj.y[i])

        # display video frame
        self.im.set_data(ses_obj.get_frame(i))

        # update progress bar
        self.pbar.update(1)

    def trial_video(self, ses_obj, trial_idx, eos=False):
        '''
        Generate a video for visualizing the trial
        '''
        # init data variables
        trial_obj = ses_obj._construct_trial(trial_idx, eos=eos)
        print('%s s%d, t%d' % (ses_obj.name, ses_obj.session, trial_idx))

        self.start_idx, self.n_frame = ses_obj._frame_index(trial_idx, eos=eos)
        _, self.n_frame_hunting = ses_obj._trial_index(trial_idx, eos=eos)

        n_chip = ses_obj.chirped[self.start_idx:self.start_idx + self.n_frame_hunting].sum()
        print('number of active chirps: %d' % n_chip)

        # create a continous color map
        self.colors = plt.cm.viridis(np.linspace(0, 1, n_chip + 1))

        # init plotting
        self._init_plot(ses_obj, trial_obj)
        self._init_vars()
        self.pbar = tqdm.tqdm(total=self.n_frame, position=0, leave=True)

        # create video
        self.save_video(ses_obj, trial_idx)

    def save_video(self, ses_obj, trial_idx):
        # create video
        # index shifted animation function
        def animate_shifted(i): return self.animate(ses_obj, i + self.start_idx)
        ani = animation.FuncAnimation(self.fig, animate_shifted,
                                      frames=self.n_frame - 1,
                                      interval=10)

        # video path
        file_path = os.path.join('./data/video', ses_obj.name)
        file_name = 's%d_t%d' % (ses_obj.session, trial_idx) + '.mp4'

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # save video
        writer = animation.writers['ffmpeg'](fps=self.target_fps, bitrate=4096)
        ani.save(os.path.join(file_path, file_name), writer=writer)
        self.pbar.close()


class TrialData(ArenaMap):

    def __init__(self, ses_ref, name, session, trial_idx, time,
                 chirp, x, y, target, chirp_loc, catch=True):
        super().__init__()

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
        self.trial_target = target[:, chirp_loc[-1]]

        self.catch = catch

        # number of chirp locations should be 2 or 3
        # (last, current, and next)
        if len(chirp_loc) < 2 or len(chirp_loc) > 3:
            warnings.warn("Chrip location is %d" % len(chirp_loc))

        # x, y data
        self.x = x
        self.y = y

        # chirp coordinates and time
        self.chirp_x = x[chirp == 1]
        self.chirp_y = y[chirp == 1]
        self.chirp_time = self.time[chirp == 1]

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
        super().__init__()

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
            tile_visit = (dist <= TRIG_RADIUS).astype(int)
            counter += (np.sum(tile_visit) > 0).astype(int)

        start_ind = np.any(np.linalg.norm(self.target - self.start, axis=0) <= TRIG_RADIUS)
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