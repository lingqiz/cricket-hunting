import cv2
import pandas as pd
import numpy as np
import tqdm
import os
import warnings
import scipy.io
from scipy.signal import butter, filtfilt
from matplotlib import colormaps

from .constants import ZABER_TO_MM, DLC_TO_MM, ISI, TRK_CTR, TILE_CENTER, \
    TILE_RAD_MM, TILE_ANGLE, ARENA_CENTER, VERT_TILE, TRIG_RADIUS, NAME_DICT

from .plottools import KP_COLORS

import matplotlib
import matplotlib.pyplot as plt
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
        return np.all(self.A.T @ pos + self.b <= 0, axis=0)

    def _draw_hex(self, plot_ax, center, alpha=1,
                  label=False, index=None):
        plot_ax.add_patch(patches.RegularPolygon(
            center, numVertices=6,
            radius=TILE_RAD_MM,
            orientation=TILE_ANGLE,
            facecolor='none', edgecolor='g',
            lw=2, alpha=alpha))

        # write index on the tile
        if label:
            plot_ax.text(center[0], center[1], NAME_DICT[index],
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

    def get_center(self):
        return self.arena_center


class SessionData(ArenaMap):

    def __init__(self, animal_name, session_type, session_index, session_dir):
        super().__init__()

        # name and session
        self.name = animal_name
        self.session_type = session_type
        self.session = session_index
        self.session_dir = session_dir

        # resolve file paths from session directory
        self._resolve_paths()

        # load behavior CSV
        df = pd.read_csv(self.csv_path, low_memory=False)

        # x-y corrdinates, use mm units
        self.zaber_x = df['zaber_x'].to_numpy() * ZABER_TO_MM
        self.zaber_y = df['zaber_y'].to_numpy() * ZABER_TO_MM

        self.dlc_x = (df['dlc_x'].to_numpy() - TRK_CTR) * DLC_TO_MM
        self.dlc_y = (df['dlc_y'].to_numpy() - TRK_CTR) * DLC_TO_MM

        # tracking coordinates
        self.x = self.zaber_x - self.dlc_x
        self.y = self.zaber_y + self.dlc_y
        self._smooth_trajectory()

        # cricket tiles (already CCF-aligned)
        zaber_target = self._target(df['locations'][0])
        self.target = zaber_target * ZABER_TO_MM

        # time
        self.time = df['relative_time'].to_numpy()
        self.frame = df['frame_no'].to_numpy().astype(int)
        self.frame_rate = self.frame[-1] / self.time[-1]

        # chirp
        self.chirped = self._int_array(df['chirped'].to_numpy())
        self.chirp_loc = self._int_array(df['chirp_loc'].to_numpy()) - 1
        self.chirp_bout = self._int_array(df['chirp_bouts'].to_numpy())
        self.triggered = self._int_array(df['triggered'].to_numpy())

        # cricket catch
        self.n_catch = np.sum(self.triggered)
        self.trigger_index = np.where(self.triggered == 1)[0]

        # video
        self.video = cv2.VideoCapture(self.video_path)

        # hs video and tracking
        # note pose data is not loaded by default
        # call _load_pose() to load tracking data
        if self.hs_path and os.path.exists(self.hs_path):
            self.has_hs = True
            self.hs = cv2.VideoCapture(self.hs_path)
            self.hs_length = int(self.hs.get(cv2.CAP_PROP_FRAME_COUNT))
            self.hs.set(cv2.CAP_PROP_POS_FRAMES, 1)
            _, frame = self.hs.read()
            self.hs_shape = frame.shape
            self._load_calib()
        else:
            self.has_hs = False
            self.hs = None
            self.hs_length = 0
            self.hs_shape = None
            self.hs_index = None

    def _find_file(self, suffix):
        '''Find a file in the session directory by suffix.
        Returns the full path, or None if not found.
        '''
        for f in os.listdir(self.session_dir):
            if f.endswith(suffix):
                return os.path.join(self.session_dir, f)
        return None

    def _resolve_paths(self):
        '''Find key files in the session directory.'''
        self.csv_path = self._find_file('_ccf_all_params_file.csv')
        self.video_path = self._find_file('_rig.avi')
        self.hs_path = self._find_file('_hs.mp4')
        self.calib_path = self._find_file('_calib.csv')
        self.track_path = self._find_file('_tracking.mat')

    def _smooth_trajectory(self):
        # sampling rate = 15 Hz
        # cutoff frequency = 5 Hz
        fs = 15
        nyquist = 0.5 * fs
        cutoff = 4
        b, a = butter(N=2, Wn=cutoff/nyquist,
                      btype='low', analog=False)

        self.x = filtfilt(b, a, self.x)
        self.y = filtfilt(b, a, self.y)

    def _load_calib(self):
        '''
        Find the corresponding frame index in the
        high-speed camera for each zaber point
        '''
        # frame rate of hs camera
        frame_rate = 120

        if os.path.exists(self.calib_path):
            self.has_hs = True
            calib_array = pd.read_csv(self.calib_path)
            calib_axis = calib_array[['video_index',
                                      'zaber_index']].to_numpy().T
            zaber_axis = calib_axis[1]
            video_axis = calib_axis[0]

            self.hs_index = np.zeros(self.time.size).astype(int)
            for idx in range(self.time.size):
                # find the closest index in zaber_axis
                zaber_idx = np.argmin(np.abs(zaber_axis - idx))
                zaber_frame = zaber_axis[zaber_idx]
                video_frame = video_axis[zaber_idx]

                delta = (self.time[idx] - self.time[zaber_frame]) * frame_rate
                self.hs_index[idx] = int(video_frame + delta)

        else:
            self.hs_index = None
            self.has_hs = False

    def _load_pose(self):
        if not self.track_path or not os.path.exists(self.track_path):
            self.pose = None
            return

        # load tracking data
        tracking = scipy.io.loadmat(self.track_path)

        # (n_points, (x, y), n_frame)
        points = tracking['points']
        points = points.reshape([-1, points.shape[-1]])
        self.track_conf = tracking['conf'] - 1.0

        # sampling rate ~= 120 Hz
        # cutoff frequency = 15 Hz
        fs = 120
        nyquist = 0.5 * fs
        cutoff = 15

        b, a = butter(N=2, Wn=cutoff/nyquist,
                      btype='low', analog=False)

        self.keypoints = filtfilt(b, a, points, axis=-1)

        # load behavior scores (e.g. scores_Rearing.mat, scores_Grooming.mat)
        self._load_scores()

        # check frame count consistency
        n_track = self.keypoints.shape[-1]
        if n_track != self.hs_length:
            warnings.warn(f'Frame count mismatch: hs_video={self.hs_length}, '
                          f'tracking={n_track}')
        for name, data in self.scores.items():
            n_scores = data['score'].shape[0]
            if n_scores != self.hs_length:
                warnings.warn(f'Frame count mismatch: hs_video={self.hs_length}, '
                              f'{name} scores={n_scores}')

    def _load_scores(self):
        '''Load behavior classification scores from scores_*.mat files.
        Populates self.scores as a dict keyed by behavior name,
        each containing "scores" and "postprocessed" arrays.
        '''
        self.scores = {}
        files = os.listdir(self.session_dir)
        for f in files:
            if f.startswith('scores_') and f.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(self.session_dir, f))
                name = str(mat['behaviorName'][0]).lower()
                allScores = mat['allScores'][0, 0]
                scores = allScores['scores'][0, 0].flatten()
                scoreNorm = allScores['scoreNorm'][0, 0].flatten()
                self.scores[name] = {
                    'score': np.clip(scores / scoreNorm, -1, 1),
                    'label': allScores['postprocessed'][0, 0].flatten(),
                }

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

    def get_eos(self):
        # get data for the end of a session (last catch to end)
        return self._construct_trial(self.n_catch, eos=True)

    def _construct_trial(self, trial_idx, eos=False):
        start_idx, n_frame = self._trial_index(trial_idx, append=1, eos=eos)
        end_idx = start_idx + n_frame

        return TrialData(self, self.name, self.session, trial_idx, self.time[start_idx:end_idx],
                         self.chirped[start_idx:end_idx], self.x[start_idx:end_idx],
                         self.y[start_idx:end_idx], self.target, self.chirp_loc[start_idx:end_idx],
                         catch=not eos)

    def _target(self, loc):
        loc = loc[1:-1].split(',')
        loc = np.array([float(l) for l in loc])
        loc_x = loc[0::2]
        loc_y = loc[1::2]

        return np.array([loc_x, loc_y])

    def _int_array(self, arr):
        arr[np.isnan(arr)] = 0
        return arr.astype(int)

    def get_frame(self, index, rgb=True):
        # get frame from video
        self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        _, frame = self.video.read()

        if rgb:
            return cv2.resize(frame, (1024, 1024))

        return cv2.resize(frame, (1024, 1024))[:, :, 0]

    def hs_frame(self, index, rgb=True, native=False):
        if not self.has_hs:
            return None

        if native:
            hs_index = index
        else:
            hs_index = self.hs_index[index]

        if hs_index < 0 or hs_index >= self.hs_length:
            frame = np.zeros(self.hs_shape).astype(np.uint8)

        else:
            self.hs.set(cv2.CAP_PROP_POS_FRAMES, hs_index)
            _, frame = self.hs.read()

        if rgb:
            return frame

        return frame[:, :, 0]

    # 30 sec ISI * average frame rate
    def _trial_index(self, trial_idx, prepend=None, append=0, eos=False):
        trigger_index = np.where(self.triggered == 1)[0]
        session_length = self.time.shape[0]

        if prepend is None:
            prepend = int(self.frame_rate * ISI)

        # if no cricket catch
        if len(trigger_index) == 0:
            return 0, session_length

        # with cricket catch
        if eos:
            start_idx = trigger_index[-1] + prepend
            n_frame = session_length - start_idx

        else:
            if trial_idx == 0:
                start_idx = 0
                n_frame = trigger_index[0] + append
            else:
                start_idx = np.min(np.where(self.chirp_loc == trial_idx)[0])
                n_frame = trigger_index[trial_idx] - start_idx + append

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

    # data reduction into stop (chirps) data
    def stop_data(self):
        self.stops = StopData(np.array([self.chirp_x, self.chirp_y]),
                              np.array([self.x[0], self.y[0]]),
                              np.array([self.x[-1], self.y[-1]]),
                              self.time[self.chirp == 1],
                              self.target, self.trial_idx)

        return self.stops


# class for different data reductions of the trial
class StopData(ArenaMap):
    def __init__(self, loc, start, end, t, target, trial_index):
        super().__init__()

        # stop location and time
        self.loc = loc
        self.t = t
        self.start = start.reshape(-1, 1)
        self.end = end.reshape(-1, 1)

        # target
        self.target = target
        self.target_visit = np.zeros(target.shape[1])
        self.end_target = trial_index
        self.end_loc = target[:, trial_index]
        self._target_visit()

        # calculate stop bout based on
        # distance threshold (10 mm)
        self._stop_bout()
        self.n_stop = self.loc.shape[1]
        self.n_bout = self.bout_loc.shape[1]

    def _target_visit(self):
        '''
        Compute target visit for the current trial
        '''
        locations = np.concatenate([self.start, self.loc, self.end], axis=1)

        for idx in range(self.target.shape[1]):
            dist = np.linalg.norm(locations - self.target[:, idx].reshape(-1, 1), axis=0)
            self.target_visit[idx] = np.any(dist <= TRIG_RADIUS)

        # if start location is inside a target
        start_ind = np.linalg.norm(self.target - self.start, axis=0) <= TRIG_RADIUS
        if np.any(start_ind):
            self.start_target = np.where(start_ind)[0][0]
        else:
            self.start_target = None

    def target_unique_visit(self):
        '''
        Count # of target visited, excluding the start location
        '''
        visit_ind = np.copy(self.target_visit)
        if self.start_target is not None:
            visit_ind[self.start_target] = 0

        return np.sum(visit_ind).astype(int)

    def _stop_bout(self, threshold=TRIG_RADIUS*2):
        '''
        Compute chirp bout based on the distance threshold
        TRIG_RADIUS * 2 ~= roughly the size of a tile
        '''
        # if no stop in trial
        if self.t.size == 0:
            self.bout_loc = np.empty((2, 0))
            self.bout_start = np.empty((0))
            self.bout_end = np.empty((0))
            return

        # compute chirp bout
        bout_loc = [self.loc[:, 0]]
        bout_start = [self.t[0]]
        bout_end = [self.t[0]]

        # calculate bout based on distance threshold
        bout_index = 0
        for idx in range(1, self.loc.shape[1]):
            if np.linalg.norm(self.loc[:, idx] - bout_loc[bout_index]) <= threshold:
                bout_end[bout_index] = self.t[idx]
            else:
                bout_index += 1
                bout_loc.append(self.loc[:, idx])
                bout_start.append(self.t[idx])
                bout_end.append(self.t[idx])

        # check if last bout not within target threshold
        if np.linalg.norm(bout_loc[-1] - self.end_loc) > TRIG_RADIUS:
            # add last bout (cricket catch)
            bout_index += 1
            bout_loc.append(self.loc[:, -1])
            bout_start.append(self.t[-1])
            bout_end.append(self.t[-1])

        # convert to numpy array
        self.bout_loc = np.array(bout_loc).T
        self.bout_start = np.array(bout_start)
        self.bout_end = np.array(bout_end)

        # if bout is a cricket tile check
        self.bout_check = np.zeros(self.bout_loc.shape[1]) - 1
        for idx in range(self.bout_loc.shape[1]):
            dist = np.linalg.norm(self.target - self.bout_loc[:, idx].reshape(-1, 1), axis=0)
            if np.any(dist <= TRIG_RADIUS):
                check_index = np.where(dist <= TRIG_RADIUS)[0][0]
                if check_index != self.start_target:
                    self.bout_check[idx] = check_index

    def delta_distance(self, bout=False):
        '''
        Distance of movement between stops
        '''
        if bout:
            delta = self.bout_loc[:, 1:] - self.bout_loc[:, :-1]
        else:
            delta = self.loc[:, 1:] - self.loc[:, :-1]

        return np.linalg.norm(delta, axis=0)


# class for generating video visualization of the data
class DataPlot():

    def __init__(self):
        # target frame rate
        self.target_fps = 10

        # color map for keypoints
        self.kp_colors = KP_COLORS.copy()

    def _init_vars(self):
        self.chirp_count = 0
        self.chirp_active = False
        self.chirp_point = None
        self.step_count = 0
        self.captured = False
        self.tile_visited = np.zeros(len(self.targets)).astype(bool)

    def _init_plot(self, ses_obj, trial_obj):
        '''
        Initialize the plot
        '''
        # create figure
        self.fig = plt.figure(figsize=(32, 12))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        self.axs = [self.fig.add_subplot(gs[i]) for i in range(3)]

        # LEFT PLOT
        ses_obj.draw_boundary(self.axs[0])
        self.target_xy = ses_obj.target
        self.targets = ses_obj.draw_target(self.axs[0], alpha=0.0, draw_hex=True)

        # trajectory and target
        self.ll, = self.axs[0].plot([], [], 'orange', alpha=0.50)

        # mark the current trial target
        self.targets[trial_obj.trial_idx].set_alpha(0.75)
        self.targets[trial_obj.trial_idx].set_facecolor('red')

        # mouse location
        radius = 35
        self.circle = plt.Circle((0, 0), radius, color='tab:blue',
                                 linewidth=2, fill=False)
        self.axs[0].add_patch(self.circle)

        # write information
        self.text = self.axs[0].text(20, -100, 'Chirps: 0, Tile Visit: 0', fontsize=12)

        # LOW RES VIDEO
        ls_init = ses_obj.get_frame(0, rgb=False)
        self.im = self.axs[1].imshow(ls_init, cmap='gray')
        self.axs[1].set_xlim(0, 1024)
        self.axs[1].set_ylim(0, 1024)
        self.axs[1].invert_xaxis()

        # add an indicator for chirp
        self.ind_right = self.axs[1].scatter(50, 50, s=625, marker='s', color='g', label='Chirp')
        self.ind_right.set_visible(False)

        # HIGH RES VIDEO
        hs_init = ses_obj.hs_frame(0, rgb=True) if ses_obj.has_hs \
            else np.zeros(ls_init.shape).astype(np.uint8)

        self.im_hs = self.axs[2].imshow(hs_init, cmap='gray')
        self.axs[2].set_xlim(0, 1024)
        self.axs[2].set_ylim(0, 1024)
        self.axs[2].invert_yaxis()

        self.ind_hs = self.axs[2].scatter(974, 974, s=625, marker='s', color='g', label='Chirp')
        self.ind_hs.set_visible(False)

        # draw keypoints
        if ses_obj.has_hs:
            points = ses_obj.keypoints[:, 0].reshape(-1, 2)
            conf = ses_obj.track_conf[:, 0] * 0.80 + 0.20
            self.keypoints = self.axs[2].scatter(points[:, 0], points[:, 1],
                                                c=self.kp_colors, alpha=conf,
                                                marker='+')

        # axis format
        self.axs[0].set_xlim(-50, 2350)
        self.axs[0].set_ylim(-150, 2250)
        self.axs[0].set_aspect('equal')
        self.axs[0].set_xlabel('X (mm)')
        self.axs[0].set_ylabel('Y (mm)')
        self.axs[0].spines[['right', 'top']].set_visible(False)

        self.axs[1].axis('off')
        self.axs[1].set_aspect('equal')

        self.axs[2].axis('off')
        self.axs[2].set_aspect('equal')

        plt.tight_layout()

    def _check_tile_visit(self, stop_point):
        dist = np.linalg.norm(self.target_xy - stop_point.reshape([2, 1]), axis=0)
        tile_visit = (dist <= TRIG_RADIUS)
        if np.sum(tile_visit) > 0:
            tile_index = np.where(tile_visit)[0][0]
            self.tile_visited[tile_index] = True
            self.targets[tile_index].set_alpha(0.75)
            self.targets[tile_index].set_facecolor('orange')

    def select_color(self, index):
        if index >= len(self.colors):
            return self.colors[-1]

        return self.colors[index]

    def animate(self, ses_obj, i):
        '''
        Per frame update function
        '''
        self.axs[0].set_title('Frame %d, Time %.3f Sec' % (i, ses_obj.time[i]))
        self.text.set_text('# Chirps: %d, # of Tile Visit: %d' %
                        (self.chirp_count, np.sum(self.tile_visited)))

        if not self.captured:
            if self.chirp_active:
                self.step_count += 1
                if self.step_count == 10:
                    self.chirp_active = False
                    self.chirp_point.set_facecolor('r')
                    self.ind_right.set_visible(False)
                    self.ind_hs.set_visible(False)
                    self.step_count = 0

            if ses_obj.chirped[i] == 1:
                self.chirp_count += 1
                self.chirp_active = True

                self.chirp_point = self.targets[ses_obj.chirp_loc[i]]
                self.chirp_point.set_facecolor('b')
                self.ind_right.set_visible(True)
                self.ind_hs.set_visible(True)

                self._check_tile_visit(np.array([ses_obj.x[i], ses_obj.y[i]]))

                self.axs[0].scatter(ses_obj.x[i], ses_obj.y[i], s=64, marker='o',
                                    color=self.select_color(self.chirp_count - 1))

            if ses_obj.triggered[i] == 1:
                self.captured = True
                self.ind_right.set_color('g')
                self.ind_right.set_visible(True)
                self.ind_hs.set_color('g')
                self.ind_hs.set_visible(True)

        # update trajectory
        self.ll.set_data(ses_obj.x[self.start_idx:i], ses_obj.y[self.start_idx:i])
        self.circle.center = (ses_obj.x[i], ses_obj.y[i])

        # update video frames
        self.im.set_data(ses_obj.get_frame(i, rgb=False))
        if ses_obj.has_hs:
            self.im_hs.set_data(ses_obj.hs_frame(i, rgb=True))

        # update keypoints
        if ses_obj.has_hs:
            kp_index = ses_obj.hs_index[i]
            if kp_index >= 0 and kp_index < ses_obj.hs_length:
                points = ses_obj.keypoints[:, kp_index].reshape(-1, 2)
                conf = ses_obj.track_conf[:, kp_index] * 0.80 + 0.20
                self.keypoints.set_offsets(points)
                self.keypoints.set_alpha(conf)

    def render_frame(self):
        '''
        Matplotlib canvas frame to RGB
        '''
        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return frame

    def render_video(self, ses_obj, trial_idx):
        '''
        Render video using OpenCV
        '''
        file_path = os.path.join('./data/Analysis', ses_obj.name, f'session_{ses_obj.session}')
        file_name = 's%d_t%d' % (ses_obj.session, trial_idx) + '.mp4'

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        width, height = self.fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(os.path.join(file_path, file_name),
                                       fourcc, self.target_fps, (width, height))

        # index shifted animation function
        for i in tqdm.tqdm(range(self.n_frame), desc='Rendering Video'):
            self.animate(ses_obj, i + self.start_idx)
            frame = self.render_frame()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()

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
        self.colors = plt.cm.viridis(np.linspace(0, 1,
                                    n_chip * 2 + 1))[n_chip:]

        # init plotting
        self._init_plot(ses_obj, trial_obj)
        self._init_vars()

        # create video
        self.render_video(ses_obj, trial_idx)