import numpy as np
from enum import IntEnum

import utils.plottools as plottools
from utils.plottools import KP_COLORS

from utils.data_loader import TRIG_RADIUS

class Mouse(IntEnum):
    '''
    Define the keypoint indices for a mouse.
    '''
    L_NOSE = 0
    R_NOSE = 1
    L_FRONT = 2
    R_FRONT = 3

    R_EYE_FRONT = 4
    R_EYE_BACK = 5
    R_EYE_IN = 6
    R_EYE_OUT = 7

    L_EYE_FRONT = 8
    L_EYE_BACK = 9
    L_EYE_IN = 10
    L_EYE_OUT = 11

    L_EAR_UI = 12
    L_EAR_LI = 13
    L_EAR_LO = 14
    L_EAR_UO = 15
    L_EAR_TIP = 16
    L_EAR_BASE = 17

    R_EAR_UO = 18
    R_EAR_LO = 19
    R_EAR_LI = 20
    R_EAR_UI = 21
    R_EAR_TIP = 22
    R_EAR_BASE = 23

    HEAD = 24
    NECK = 25
    BODY = 26
    TAIL_BASE = 27

    TAIL_T = 28
    TAIL_M = 29
    TAIL_E = 30

    LU_BODY = 31
    LM_BODY = 32
    LL_BODY = 33

    RU_BODY = 34
    RM_BODY = 35
    RL_BODY = 36

    @classmethod
    def colors(cls):
        return KP_COLORS

    # Define point groups
    @classmethod
    def nose(cls):
        return [cls.L_NOSE, cls.R_NOSE]

    @classmethod
    def front(cls):
        return [cls.L_NOSE, cls.R_NOSE, cls.L_FRONT, cls.R_FRONT]

    @classmethod
    def right_eye(cls):
        return [cls.R_EYE_FRONT, cls.R_EYE_BACK, cls.R_EYE_IN, cls.R_EYE_OUT]

    @classmethod
    def left_eye(cls):
        return [cls.L_EYE_FRONT, cls.L_EYE_BACK, cls.L_EYE_IN, cls.L_EYE_OUT]

    @classmethod
    def left_ear(cls):
        return [cls.L_EAR_UI, cls.L_EAR_LI, cls.L_EAR_LO, cls.L_EAR_UO, cls.L_EAR_TIP, cls.L_EAR_BASE]

    @classmethod
    def right_ear(cls):
        return [cls.R_EAR_UO, cls.R_EAR_LO, cls.R_EAR_LI, cls.R_EAR_UI, cls.R_EAR_TIP, cls.R_EAR_BASE]

    @classmethod
    def tail(cls):
        return [cls.TAIL_T, cls.TAIL_M, cls.TAIL_E]

class AnimalPose():
    def __init__(self, pose):
        '''
        pose: np.array, shape=(k, n),
        where k is the number of keypoints,
        and n is the number of frames.
        '''
        # 2D array
        if len(pose.shape) == 2:
            self.pose = pose.copy()
            self.xy = self._to_xy(pose)

        # 3D array
        elif len(pose.shape) == 3:
            self.xy = pose.copy()
            self.pose = self._to_vector(pose)

        else:
            raise ValueError('Invalid shape of pose data')

    def _to_xy(self, pose):
        return pose.reshape([-1, 2, pose.shape[-1]])

    def _to_vector(self, xy):
        return xy.reshape([-1, xy.shape[-1]])

    def average_point(self, group):
        return np.mean(self.xy[group, :, :], axis=0)

    def center(self, anchor_points=[Mouse.HEAD]):
        '''
        Center the pose data around the average of an anchor,
        defined by center_points (default Mouse.HEAD).
        '''
        return self.center_points(self.xy, anchor_points)

    def center_points(self, xy, anchor_points=[Mouse.HEAD]):
        mouse_center = self.average_point(anchor_points)
        return AnimalPose(xy - mouse_center[np.newaxis, :, :])

    def rotate_angle(self, rot_angle):
        '''
        Rotate the pose data by a fixed amount.
        '''
        return self.rotate(rot_angle=rot_angle)

    def rotate(self, anchor_points=[Mouse.TAIL_BASE], rot_angle=None, target=-np.pi/2):
        '''
        Rotate the pose data so a set of anchor points
        are aligned with target angle (-y by default).
        '''
        return self.rotate_points(self.xy, anchor_points, rot_angle, target)

    def compute_angle(self, anchor_points):
        '''
        Compute the orientation of a set of anchor points
        assuming the coordinates are centered
        '''
        anchor = self.average_point(anchor_points)
        angle = np.arctan2(anchor[1, :], anchor[0, :])

        return angle

    def rotate_points(self, xy, anchor_points=[Mouse.TAIL_BASE], rot_angle=None, target=-np.pi/2):
        if rot_angle is None:
            angle = self.compute_angle(anchor_points)
            theta = target - angle
        else:
            theta = rot_angle * np.ones(self.xy.shape[-1])

        # batch rotation matrix
        # Create batched rotation matrices (2, 2, n_data)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        rotations = np.array([
            [cos_t, -sin_t],
            [sin_t, cos_t]])

        # apply rotation
        rotated = np.einsum('idk, mdk -> mik', rotations, xy)
        return AnimalPose(rotated)


class StopPose():
    '''
    Class for analyzing pose data around stop (chirp) events.
    '''

    # Frame rate
    FR = 120
    SEC_TO_MS = 1000

    def __init__(self, session, pre=0.60, post=0.10,
                 center=True, rotate=True, exclude=None):
        self.session = session
        self.session._load_pose()

        # define time window around chirp
        self.pre = pre
        self.post = post

        # cricket release time
        trigger_index = session.trigger_index
        self.trigger_time = session.time[trigger_index]

        # chirp index, time in zaber frames, and location
        chirp_index = np.where(session.chirped == 1)[0]

        # change to index in hs frames
        self.chirp_index = session.hs_index[chirp_index]
        self.index_start = self.chirp_index - int(self.pre * self.FR)
        self.index_end = self.chirp_index + int(self.post * self.FR)
        self.n_frames = self.index_end[0] - self.index_start[0]
        self.pose_category = None # pose category for each chirp stop
        self._frame_check()

        # chirp time and location
        chirp_index = chirp_index[self.frame_bound]
        self.chirp_time = session.time[chirp_index]
        self.n_chirps = len(chirp_index)

        x = session.x[chirp_index]
        y = session.y[chirp_index]
        self.chirp_loc = np.stack([x, y], axis=0)
        self._tile_check()

        # key points data
        # (n_chirps, n_keypoints, n_frames)
        self.center = center
        self.rotate = rotate
        self.exclude = exclude
        self.process_keypoints(session)

    def _circ_mean(self, angles):
        '''
        Compute the circular mean of a set of angles.
        '''
        return np.arctan2(np.mean(np.sin(angles)),
                          np.mean(np.cos(angles)))

    def _frame_check(self):
        '''
        Check if the chirp location is within the frame boundaries.
        '''
        self.frame_bound = np.logical_and(
            self.index_start >= 0,
            self.index_end <= self.session.keypoints.shape[1])

        self.index_start = self.index_start[self.frame_bound]
        self.index_end = self.index_end[self.frame_bound]

    def _tile_check(self, threshold_mul=1.2):
        '''
        Mark if the chirp location is close to a cricket tile,
        using a more lenient threshold than the trigger radius.
        '''
        target = self.session.target
        tile_check = np.zeros(self.chirp_loc.shape[-1]).astype(bool)
        for idx in range(target.shape[-1]):
            t = target[:, idx].reshape(2, -1)
            dist = np.linalg.norm(self.chirp_loc - t, axis=0)
            tile_check = np.logical_or(tile_check, dist <= threshold_mul * TRIG_RADIUS)

        self.tile_check = tile_check

    def set_category(self, category):
        self.pose_category = category

    def process_keypoints(self, session):
        '''
        Process keypoint data to AnimalPose objects.
        '''
        all_kp = []
        kp_index = []

        for i in range(self.n_chirps):
            # key points segment, skip if out of bounds
            if (self.index_start[i] < 0 or
                self.index_end[i] > session.keypoints.shape[1]):
                continue

            kp = session.keypoints[:, self.index_start[i]:self.index_end[i]]
            kp_index.append(np.arange(self.index_start[i], self.index_end[i]))

            if self.center and self.rotate:
                # center and rotate pose data for each segment
                kp = AnimalPose(kp).center()

                # rotate the pose data so the body is aligned with the -y direction
                angles = kp.compute_angle([Mouse.TAIL_BASE])
                angle = self._circ_mean(angles)
                target = -np.pi/2
                kp = kp.rotate(rot_angle=(target - angle))

            elif self.center:
                kp = AnimalPose(kp).center()

            # exclude keypoints
            if self.exclude is None:
                kp = kp.pose

            else:
                # array of keypoints
                indicator = np.ones(kp.xy.shape[0], dtype=bool)
                indicator[self.exclude] = False

                xy_exclude = kp.xy[indicator, :, :]
                kp = xy_exclude.reshape(-1, kp.xy.shape[-1])

            # record each segment
            all_kp.append(kp)

        self.kp = np.stack(all_kp, axis=0)
        self.kp_index = np.stack(kp_index, axis=0)

    def _generate_index(self, index='linear', shift=0):
        n_image = 9

        if index == 'linear':
            index = np.arange(n_image) + shift
        elif index == 'random':
            index = np.random.choice(len(self.kp), n_image)
            index = np.sort(index)

        return index

    def get_movie(self, stop_index):
        start = self.index_start[stop_index]
        end = self.index_end[stop_index]

        frames = []
        for i in range(start, end):
            frames.append(self.session.hs_frame(i, native=True).mean(axis=-1))

        return np.array(frames)

    def movie_to_gifs(self, index='linear', shift=0):
        # Select frames to plot
        if not isinstance(index, np.ndarray):
            index = self._generate_index(index, shift)

        # Collect movie frames
        frame_size = self.session.hs_frame(0, native=True).shape
        all_movies = np.zeros((len(index), self.n_frames, *frame_size[:-1])).astype(np.uint8)
        for i in range(len(index)):
            all_movies[i, :, :, :] = self.get_movie(index[i])

        # Save the combined frames as a single GIF
        gif_filename = "./pose_movies_%s_%d.gif" % (self.session.name,
                                                    self.session.session)

        plottools.movie_to_gifs(all_movies, self.FR, self.pre, gif_filename)

    def pose_to_gifs(self, index='linear', shift=0):
        # Select frames to plot
        if not isinstance(index, np.ndarray):
            index = self._generate_index(index, shift)
        pose_frames = self.kp[index, :, :]

        gif_filename = "./keypoint_movies_%s_%d.gif" % (self.session.name,
                                                    self.session.session)

        plottools.pose_to_gifs(pose_frames, self.FR, self.pre,
                               self.center, self.rotate, gif_filename)