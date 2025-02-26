import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import IntEnum
from matplotlib import colormaps
cmap = colormaps.get_cmap('Spectral')

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
        plot_colors = [cmap(i / (len(cls) - 1))
                       for i in range(len(cls))]

        return plot_colors

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

    def compute_angle(self, anchor_points):
        '''
        Compute the orientation of an anchor point
        assuming the coordinates are centered
        '''
        anchor = self.average_point(anchor_points)
        angle = np.arctan2(anchor[1, :], anchor[0, :])

        return angle

    def rotate(self, anchor_points=[Mouse.TAIL_BASE], target=-np.pi/2):
        '''
        Rotate the pose data so anchor points
        are aligned with target angle (-y by default).
        '''
        return self.rotate_points(self.xy, anchor_points, target)

    def rotate_points(self, xy, anchor_points=[Mouse.TAIL_BASE], target=-np.pi/2):
        angle = self.compute_angle(anchor_points)
        theta = target - angle

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

    FR = 120
    SEC_TO_MS = 1000
    PRE = 0.75
    POST = 0.25

    def __init__(self, session):
        self.session = session
        self.session._load_pose()

        # chirp index and time in zaber frames
        chirp_index = np.where(session.chirped == 1)[0]
        self.chirp_time = session.time[chirp_index]
        self.n_chirps = len(chirp_index)

        # change to index in hs frames
        self.chirp_index = session.hs_index[chirp_index]
        self.index_start = self.chirp_index - int(self.PRE * self.FR)
        self.index_end = self.chirp_index + int(self.POST * self.FR)
        self.n_frames = self.index_end[0] - self.index_start[0]

        # key points data
        # (n_chirps, n_keypoints, n_frames)
        keypoints = session.keypoints
        self.kp = np.stack([keypoints[:, self.index_start[i]:self.index_end[i]]
                            for i in range(len(self.index_start))], axis=0)

    def to_gifs(self, index='linear', shift=0):
        # Select frames to plot
        n_image = 9

        if index == 'linear':
            index = np.arange(n_image) + shift
        elif index == 'random':
            index = np.random.choice(len(self.kp), n_image)
            index = np.sort(index)

        # Create a list of combined frames
        combined_frames = []

        for frame_idx, frame in range(self.n_frames):
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))

            # Plot each movie's frame
            for i, ax in enumerate(axes.flat):
                frame_index =
                ax.imshow(hs_video[i, frame], cmap='gray')

                # write out some information
                if i == 1:
                    ax.title.set_text('Time %.1f ms' % (frame / fr * sec_to_ms))
                if i == 4 and frame_idx >= int(pre*fr):
                    ax.scatter(975, 975, s=400, marker='s', color='tab:blue')

                ax.set_xlim(0, 1024)
                ax.set_ylim(0, 1024)
                ax.invert_yaxis()
                ax.axis("off")  # Hide axes

            # Save the current figure as an image in memory
            fig.tight_layout()
            fig.canvas.draw()
            img = Image.fromarray(np.array(fig.canvas.buffer_rgba()))
            combined_frames.append(img)

            plt.close(fig)  # Close to free memory

        # Save the combined frames as a single GIF
        gif_filename = "./pose_movies_%s_%d.gif" % (self.session.name,
                                                    self.session.session)

        combined_frames[0].save(gif_filename, save_all=True,
                                append_images=combined_frames[1:],
                                duration=10, loop=0)