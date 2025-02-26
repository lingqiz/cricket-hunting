import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import IntEnum
from utils.plottools import KP_COLORS

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

    # Pre and post chirp time to include
    PRE = 0.60
    POST = 0.10

    def __init__(self, session, center=True, rotate=True):
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
        self.center = center
        self.rotate = rotate
        self.process_keypoints(session)

    def _circ_mean(self, angles):
        '''
        Compute the circular mean of a set of angles.
        '''
        return np.arctan2(np.mean(np.sin(angles)),
                          np.mean(np.cos(angles)))

    def process_keypoints(self, session):
        '''
        Process keypoint data to AnimalPose objects.
        '''
        all_kp = []

        for i in range(self.n_chirps):
            # key points segment, skip if out of bounds
            if (self.index_start[i] < 0 or
                self.index_end[i] > session.keypoints.shape[1]):
                continue

            kp = session.keypoints[:, self.index_start[i]:self.index_end[i]]

            if self.center and self.rotate:
                # center and rotate pose data for each segment
                kp = AnimalPose(kp).center()

                # rotate the pose data so the body is aligned with the -y direction
                angles = kp.compute_angle([Mouse.TAIL_BASE])
                angle = self._circ_mean(angles)
                target = -np.pi/2
                kp = kp.rotate(rot_angle=(target - angle)).pose

            elif self.center:
                kp = AnimalPose(kp).center().pose

            # record each segment
            all_kp.append(kp)

        self.kp = np.stack(all_kp, axis=0)

    def _generate_index(self, index='linear', shift=0):
        n_image = 9

        if index == 'linear':
            index = np.arange(n_image) + shift
        elif index == 'random':
            index = np.random.choice(len(self.kp), n_image)
            index = np.sort(index)

        return index

    def movie_to_gifs(self, index='linear', shift=0):
        # Select frames to plot
        if not isinstance(index, np.ndarray):
            index = self._generate_index(index, shift)

        # Create a list of combined frames
        combined_frames = []

        for frame_idx in range(self.n_frames):
            fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=150)

            # Plot each movie's frame
            for i, ax in enumerate(axes.flat):
                total_index = self.index_start[index[i]] + frame_idx
                frame = self.session.hs_frame(total_index, native=True)

                ax.imshow(frame, cmap='gray')

                # write out some information
                if i == 1:
                    ax.title.set_text('Time %.1f ms' % (frame_idx / self.FR * self.SEC_TO_MS))
                if frame_idx >= int(self.FR * self.PRE):
                    ax.scatter(975, 975, s=400, marker='s', color='tab:blue')

                ax.set_xlim(0, 1024)
                ax.set_ylim(0, 1024)
                ax.invert_yaxis()
                ax.axis("off")  # Hide axes

            # Save the current figure as an image in memory
            fig.tight_layout()
            fig.canvas.draw()
            frame_img = Image.fromarray(np.array(fig.canvas.buffer_rgba()))
            combined_frames.append(frame_img)

            plt.close(fig)  # Close to free memory

        # Save the combined frames as a single GIF
        gif_filename = "./pose_movies_%s_%d.gif" % (self.session.name,
                                                    self.session.session)

        combined_frames[0].save(gif_filename, save_all=True,
                                append_images=combined_frames[1:],
                                duration=10, loop=0)

    def pose_to_gifs(self, index='linear', shift=0):
        # Select frames to plot
        if not isinstance(index, np.ndarray):
            index = self._generate_index(index, shift)

        # Create a list of combined frames
        combined_frames = []

        for frame_idx in range(self.n_frames):
            fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=150)

            # Plot each movie's frame
            for i, ax in enumerate(axes.flat):
                pose_frame = self.kp[index[i], :, frame_idx].reshape(-1, 2)
                ax.scatter(pose_frame[:, 0], pose_frame[:, 1],
                           c=Mouse.colors(), alpha=0.90, marker='+')

                # write out some information
                if i == 1:
                    ax.title.set_text('Time %.1f ms' % (frame_idx / self.FR * self.SEC_TO_MS))
                if frame_idx >= int(self.FR * self.PRE):
                    ax.scatter(975, 975, s=400, marker='s', color='tab:blue')

                if self.center:
                    ax.set_xlim(-512, 512)
                    ax.set_ylim(-512, 512)
                else:
                    ax.set_xlim(0, 1024)
                    ax.set_ylim(0, 1024)

                if not self.rotate:
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

        # Save the combined frames as a single GIF
        gif_filename = "./keypoint_movies_%s_%d.gif" % (self.session.name,
                                                    self.session.session)

        combined_frames[0].save(gif_filename, save_all=True,
                                append_images=combined_frames[1:],
                                duration=10, loop=0)