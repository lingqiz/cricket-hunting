import numpy as np
from enum import IntEnum

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

class AnimalPose():
    def __init__(self, pose):
        '''
        pose: np.array, shape=(k, n),
        where k is the number of keypoints,
        and n is the number of frames.
        '''
        self.pose = pose.copy()
        self.xy = pose.reshape([-1, 2, pose.shape[-1]])

    def average_point(self, group):
        return np.mean(self.xy[group, :, :], axis=0)

