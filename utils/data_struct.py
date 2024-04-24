import cv2
import numpy as np
from .data_loader import ZABER_TO_MM

class SessionData:

    def __init__(self, name, df, video_path):
        # session name
        self.name = name

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
