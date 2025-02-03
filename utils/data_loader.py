import numpy as np
import pandas as pd
import math, os
from datetime import datetime

import pathlib
DIR_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DIR_HOME = pathlib.Path.home()

# unit conversions
ZEBER_TO_DLC = 1896 / 72248
ZABER_TO_MM = 508 / 72248
DLC_TO_MM = ZABER_TO_MM / ZEBER_TO_DLC
TRK_CTR = 948
TRIG_RADIUS_ZABER = 5500
TRIG_RADIUS = TRIG_RADIUS_ZABER * ZABER_TO_MM

# 30 sec ISI
ISI = 30

def tile_angle():
    '''
    Angle of the hex tiles as defined in:
    https://github.com/the-dennis-lab/ejd_behavior_sandbox/blob/main/notebooks/arena_mapping.ipynb
    '''
    # two reference points

    p1 = (3736, 166646)
    p2 = (300372, 187681)

    # Difference in x coordinates
    dx = p2[0] - p1[0]

    # Difference in y coordinates
    dy = p2[1] - p1[1]

    # Angle between p1 and p2 in radians
    theta = math.atan2(dy, dx)
    rot_in_rad = math.radians(90) + theta

    return rot_in_rad

TILE_ANGLE = tile_angle()
TILE_RAD_ZABER = 19440 / 2
TILE_RAD_MM = TILE_RAD_ZABER * ZABER_TO_MM

# Arena Map based on 20240104
def arena_map():
    filename = os.path.join(DIR_ROOT, 'data/zaber_centers_20240104.csv')
    tile_centers = pd.read_csv(filename)
    mm_centers = tile_centers.copy()
    mm_centers.ls_guess = np.multiply(mm_centers.ls_guess, ZABER_TO_MM)
    mm_centers.ax_new_guess = np.multiply(mm_centers.ax_new_guess, ZABER_TO_MM)

    return (mm_centers.ls_guess.to_numpy(),
            mm_centers.ax_new_guess.to_numpy())

# CCF Map
# TODO: Unified Coordinate System
def ccf_map():
    filename = os.path.join(DIR_ROOT, 'data/zaber_ccf.csv')
    tile_centers = pd.read_csv(filename)
    mm_centers = tile_centers.copy()
    mm_centers.ls = np.multiply(mm_centers.ls, ZABER_TO_MM)
    mm_centers.ax3 = np.multiply(mm_centers.ax3, ZABER_TO_MM)
    return (mm_centers.ls.to_numpy(),
            mm_centers.ax3.to_numpy())

# Deprecated: Change to CCF
TILE_CENTER = arena_map()
ARENA_CENTER = (1140, 1200)

# tile index that defines the vertices of the arena
# in counter-clockwise order
VERT_TILE = [0, 34, 134, 156, 122, 22]

# base directories for videos
video_base = '/groups/dennis/dennislab/data/rig'
track_base = '/groups/dennis/dennislab/data/hs_cam'
hs_video = [x for x in os.listdir(track_base) if x.endswith('.mp4')]

# Organize files for different mice cohorts
def time_diff(time1, time2, format="%H_%M_%S"):
    # Convert time strings to datetime.time objects
    t1 = datetime.strptime(time1, format).time()
    t2 = datetime.strptime(time2, "%H_%M_%S").time()

    # Convert time objects to seconds since midnight
    seconds1 = t1.hour * 3600 + t1.minute * 60 + t1.second
    seconds2 = t2.hour * 3600 + t2.minute * 60 + t2.second

    # Calculate the difference in seconds
    return abs(seconds1 - seconds2)

def load_data(dict, base_dir):
    base_dir = os.path.join(DIR_ROOT, base_dir)
    fl_list = os.listdir(base_dir)
    fl_list.sort()
    for fl in fl_list:
        # extract date, time, and mice id from the file name
        # name format: 2024-02-22T09_46_32_ p16_all_params_file.csv
        date_str = fl[:10]
        time_str = fl[11:19]
        mice_str = fl[-23:-20]

        # find the folder with the same date
        rig_folder = os.path.join(video_base, date_str.replace('-', ''))

        # find low-res video: read all files in the folder with .avi extension
        rig_files = os.listdir(rig_folder)
        video_files = [x for x in rig_files if x.startswith('video_basler_') and x.endswith('.avi')]
        video_time = [time_diff(x[-12:-4], time_str) for x in video_files]
        video_file = video_files[np.argmin(video_time)]

        # find high-res video (for pose tracking)
        videos = [x for x in hs_video if date_str.replace('-', '') in x]
        video_time = [time_diff(x[9:15], time_str, format="%H%M%S") for x in videos]

        if len(video_time) == 0:
            hs_file = 'None'
        else:
            hs_file = videos[np.argmin(video_time)]

        # add file to list
        dict[mice_str].append((os.path.join(base_dir, fl),
                               os.path.join(rig_folder, video_file),
                               os.path.join(track_base, hs_file)))

# b12b13 (2023 Fall)
B_MICE = {'b12': [], 'b13': []}
base_dir = 'data/b12b13'
load_data(B_MICE, base_dir)

# p16p17p18 (2024 Spring)
P_MICE = {'p16': [], 'p17': [], 'p18': []}
base_dir = 'data/p16p17p18'
load_data(P_MICE, base_dir)

# p16p17p18 (2024 Fall)
P_MALE = {'p20': [], 'p21': []}
base_dir = 'data/p20p21'
load_data(P_MALE, base_dir)

ALL_MICE = {**B_MICE, **P_MICE, **P_MALE}