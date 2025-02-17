import numpy as np
import pandas as pd
import math, os
import re
from datetime import datetime

import pathlib
DIR_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DIR_HOME = pathlib.Path.home()

# Unit Conversions
ZEBER_TO_DLC = 1896 / 72248
ZABER_TO_MM = 508 / 72248
DLC_TO_MM = ZABER_TO_MM / ZEBER_TO_DLC
TRK_CTR = 948
TRIG_RADIUS_ZABER = 5500
TRIG_RADIUS = TRIG_RADIUS_ZABER * ZABER_TO_MM

# Tile Names
TILE_NAMES = [
    "A1",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
    "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13",
    "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12",
    "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13",
    "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12",
    "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10", "K11", "K12", "K13",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7",
    "P1", "P2", "P3", "P4",
    "Q1"]

TILE_DICT = {name: idx for idx, name in enumerate(TILE_NAMES)}

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

# CCF Map
def ccf_map():
    filename = os.path.join(DIR_ROOT, 'data/zaber_ccf.csv')
    tile_centers = pd.read_csv(filename)
    mm_centers = tile_centers.copy()
    mm_centers.ls = np.multiply(mm_centers.ls, ZABER_TO_MM)
    mm_centers.ax3 = np.multiply(mm_centers.ax3, ZABER_TO_MM)
    return (mm_centers.ls.to_numpy(),
            mm_centers.ax3.to_numpy())

TILE_CENTER = ccf_map()
ARENA_CENTER = (1150, 1067)

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
        # name format: 2024-02-22T09_46_32_p16_all_params_file.csv
        date_str = fl[:10]
        time_str = fl[11:19]
        mice_str = fl[-23:-20]

        # find the folder with the same date
        rig_folder = os.path.join(video_base, date_str.replace('-', ''))

        # find low-res video: read all files in the folder with .avi extension
        rig_files = os.listdir(rig_folder)
        video_files = [x for x in rig_files if x.startswith('video_basler_')
                       and x.endswith('.avi')]
        video_time = [time_diff(x[-12:-4], time_str) for x in video_files]
        video_file = video_files[np.argmin(video_time)]

        # find calibrated tile name
        tile_files = [x for x in rig_files if x.startswith('location_inputs_')
                      and re.search(r'\d{2}\.csv$', x)]
        tile_time = [time_diff(x[-12:-4], time_str) for x in tile_files]
        tile_file = tile_files[np.argmin(tile_time)]

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
                               os.path.join(track_base, hs_file),
                               os.path.join(rig_folder, tile_file)))

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