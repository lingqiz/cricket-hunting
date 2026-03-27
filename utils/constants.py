import numpy as np
import pandas as pd
import math, os

import pathlib
DIR_ROOT = pathlib.Path(__file__).parent.parent.resolve()

# Unit Conversions
ZEBER_TO_DLC = 1896 / 72248
ZABER_TO_MM = 508 / 72248
DLC_TO_MM = ZABER_TO_MM / ZEBER_TO_DLC
TRK_CTR = 948
TRIG_RADIUS_ZABER = 5500
TRIG_RADIUS = TRIG_RADIUS_ZABER * ZABER_TO_MM

# 30 sec ISI
ISI = 30

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
NAME_DICT = {idx: name for idx, name in enumerate(TILE_NAMES)}

def tile_angle():
    '''
    Angle of the hex tiles as defined in:
    https://github.com/the-dennis-lab/ejd_behavior_sandbox/blob/main/notebooks/arena_mapping.ipynb
    '''
    # two reference points
    p1 = (3736, 166646)
    p2 = (300372, 187681)

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

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
