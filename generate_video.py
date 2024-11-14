import pandas as pd
import numpy as np
import sys
from utils.data_loader import *
from utils.data_struct import *

# NAME = 'p20'
# SESS = [49, 50, 51, 52]

# read in name and session number
# from command line
name = sys.argv[1]
sess = [int(s) for s in sys.argv[2:]]

all_files = ALL_MICE[name]

all_data = []
for idx, file in enumerate(all_files):
    data_frame = pd.read_csv(file[0], low_memory=False)
    data = SessionData(name, idx, data_frame, file[1])
    all_data.append(data)

for ses_idx in sess:

    data = all_data[ses_idx]
    length = int(data.video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('video frame', length)
    print('data length', *data.time.shape)
    print('cricket time', np.where(data.triggered == 1)[0])

    # generate video for the session
    print('gennrating video for session', ses_idx)
    data.all_video()
