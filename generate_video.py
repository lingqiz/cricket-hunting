import pandas as pd
import numpy as np
import sys
from utils.data_loader import *
from utils.data_struct import *

# read in name and session number from command line
# example: python3 generate_video.py p16 -1 12 39 40
name = sys.argv[1]
eos = int(sys.argv[2]) == 1
sess = [int(s) for s in sys.argv[3:]]

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
    # set eos to True to include the end of session
    print('gennrating video for session', ses_idx)
    data.all_video(max_frame=64800, eos=eos)
