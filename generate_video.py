import pandas as pd
import numpy as np
from utils.data_loader import *
from utils.data_struct import *

# read in name and index from command line
NAME = 'b12'
SESS = [11, 12, 13, 14]

all_files = ALL_MICE[NAME]

all_data = []
for idx, file in enumerate(all_files):
    data_frame = pd.read_csv(file[0], low_memory=False)
    data = SessionData(NAME, idx, data_frame, file[1])
    all_data.append(data)

for ses_idx in SESS:

    data = all_data[ses_idx]
    length = int(data.video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('video frame', length)
    print('data length', *data.time.shape)
    print('cricket time', np.where(data.triggered == 1)[0])

    # generate video for the session
    print('gennrating video for session', ses_idx)
    data.all_video()
