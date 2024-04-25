import pandas as pd
import numpy as np
import sys
from utils.data_loader import *
from utils.data_struct import *

# read in name and index from command line
NAME = sys.argv[1]
SES_IDX = int(sys.argv[2])

all_files = P_MICE[NAME]
all_data = []
for idx, file in enumerate(all_files):
    data_frame = pd.read_csv(file[0], low_memory=False)
    data = SessionData(NAME, idx, data_frame, file[1])
    all_data.append(data)

data = all_data[SES_IDX]
length = int(data.video.get(cv2.CAP_PROP_FRAME_COUNT))
print('video frame', length)
print('data length', *data.time.shape)
print('num cricket', np.where(data.triggered == 1)[0])

data.all_video()