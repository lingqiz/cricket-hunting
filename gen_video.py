import numpy as np
import argparse
from utils.data_loader import *
from utils.data_struct import *
from utils.curation import *

'''
This script generates video from session(s) of a mouse hunting.
Example usage:
python3 gen_video.py --name p16 --eos True --session 0 1 2
python3 gen_video.py --name p20 --session 10 11
'''

# setup argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str)
parser.add_argument("--eos", type=bool, default=False)
parser.add_argument("--nosound", type=bool, default=False)
parser.add_argument("--session", nargs='+', type=int)

# read in arguments
args = parser.parse_args()
name, eos, nosound, sess = (args.name, args.eos, args.nosound, args.session)

# print configuration
print(f"Generating video for {name}, No sound: {nosound} \
      Session(s): {sess}, Include end of session: {eos}")

load_data([args.name])
if nosound:
    all_session = MICE_NOSOUND[args.name]
else:
    all_session = MICE_HUNTING[args.name]

for ses_index in sess:
    data = all_session[ses_index]
    data._load_pose()

    length = int(data.video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('video frame', length)
    print('data length', *data.time.shape)
    print('cricket time', np.where(data.triggered == 1)[0])

    # generate video for the session
    # set eos to True to include the end of session
    print('gennrating video for session', ses_index)
    data.all_video(max_frame=64800, eos=eos)