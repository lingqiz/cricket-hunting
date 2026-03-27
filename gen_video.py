import cv2
import numpy as np
import argparse
from utils.data_loader import load_sessions

'''
This script generates video from session(s) of a mouse hunting.
Example usage:
python3 gen_video.py --name p16 --type hunting --eos True --session 0 1 2
python3 gen_video.py --name p20 --type hunting --session 10 11
'''

# setup argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str)
parser.add_argument("--type", type=str, default='hunting')
parser.add_argument("--eos", type=bool, default=False)
parser.add_argument("--session", nargs='+', type=int)

# read in arguments
args = parser.parse_args()

# print configuration
print(f"Generating video for {args.name}, type: {args.type}, "
      f"Session(s): {args.session}, Include end of session: {args.eos}")

all_sessions = load_sessions(args.name, args.type)

for ses_index in args.session:
    data = all_sessions[ses_index]
    print(data.hs_path)
    data._load_pose()

    length = int(data.video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('video frame', length)
    print('data length', *data.time.shape)
    print('cricket time', np.where(data.triggered == 1)[0])

    # generate video for the session
    # set eos to True to include the end of session
    print('generating video for session', ses_index)
    data.all_video(max_frame=64800, eos=args.eos)