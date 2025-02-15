import cv2
import sys, os
import numpy as np
import scipy.io
import argparse
from matplotlib import colormaps
from utils.curation import *

BASE_PATH = '/groups/dennis/dennislab/data/hs_cam'

# set up color map
num_points = 37
colors = []
cmap = colormaps.get_cmap('Spectral')
for i in range(num_points):
    r, g, b, _ = cmap(i / (num_points - 1))
    colors.append((b * 255, g * 255, r * 255))

# draw tracking points
def draw_cross(frame, points, conf, size=4, thickness=2):
    """
    Draw a `+` cross marker at each point instead of a circle.

    Args:
        frame: The video frame to annotate.
        points: List of (x, y) coordinates for annotations.
        size: Length of the cross arms.
        color: BGR color (default is red).
        thickness: Line thickness.
    """
    for i, (x, y) in enumerate(points):
        x, y = int(x), int(y)
        if conf[i] >= 0.0:
            # Draw horizontal line of the cross with alpha based on confidence
            cv2.line(frame, (x - size, y), (x + size, y), colors[i], thickness)

            # Draw vertical line of the cross
            cv2.line(frame, (x, y - size), (x, y + size), colors[i], thickness)

    return frame

# video playback
def hs_player(video_path, pose_data, pose_conf):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    paused = False
    delay = int(1/120 * 1000)
    text_color = (62, 176, 69)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break  # Exit loop when video ends

            # Draw pose on the frame
            points = pose_data[:, frame_idx].reshape(-1, 2)
            conf = pose_conf[:, frame_idx]
            frame = draw_cross(frame, points, conf)

            # Write frame number and playback speed on the frame
            frame_text = f"Frame: {frame_idx}"
            speed_text = f"Speed: {1 / (delay / 1000):.2f} FPS"
            cv2.putText(frame, frame_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(frame, speed_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Display the frame
            cv2.imshow('Pose Tracking', frame)
            frame_idx += 1

        # Keyboard controls
        key = cv2.waitKey(delay)  # Delay in milliseconds (adjust based on FPS)

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused

        elif key == ord('d'):
            frame_idx += 120
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == ord('a'):
            frame_idx -= 120
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == ord('l'):
            frame_idx += 120 * 60 # Skip 1 minute
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        elif key == ord('w'):
            delay = delay // 2
        elif key == ord('s'):
            delay = delay * 2

    cap.release()
    cv2.destroyAllWindows()

def dual_player(session_data):
    text_color = (62, 176, 69)
    pose_data = session_data.keypoints
    pose_conf = session_data.track_conf

    frame_idx = 0
    ref_frame = session_data.get_frame(frame_idx)

    separator_width = 10  # Adjust this for more/less separation
    separator = np.full((ref_frame.shape[0], separator_width, 3),
                        (128, 128, 128), dtype=np.uint8)

    paused = False
    while True:

        if not paused:
            low_res = session_data.get_frame(frame_idx)
            high_res = session_data.hs_frame(frame_idx)

            if high_res is not None:
                # Draw pose on the frame
                pose_idx = session_data.hs_index[frame_idx]
                points = pose_data[:, pose_idx].reshape(-1, 2)
                conf = pose_conf[:, pose_idx]
                high_res = draw_cross(high_res, points, conf)
                high_res = cv2.resize(high_res, (low_res.shape[1], low_res.shape[0]))

            else:
                # Empty frame
                high_res = np.zeros_like(low_res)

            # Combine low-res and high-res frames side-by-side
            low_res = cv2.flip(low_res, -1)
            combined_frame = np.hstack((low_res, separator, high_res))

            # Add text
            frame_text = f"Frame: {frame_idx}"
            time_text = f"Time: {session_data.time[frame_idx]:.2f} sec"
            cv2.putText(combined_frame, frame_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(combined_frame, time_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            cv2.imshow('Pose Tracking', combined_frame)
            # Next frame
            frame_idx += 1

        # Delay in milliseconds (adjust based on FPS)
        key = cv2.waitKey(int((session_data.time[frame_idx + 1] -
                          session_data.time[frame_idx]) * 1000 / 4))

        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('d'):
            frame_idx += 50

# Setup arguments
parser = argparse.ArgumentParser(description='Play video tracking data')
parser.add_argument('-mode', type=str, default='hs')
parser.add_argument('-file_path', type=str, default=None)
parser.add_argument('-animal', type=str, default=None)
parser.add_argument('-session', type=int, default=None)

args = parser.parse_args()

if args.mode == 'hs':

    if args.file_path is not None:
        file_path = args.file_path
        video_path = os.path.join(BASE_PATH, file_path + '.mp4')

        # load tracking data
        track_path = os.path.join(BASE_PATH, file_path + '.mat')
        tracking = scipy.io.loadmat(track_path)

        # (n_points, (x, y), n_frame)
        points = tracking['points']
        points = points.reshape([-1, points.shape[-1]])
        track_conf = tracking['conf'] - 1.0

    else:
        load_data([args.animal])

        # load session data and play video
        session = MICE_HUNTING[args.animal]

        session_data = session[args.session]
        session_data._load_pose()

        video_path = session_data.hs_path
        points = session_data.keypoints
        track_conf = session_data.track_conf

    # play video
    hs_player(video_path, points, track_conf)

# dual video player
elif args.mode == 'dual':
    load_data([args.animal])

    # load session data and play video
    session = MICE_HUNTING[args.animal]

    session_data = session[args.session]
    session_data._load_pose()

    dual_player(session_data)