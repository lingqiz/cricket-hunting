import cv2
import sys, os
import numpy as np
import scipy.io
import argparse
from matplotlib import colormaps
from simple_term_menu import TerminalMenu
from utils.data_loader import load_session, get_animals, get_session_types, ALL_DATA

'''
Play back video with pose tracking overlay.

Example usage:
  python3 play_track.py -animal p16 -type hunting -session 0
  python3 play_track.py -mode dual -animal b12 -type exploration -session 2

Controls: p=pause, q=quit, w/s=speed up/down, d/a=skip forward/backward
'''

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
        if conf[i] >= 0.25:
            # Draw horizontal line of the cross with alpha based on confidence
            cv2.line(frame, (x - size, y), (x + size, y), colors[i], thickness)

            # Draw vertical line of the cross
            cv2.line(frame, (x, y - size), (x, y + size), colors[i], thickness)

    return frame

def annotate_hs_frame(frame, hs_frame_idx, session_data):
    '''Draw pose keypoints and behavior scores on a high-speed camera frame.'''
    text_color = (62, 176, 69)

    # draw pose keypoints
    points = session_data.keypoints[:, hs_frame_idx].reshape(-1, 2)
    conf = session_data.track_conf[:, hs_frame_idx]
    frame = draw_cross(frame, points, conf)

    # draw behavior scores
    for i, name in enumerate(sorted(session_data.scores.keys())):
        score_arr = session_data.scores[name]['scores']
        if hs_frame_idx < len(score_arr):
            val = score_arr[hs_frame_idx]
            active = session_data.scores[name]['postprocessed'][hs_frame_idx]
            label_color = (0, 200, 255) if active else text_color
            cv2.putText(frame, f"{name}: {val:.2f}",
                        (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

    return frame

# video playback
def hs_player(session_data):
    cap = cv2.VideoCapture(session_data.hs_path)

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
                break

            frame = annotate_hs_frame(frame, frame_idx, session_data)

            # Write frame number and playback speed
            frame_text = f"Frame: {frame_idx}"
            speed_text = f"Speed: {1 / (delay / 1000):.2f} FPS"
            cv2.putText(frame, frame_text, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(frame, speed_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            cv2.imshow('Pose Tracking', frame)
            frame_idx += 1

        # Keyboard controls
        key = cv2.waitKey(delay)

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
            frame_idx += 120 * 60
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == ord('w'):
            delay = delay // 2
        elif key == ord('s'):
            delay = delay * 2

    cap.release()
    cv2.destroyAllWindows()

def dual_player(session_data):
    text_color = (62, 176, 69)

    frame_idx = 0
    ref_frame = session_data.get_frame(frame_idx)

    separator_width = 10
    separator = np.full((ref_frame.shape[0], separator_width, 3),
                        (128, 128, 128), dtype=np.uint8)

    paused = False
    while True:

        if not paused:
            low_res = session_data.get_frame(frame_idx)
            high_res = session_data.hs_frame(frame_idx)

            if np.sum(high_res) > 0:
                hs_idx = session_data.hs_index[frame_idx]
                high_res = annotate_hs_frame(high_res, hs_idx, session_data)

            high_res = cv2.resize(high_res, (low_res.shape[1], low_res.shape[0]))

            # Combine low-res and high-res frames side-by-side
            low_res = cv2.flip(low_res, -1)
            combined_frame = np.hstack((low_res, separator, high_res))

            # Add text
            frame_text = f"Frame: {frame_idx}"
            time_text = f"Time: {session_data.time[frame_idx]:.2f} sec"
            cv2.putText(combined_frame, frame_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(combined_frame, time_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            cv2.imshow('Pose Tracking', combined_frame)
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

def pick_from_menu(title, options):
    '''Arrow-key menu. Returns (index, selected_option) or exits on abort.'''
    menu = TerminalMenu(options, title=title)
    idx = menu.show()
    if idx is None:
        print('Cancelled.')
        sys.exit(0)
    return idx, options[idx]

def interactive_menu(mode):
    '''Walk through animal -> session type -> session selection.'''
    animals = get_animals()
    _, animal = pick_from_menu('Select animal:', animals)

    types = get_session_types(animal)
    _, session_type = pick_from_menu('Select session type:', types)

    session_dirs = ALL_DATA[animal][session_type]
    labels = []
    for i, d in enumerate(session_dirs):
        dt_str = os.path.basename(d).split('_', 1)[1]  # "2024-03-12-12-04-01"
        date = dt_str[:10]                               # "2024-03-12"
        time = dt_str[11:].replace('-', ':')             # "12:04:01"
        labels.append(f'session {i + 1}  ({date} {time})')
    session_idx, _ = pick_from_menu('Select session:', labels)

    print(f'\nLoading {animal} / {session_type} / session {session_idx + 1} ...')
    return animal, session_type, session_idx, mode

# Setup arguments
parser = argparse.ArgumentParser(description='Play video tracking data')
parser.add_argument('-mode', type=str, default='dual')
parser.add_argument('-animal', type=str, default=None)
parser.add_argument('-type', type=str, default=None)
parser.add_argument('-session', type=int, default=None)

args = parser.parse_args()

# use interactive menu or command line arguments to select session
if args.animal is None or args.type is None or args.session is None:
    animal, session_type, session_idx, mode = interactive_menu(args.mode)
else:
    animal, session_type, session_idx, mode = (
        args.animal, args.type, args.session, args.mode)

# load session data and play video
session_data = load_session(animal, session_type, session_idx)
if not session_data.has_hs:
    print(f'No high-speed video available for {animal} / {session_type} / session {session_idx}.')
    sys.exit(1)

session_data._load_pose()
if mode == 'hs':
    hs_player(session_data)
elif mode == 'dual':
    dual_player(session_data)