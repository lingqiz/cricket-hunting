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

def score_to_color(val):
    '''Map value in [-1, 1] to BGR color: blue (-1) -> white (0) -> red (1).
    Also works for binary labels: 0 -> blue, 1 -> red.
    '''
    val = np.clip(val, -1, 1)
    if val < 0:
        # blue to white
        t = val + 1  # 0 to 1
        return (255, int(255 * t), int(255 * t))
    else:
        # white to red
        t = 1 - val  # 1 to 0
        return (int(255 * t), int(255 * t), 255)

def label_to_color(val):
    '''Map binary label to blue (0) or red (1).'''
    if val > 0:
        return (0, 0, 255)
    else:
        return (255, 0, 0)

def draw_score_strip(frame, score_arr, hs_frame_idx, y_offset,
                     is_label=False, strip_width=600, strip_height=20,
                     window=1200, border=2):
    '''Draw a rolling color strip for a behavior score.

    Shows a +-5 second window (1200 frames at 120fps) centered on
    the current frame. Each frame's value is mapped to a color.
    A vertical line marks the current frame position.
    '''
    h, w = frame.shape[:2]
    n_frames = len(score_arr)
    color_fn = label_to_color if is_label else score_to_color

    # window range
    half = window // 2
    start = hs_frame_idx - half

    # build the strip directly at target width
    strip = np.full((strip_height, strip_width, 3), 128, dtype=np.uint8)
    for px in range(strip_width):
        src_idx = start + int(px * window / strip_width)
        if 0 <= src_idx < n_frames:
            strip[:, px] = color_fn(score_arr[src_idx])

    # position: right-aligned, at y_offset from bottom
    x0 = w - strip_width - 10
    y0 = h - y_offset - strip_height

    # black border
    cv2.rectangle(frame,
                  (x0 - border, y0 - border),
                  (x0 + strip_width + border, y0 + strip_height + border),
                  (0, 0, 0), -1)

    # paste strip onto frame
    frame[y0:y0 + strip_height, x0:x0 + strip_width] = strip

    # draw current frame marker (center line)
    cx = x0 + strip_width // 2
    cv2.line(frame, (cx, y0 - border), (cx, y0 + strip_height + border),
             (255, 255, 255), 1)

    return frame

# toggle between 'score' and 'label' display
show_label = False

def outlined_text(frame, text, pos, scale, color, thickness=1):
    '''Draw text with a black outline for readability.'''
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def annotate_hs_frame(frame, hs_frame_idx, session_data):
    '''Draw pose keypoints and behavior score strips on a high-speed frame.'''
    text_color = (220, 220, 220)
    active_color = (0, 0, 255)

    # draw pose keypoints first (behind the strips)
    points = session_data.keypoints[:, hs_frame_idx].reshape(-1, 2)
    conf = session_data.track_conf[:, hs_frame_idx]
    frame = draw_cross(frame, points, conf)

    # draw behavior score strips on top
    strip_height = 20
    border = 2
    label_height = 22
    slot_height = strip_height + border * 2 + label_height
    behavior_names = sorted(session_data.scores.keys())
    field = 'label' if show_label else 'score'
    for i, name in enumerate(behavior_names):
        score_arr = session_data.scores[name][field]
        label_arr = session_data.scores[name]['label']
        y_offset = 10 + i * (slot_height + 4)
        frame = draw_score_strip(frame, score_arr, hs_frame_idx,
                                 y_offset, is_label=show_label,
                                 strip_height=strip_height)

        # label above the strip — red when behavior is active
        is_active = (hs_frame_idx < len(label_arr)
                     and label_arr[hs_frame_idx] > 0)
        color = active_color if is_active else text_color
        label_y = frame.shape[0] - y_offset - strip_height - border - 4
        label_x = frame.shape[1] - 610
        outlined_text(frame, name.capitalize(), (label_x, label_y),
                      0.6, color)

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
        elif key == ord('b'):
            global show_label
            show_label = not show_label

    cap.release()
    cv2.destroyAllWindows()

def dual_player(session_data):
    text_color = (220, 220, 220)

    frame_idx = 0
    ref_frame = session_data.get_frame(frame_idx)

    separator_width = 10
    separator = np.full((ref_frame.shape[0], separator_width, 3),
                        (128, 128, 128), dtype=np.uint8)

    paused = False
    speed = 1  # 1x = baseline speed (equivalent to old 4x real-time)
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

            # Add frame/time info to rig video (bottom-left, after flip)
            h = low_res.shape[0]
            outlined_text(low_res, f"Frame: {frame_idx}",
                          (10, h - 40), 0.6, text_color)
            outlined_text(low_res, f"Time: {session_data.time[frame_idx]:.2f} sec",
                          (10, h - 15), 0.6, text_color)
            outlined_text(low_res, f"Speed: {speed}x",
                          (10, h - 65), 0.6, text_color)

            combined_frame = np.hstack((low_res, separator, high_res))

            cv2.imshow('Pose Tracking', combined_frame)
            frame_idx += 1

        # Delay in milliseconds
        dt = session_data.time[frame_idx + 1] - session_data.time[frame_idx]
        delay = max(1, int(dt * 1000 / (4 * speed)))
        key = cv2.waitKey(delay)

        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('d'):
            frame_idx += 50
        if key == ord('a'):
            frame_idx = max(0, frame_idx - 50)
        if key == ord('w'):
            speed *= 2
        if key == ord('s'):
            speed = max(1, speed // 2)
        if key == ord('b'):
            global show_label
            show_label = not show_label

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