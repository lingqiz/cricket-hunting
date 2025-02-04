import cv2
import numpy as np
from matplotlib import colormaps
from utils.curation import *
load_data(['p16', 'p18'])

# set up color map
num_points = 37
colors = []
cmap = colormaps.get_cmap('Spectral')
for i in range(num_points):
    r, g, b, _ = cmap(i / (num_points - 1))
    colors.append((b * 255, g * 255, r * 255))

# draw tracking points
def draw_cross(frame, points, size=4, thickness=2):
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

        # Draw horizontal line of the cross
        cv2.line(frame, (x - size, y), (x + size, y), colors[i], thickness)

        # Draw vertical line of the cross
        cv2.line(frame, (x, y - size), (x, y + size), colors[i], thickness)

    return frame

# video playback
def video_player(video_path, pose_data):
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
            draw_cross(frame, points)

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


# load session data and play video
ses_id = 12
session = MICE_HUNTING['p16']

session_data = session[ses_id]
session_data._load_pose()

# play video
video_player(session_data.hs_path,
             session_data.pose)