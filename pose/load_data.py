import json
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

BASE_DIR = '/groups/zhang/home/zhangl5/Emily/Video_Process/training'

def load_apt(file_path):
    '''Load labeled data in coco format'''
    file_path = os.path.join(BASE_DIR, file_path)
    dataset = json.load(open(file_path, 'r'))['locdata']

    # Read in labeled data
    images = []
    points = []
    occlude = []
    for frame in dataset:
        img_path = os.path.join(BASE_DIR, frame['img'][0])
        images.append(plt.imread(img_path))

        points.append(frame['pabs'])
        occlude.append(frame['occ'])

    # Convert to numpy arrays
    # Ensure appropriate x - y arrangement
    points = np.array(points).T
    points = points.reshape((2, -1, points.shape[-1]))
    points = points.transpose((1, 0, 2)).reshape((-1, points.shape[-1]))

    return images, points, np.array(occlude).T

def load_pred(file_path):
    '''Load APT output data'''
    track_path = os.path.join(BASE_DIR, '.temp', file_path)

    # load tracking data
    tracking = scipy.io.loadmat(track_path)

    # tracking results in shape (n_points, (x, y), n_frame)
    points = tracking['points']
    points = points.reshape([-1, points.shape[-1]])

    return points
