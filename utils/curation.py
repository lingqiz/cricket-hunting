from .data_loader import ALL_MICE
from .data_struct import SessionData
import pandas as pd
import numpy as np

MICE_HUNTING = {}
MICE_NOSOUND = {}

def load_data(list_name):
    '''
    Load data for a list of mice
    '''
    for name in list_name:
        all_data = []
        for idx, file in enumerate(ALL_MICE[name]):
            data_frame = pd.read_csv(file[0], low_memory=False)
            data = SessionData(name, idx, data_frame, *file[1:])
            all_data.append(data)

        # PWK female
        if name.startswith('p1'):
            idx_select = 15
            idx_nosound = -4

            # first 15 sessions
            select_data = all_data[:idx_select]

            # rest of the sessions with high catch number
            # + session 30/32 (2nd tile sound off), and 31
            for data in all_data[idx_select:]:
                if data.n_catch >= 4 or (data.session >= 30 and data.session <= 32):
                    select_data.append(data)

                # target tile correction for session 30
                if data.session == 30:
                    data.target[:, 6:] += np.array([55, 40]).reshape(2, 1)

            MICE_HUNTING[name] = select_data
            no_sound = all_data[idx_nosound:]
            MICE_NOSOUND[name] = no_sound

            print(f'{name}: {len(select_data)} hunting sessions, ' +
                f'{len(no_sound)} no sound sessions')

        # PWK male
        elif name.startswith('p2'):
            select_data = all_data[-12:-8] + all_data[-5:]
            MICE_NOSOUND[name] = select_data
            print(f'{name}: {len(select_data)} no sound sessions')

        # B6 mice
        elif name.startswith('b'):
            MICE_HUNTING[name] = all_data
            print(f'{name}: {len(select_data)} hunting sessions')

def load_all():
    '''
    Load data for all mice in the dataset
    '''
    load_data(['p16', 'p17', 'p18', 'b12', 'b13', 'p20', 'p21'])