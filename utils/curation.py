from .data_loader import ALL_MICE
from .data_struct import SessionData
import pandas as pd

MICE_HUNTING = {}
MICE_NOSOUND = {}

def load_data(list_name):
    for name in list_name:
        all_data = []
        for idx, file in enumerate(ALL_MICE[name]):
            data_frame = pd.read_csv(file[0], low_memory=False)
            data = SessionData(name, idx, data_frame, file[1], file[2])
            all_data.append(data)

        # PWK mice
        if name.startswith('p1'):
            idx_select = 15
            idx_nosound = -4

            # first 15 sessions
            select_data = all_data[:idx_select]

            # rest of the sessions with high catch number
            for data in all_data[idx_select:]:
                if data.n_catch >= 4:
                    select_data.append(data)

            MICE_HUNTING[name] = select_data
            no_sound = all_data[idx_nosound:]
            MICE_NOSOUND[name] = no_sound

            print(f'{name}: {len(select_data)} hunting sessions, ' +
                f'{len(no_sound)} no sound sessions')

        # PWK male
        if name.startswith('p2'):
            select_data = all_data[-12:-8] + all_data[-5:]
            MICE_NOSOUND[name] = select_data
            print(f'{name}: {len(select_data)} no sound sessions')

        # B6 mice
        elif name.startswith('b'):
            MICE_HUNTING[name] = all_data
            print(f'{name}: {len(select_data)} hunting sessions')

def load_all():
    load_data(['p16', 'p17', 'p18', 'b12', 'b13', 'p20', 'p21'])