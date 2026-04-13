import os, json
from .constants import *

# Base directory for consolidated data
DATA_BASE = '/groups/dennis/dennislab/data/new_format'

def _ts_to_dirname(ts):
    '''Convert data.json timestamp to directory name suffix.
    e.g. "2024-02-22T09_46_32" -> "2024-02-22-09-46-32"
    '''
    return ts.replace('T', '-').replace('_', '-')

def _build_data_index():
    '''Read data.json and build ALL_DATA[mouse][phase] = [session_dir, ...].
    data.json is organized by animal name; each entry is the full path to
    a session directory.
    '''
    with open(os.path.join(DATA_BASE, 'data.json')) as f:
        data_json = json.load(f)

    all_data = {}
    for mouse, phases in data_json.items():
        for phase, timestamps in phases.items():
            for ts in timestamps:
                dir_path = os.path.join(DATA_BASE, mouse,
                                        f'{mouse}_{_ts_to_dirname(ts)}')
                all_data.setdefault(mouse, {}).setdefault(phase, []).append(dir_path)

    return all_data


# Build the ALL_DATA index at module load time
ALL_DATA = _build_data_index()

def get_animals():
    '''Return list of all animal names.'''
    return list(ALL_DATA.keys())

def get_session_types(animal_name):
    '''Return list of session types available for a given animal.'''
    return list(ALL_DATA[animal_name].keys())

def load_session(animal_name, session_type, session_index):
    '''Load a single SessionData object.'''
    from .data_struct import SessionData
    session_dir = ALL_DATA[animal_name][session_type][session_index]
    return SessionData(animal_name, session_type, session_index, session_dir)

def load_sessions(animal_name, session_type):
    '''Load SessionData objects for a given animal (or list of animals)
    and session type.

    If animal_name is a string, returns a list of SessionData.
    If animal_name is a list, returns a dict of {name: [SessionData, ...]}.
    '''
    from .data_struct import SessionData
    if isinstance(animal_name, list):
        return {name: load_sessions(name, session_type)
                for name in animal_name}

    return [SessionData(animal_name, session_type, idx, session_dir)
            for idx, session_dir in enumerate(ALL_DATA[animal_name][session_type])]
