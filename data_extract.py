# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:42:07 2022

@author: georg
"""

import pandas as pd
import numpy as np
import itertools
import gc  # garbage collection

# data loading
DEBUG = False
SAMPLE = 0.01
SEED = 42

col_dtypes = {
    'game_num': 'int8', 'event_id': 'int8', 'event_time': 'float16',
    'ball_pos_x': 'float16', 'ball_pos_y': 'float16', 'ball_pos_z': 'float16',
    'ball_vel_x': 'float16', 'ball_vel_y': 'float16', 'ball_vel_z': 'float16',
    'p0_pos_x': 'float16', 'p0_pos_y': 'float16', 'p0_pos_z': 'float16',
    'p0_vel_x': 'float16', 'p0_vel_y': 'float16', 'p0_vel_z': 'float16',
    'p0_boost': 'float16', 'p1_pos_x': 'float16', 'p1_pos_y': 'float16',
    'p1_pos_z': 'float16', 'p1_vel_x': 'float16', 'p1_vel_y': 'float16',
    'p1_vel_z': 'float16', 'p1_boost': 'float16', 'p2_pos_x': 'float16',
    'p2_pos_y': 'float16', 'p2_pos_z': 'float16', 'p2_vel_x': 'float16',
    'p2_vel_y': 'float16', 'p2_vel_z': 'float16', 'p2_boost': 'float16',
    'p3_pos_x': 'float16', 'p3_pos_y': 'float16', 'p3_pos_z': 'float16',
    'p3_vel_x': 'float16', 'p3_vel_y': 'float16', 'p3_vel_z': 'float16',
    'p3_boost': 'float16', 'p4_pos_x': 'float16', 'p4_pos_y': 'float16',
    'p4_pos_z': 'float16', 'p4_vel_x': 'float16', 'p4_vel_y': 'float16',
    'p4_vel_z': 'float16', 'p4_boost': 'float16', 'p5_pos_x': 'float16',
    'p5_pos_y': 'float16', 'p5_pos_z': 'float16', 'p5_vel_x': 'float16',
    'p5_vel_y': 'float16', 'p5_vel_z': 'float16', 'p5_boost': 'float16',
    'boost0_timer': 'float16', 'boost1_timer': 'float16', 'boost2_timer': 'float16',
    'boost3_timer': 'float16', 'boost4_timer': 'float16', 'boost5_timer': 'float16',
    'player_scoring_next': 'O', 'team_scoring_next': 'O', 'team_A_scoring_within_10sec': 'O',
    'team_B_scoring_within_10sec': 'O'
}
cols = list(col_dtypes.keys())

path_to_data = 'data'
df = pd.DataFrame({}, columns=cols)
for i in range(2):
    df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=col_dtypes)
    if SAMPLE < 1:
        df_tmp = df_tmp.sample(frac=SAMPLE, random_state=SEED)
        
    df = pd.concat([df, df_tmp])
    del df_tmp
    gc.collect()
    if DEBUG:
        break
    

def euclidian_norm(x):
    return np.linalg.norm(x, axis=1)


pos_groups = {
    f"{el}_pos": [f'{el}_pos_x', f'{el}_pos_y', f'{el}_pos_z']
    for el in ['ball'] + [f'p{i}' for i in range(6)]
}


for col, vec in pos_groups.items():
    df[col + "_ball_dist"] = euclidian_norm(df[vec].values - df[pos_groups["ball_pos"]].values)