# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:57:17 2022

@author: gbournigal
"""

import numpy as np


cols_to_drop = [
    'game_num', 
    'event_id', 
    'event_time', 
    'player_scoring_next', 
    'team_scoring_next', 
    'ball_pos_ball_dist',
    'goal_A_pos_x',
    'goal_B_pos_x',
    'goal_A_pos_y',
    'goal_B_pos_y',
    'goal_A_pos_z',
    'goal_B_pos_z'
]

### Feature Engineer ###
def euclidian_norm(x):
    return np.linalg.norm(x, axis=1)


pos_groups = {
    f"{el}_pos": [f'{el}_pos_x', f'{el}_pos_y', f'{el}_pos_z']
    for el in ['ball', 'goal_A', 'goal_B'] + [f'p{i}' for i in range(6)]
}


def distances(df):
    df['goal_A_pos_x'] = 0
    df['goal_B_pos_x'] = 0
    df['goal_A_pos_y'] = -102.5
    df['goal_B_pos_y'] = 102.5
    df['goal_A_pos_z'] = -1.2
    df['goal_B_pos_z'] = 1.2
    
    for col, vec in pos_groups.items():
        df[col + "_ball_dist"] = euclidian_norm(df[vec].values - df[pos_groups["ball_pos"]].values)
    
    return df


def demolitions(df):
    for i in range(6):
        df[f'p{i}_demo'] = (df[f'p{i}_pos_x'].isna()).astype(int)
    df['active_players_A'] = 3-df['p0_demo']-df['p1_demo']-df['p2_demo']
    df['active_players_B'] = 3-df['p3_demo']-df['p4_demo']-df['p5_demo']
    df.drop(columns=['p0_demo',
                     'p1_demo',
                     'p2_demo',
                     'p3_demo',
                     'p4_demo',
                     'p5_demo',
                     ],
            inplace=True)
    return df


def calc_speeds(df):
    df['ball_speed'] = np.sqrt((df['ball_vel_x']**2)+(df['ball_vel_y']**2)+(df['ball_vel_z']**2))
    for i in range(6):
        df[f'p{i}_speed'] = np.sqrt((df[f'p{i}_vel_x']**2)+(df[f'p{i}_vel_y']**2)+(df[f'p{i}_vel_z']**2))
    return df
