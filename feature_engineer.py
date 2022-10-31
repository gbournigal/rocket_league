# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:57:17 2022

@author: gbournigal
"""

import numpy as np
import math

cols_to_drop = [
    'game_num', 
    'event_id', 
    'event_time', 
    'player_scoring_next', 
    'team_scoring_next',
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
        if col != 'ball_pos':
            df[col + "_ball_dist"] = euclidian_norm(df[vec].values - df[pos_groups["ball_pos"]].values)
        if col not in ['goal_A_pos', 'goal_B_pos', 'ball_pos']:
            df[col + "_goal_A_dist"] = euclidian_norm(df[vec].values - df[pos_groups["goal_A_pos"]].values)
            df[col + "_goal_B_dist"] = euclidian_norm(df[vec].values - df[pos_groups["goal_B_pos"]].values)
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


def min_dist_to_goal(df):
    # Team A
    df['min_dist_to_goal1_A'] = df[[f'p{i}_pos_goal_A_dist' for i in range(3)]].min(axis=1)
    df['min_dist_to_goal2_A'] = df[[f'p{i}_pos_goal_B_dist' for i in range(3)]].min(axis=1)
    
    # Team B
    df['min_dist_to_goal1_B'] = df[[f'p{i}_pos_goal_A_dist' for i in range(3,6)]].min(axis=1)
    df['min_dist_to_goal2_B'] = df[[f'p{i}_pos_goal_B_dist' for i in range(3,6)]].min(axis=1)
    return df

def max_dist_to_goal(df):
    # Team A
    df['max_dist_to_goal1_A'] = df[[f'p{i}_pos_goal_A_dist' for i in range(3)]].max(axis=1)
    df['max_dist_to_goal2_A'] = df[[f'p{i}_pos_goal_B_dist' for i in range(3)]].max(axis=1)
    
    # Team B
    df['max_dist_to_goal1_B'] = df[[f'p{i}_pos_goal_A_dist' for i in range(3,6)]].max(axis=1)
    df['max_dist_to_goal2_B'] = df[[f'p{i}_pos_goal_B_dist' for i in range(3,6)]].max(axis=1)
    return df

def mean_dist_to_goal(df):
    # Team A
    
    df['mean_dist_to_goal1_A'] = df[[f'p{i}_pos_goal_A_dist' for i in range(3)]].mean(axis=1)
    df['mean_dist_to_goal2_A'] = df[[f'p{i}_pos_goal_B_dist' for i in range(3)]].mean(axis=1)
    
    # Team B
    df['mean_dist_to_goal1_B'] = df[[f'p{i}_pos_goal_A_dist' for i in range(3,6)]].mean(axis=1)
    df['mean_dist_to_goal2_B'] = df[[f'p{i}_pos_goal_B_dist' for i in range(3,6)]].mean(axis=1)
    return df


def add_angle_features(df):
    # Goal Line Angle
    df['A_goal_angle'] = -1
    df['B_goal_angle'] = -1    
    ball_point_tpls = [tuple(x) for x in df[['ball_pos_x', 'ball_pos_y']].to_numpy()]

    ## Team A
    a_angle, b_angle = [], []
    for ball_pos in ball_point_tpls:
        # A
        vec1 = (-16.37 - ball_pos[0], 100 - ball_pos[1])
        vec2 = (16.47 - ball_pos[0], 100 - ball_pos[1])

        dot_value = vec1[0]*vec2[0]+vec1[1]*vec2[1]
        cos_angle = dot_value / (np.sqrt(vec1[0]**2+vec1[1]**2)*np.sqrt(vec2[0]**2+vec2[1]**2))
        angle = math.acos(cos_angle) * (180.0 / math.pi)
        a_angle.append(angle)

        # B
        vec1 = (-16.37 - ball_pos[0], -100 - ball_pos[1])
        vec2 = (16.47 - ball_pos[0], -100 - ball_pos[1])

        dot_value = vec1[0]*vec2[0]+vec1[1]*vec2[1]
        cos_angle = dot_value / (np.sqrt(vec1[0]**2+vec1[1]**2)*np.sqrt(vec2[0]**2+vec2[1]**2))
        angle = math.acos(cos_angle) * (180.0 / math.pi)
        b_angle.append(angle)

    df['A_goal_angle'] = a_angle
    df['B_goal_angle'] = b_angle
    return df

# def feature_scaling(df):
#     for feature in df.columns:
#         if feature.endswith('_x'):
#             df[feature] = (df[feature] / 82).astype('float16')
#         if feature.endswith('_y'):
#             df[feature] = (df[feature] / 120).astype('float16')
#         if feature.endswith('_z'):
#             df[feature] = (df[feature] / 40).astype('float16')
#         if feature.endswith('_boost'):
#             df[feature] = (df[feature] / 100).astype('float16')
#         if feature.endswith('_timer'):
#             df[feature] = (-df[feature] / 100).astype('float16')
#     return df
