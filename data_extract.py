# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:42:07 2022

@author: georg
"""

import pandas as pd
import gc  # garbage collection


# data loading

def data_load(SAMPLE=1, SEED=42, df_size='full'):
    if df_size == 'full':
        rng_val = 10
    elif df_size == 'experimentation':
        rng_val = 9
    elif df_size == 'validation':
        rng_val_inf = 9
        rng_val_sup = 10
    else:
        raise Exception('Wrong parameter for df')
        
        
    
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
    if df_size in ['full', 'experimentation']:
        for i in range(rng_val):
            df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=col_dtypes)
            if SAMPLE < 1:
                df_tmp = df_tmp.sample(frac=SAMPLE, random_state=SEED)
                
            df = pd.concat([df, df_tmp])
            del df_tmp
            gc.collect()
    else:
        for i in range(rng_val_inf, rng_val_sup):
            df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=col_dtypes)
            if SAMPLE < 1:
                df_tmp = df_tmp.sample(frac=SAMPLE, random_state=SEED)
                
            df = pd.concat([df, df_tmp])
            del df_tmp
            gc.collect()
    return df















































# from xgboost import XGBClassifier
# from timeit import default_timer as timer
# df_tmp = df.sample(frac=0.05, random_state=SEED)

# classifier = XGBClassifier(n_jobs=-1)
# start = timer()
# model = classifier.fit(df_tmp.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
#                        df_tmp['team_B_scoring_within_10sec'].astype('int').values)
# print("without GPU:", timer()-start) 



# classifier = XGBClassifier(n_jobs=-1,
#                             tree_method='gpu_hist')
# start = timer()
# model = classifier.fit(df_tmp.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
#                         df_tmp['team_B_scoring_within_10sec'].astype('int').values)
# print("without GPU:", timer()-start) 


# classifier = LGBMClassifier(n_jobs=-1)
# start = timer()
# model = classifier.fit(df_tmp.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
#                        df_tmp['team_B_scoring_within_10sec'].astype('int').values)
# print("without GPU:", timer()-start) 




