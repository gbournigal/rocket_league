# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:42:07 2022

@author: georg
"""

import pandas as pd
import numpy as np
import itertools
import gc  # garbage collection
from lightgbm import LGBMClassifier
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV


# data loading
DEBUG = False
SAMPLE = 1
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
for i in range(10):
    df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=col_dtypes)
    if SAMPLE < 1:
        df_tmp = df_tmp.sample(frac=SAMPLE, random_state=SEED)
        
    df = pd.concat([df, df_tmp])
    del df_tmp
    gc.collect()
    if DEBUG:
        break
    

### Feature Engineer ###
def euclidian_norm(x):
    return np.linalg.norm(x, axis=1)


pos_groups = {
    f"{el}_pos": [f'{el}_pos_x', f'{el}_pos_y', f'{el}_pos_z']
    for el in ['ball'] + [f'p{i}' for i in range(6)]
}


for col, vec in pos_groups.items():
    df[col + "_ball_dist"] = euclidian_norm(df[vec].values - df[pos_groups["ball_pos"]].values)


# import math
# def calculate_distance_1(x1,y1,z1,x2,y2,z2):
#     d = 0.0
#     d+= (x1-x2)**2
#     d+= (y1-y2)**2
#     d+= (z1-z2)**2
#     return math.sqrt(d)

    
# for p in ['p0','p1','p2','p3','p4','p5']:
#     col1 = p+'_dist_B_goal'
#     col2 = p+'_dist_A_goal'
#     p_x = p+'_pos_x'
#     p_y = p+'_pos_y'
#     p_z = p+'_pos_z'
#     df[col1] = df.apply(lambda x: calculate_distance_1(x[p_x], x[p_y], x[p_z], 0, -100, 6.8), axis=1)
#     df[col2] = df.apply(lambda x: calculate_distance_1(x[p_x], x[p_y], x[p_z], 0, 100, 6.8), axis=1)
    
    
cols_to_drop = [
    'game_num', 'event_id', 'event_time', 'player_scoring_next', 'team_scoring_next'
]

### Hypertune ###


df = df.drop(columns=cols_to_drop)
df = df.dropna(axis=0)
gc.collect()


param_distribs = {
    "objective": ["binary"],
    "num_leaves": range(20, 200, 20),
    "n_estimators": range(100, 1000, 50),
    "learning_rate": uniform(0.01, 0.29),
    "max_depth": range(3, 12),
    "feature_fraction": uniform(0.2, 0.75),
    "subsample": [0.7],
    "subsample_freq": [8],
    "n_jobs": [-1],
    "reg_alpha": [1],
    "reg_lambda": [2],
    "min_child_samples": [90]    
}

lgbm = LGBMClassifier()
rnd_search = RandomizedSearchCV(
    lgbm,
    param_distributions=param_distribs,
    n_iter=100,
    cv=5,
    scoring="neg_log_loss",
    verbose=2,
    random_state=1,
    n_jobs=-1,
    return_train_score=True
)
rnd_search.fit(df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values, 
               df['team_A_scoring_within_10sec'].values)

# Train
params = {
    'objective': 'binary',
    'num_leaves': 140, # was 128
    'n_estimators': 1000,
    'max_depth': 10, # was 10
    'learning_rate': 0.014045956, # was 0.1
    'feature_fraction': 0.7083506685757333, # was 0.75
    'subsample': 0.7,
    'subsample_freq': 8,
    'n_jobs': -1,
    'reg_alpha': 1,
    'reg_lambda': 2,
    'min_child_samples': 90,
}

model_a = LGBMClassifier(**params)
model_a.fit(
    X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
    y=df['team_A_scoring_within_10sec'].values
    )




model_b = LGBMClassifier(**params)
# from sklearn.model_selection import cross_validate

# cv_results = cross_validate(model_b, 
#                             X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
#                             y=df['team_B_scoring_within_10sec'].values, 
#                             cv=5,
#                             n_jobs=-1,
#                             scoring="neg_log_loss",
#                             return_train_score=True
#                             )

model_b.fit(
    X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
    y=df['team_B_scoring_within_10sec'].values
    )


df_test = pd.read_csv('data/test.csv')
for col, vec in pos_groups.items():
    df_test[col + "_ball_dist"] = euclidian_norm(df_test[vec].values - df_test[pos_groups["ball_pos"]].values)


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

pred_a = np.zeros(df_test.shape[0])
pred_a_X = imp.fit_transform(df_test.drop(columns=['id']))
pred_a += model_a.predict_proba(pred_a_X)[:, 1]


pred_b = np.zeros(df_test.shape[0])
pred_b_X = imp.fit_transform(df_test.drop(columns=['id']))
pred_b += model_b.predict_proba(pred_b_X)[:, 1]



df_submission = pd.DataFrame(
    {
        "id": df_test['id'],
        "team_A_scoring_within_10sec": pred_a,
        "team_B_scoring_within_10sec": pred_b
    }
)

df_submission.to_csv('submission.csv', index=False)

