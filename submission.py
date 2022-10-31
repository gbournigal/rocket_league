# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:18:43 2022

@author: gbournigal
"""

import pickle
import pandas as pd
import numpy as np
from feature_engineer import distances, demolitions, calc_speeds, min_dist_to_goal, add_angle_features, mean_dist_to_goal, max_dist_to_goal

### TEST ###
models = pickle.load(open('results/final_models.pickle', 'rb'))

model_a = models['model_a']
model_b = models['model_b']

df_test = pd.read_csv('data/test.csv')
df_test = distances(df_test)
df_test = demolitions(df_test)
df_test = calc_speeds(df_test)
df_test = min_dist_to_goal(df_test)
df_test = add_angle_features(df_test)
df_test = max_dist_to_goal(df_test)
df_test = mean_dist_to_goal(df_test)

df_test = df_test.drop(columns=[
                                'goal_A_pos_x',
                                'goal_B_pos_x',
                                'goal_A_pos_y',
                                'goal_B_pos_y',
                                'goal_A_pos_z',
                                'goal_B_pos_z'])


# from sklearn.impute import SimpleImputer
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')

pred_a = np.zeros(df_test.shape[0])
# pred_a_X = imp.fit_transform(df_test.drop(columns=['id']))
pred_a_X =df_test.drop(columns=['id'])
pred_a += model_a.predict_proba(pred_a_X)[:, 1]


pred_b = np.zeros(df_test.shape[0])
# pred_b_X = imp.fit_transform(df_test.drop(columns=['id']))
pred_b_X = df_test.drop(columns=['id'])
pred_b += model_b.predict_proba(pred_b_X)[:, 1]



df_submission = pd.DataFrame(
    {
        "id": df_test['id'],
        "team_A_scoring_within_10sec": pred_a,
        "team_B_scoring_within_10sec": pred_b
    }
)

df_submission.to_csv('submission.csv', index=False)
